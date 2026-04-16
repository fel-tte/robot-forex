"""
decision_engine.py
==================
DECISION ENGINE + CONTROL SYSTEM + PRODUCTION INFRASTRUCTURE

Điều khiển toàn bộ nhịp vận hành hệ thống. Hệ thống có thể:

  tự chọn việc  — decide_work()        : chọn LIVE / PAPER / LEARNING / PAUSED
  tự quyết định — decide_entry()       : kết hợp brain + predictor + learner
  tự mô phỏng   — run_paper_cycle()    : chạy không đặt lệnh thật
  tự hành động  — run_live_cycle()     : đặt lệnh thật
  tự sửa lỗi    — self_heal()          : retry, circuit breaker, mode fallback
  tự học        — trigger_learning()   : phân tích lịch sử, cập nhật params
  tự dự đoán    — predict_entry()      : win probability với confidence score
  tự scale      — self_scale()         : mở rộng / thu hẹp pool thị trường

Trạng thái hệ thống (lưu Redis, bền vững qua restart):
  LIVE      : Giao dịch thật với tiền thật
  PAPER     : Mô phỏng — không đặt lệnh thật
  PAUSED    : Tạm dừng (risk gate hoặc circuit breaker)
  LEARNING  : Đang chạy learning cycle

Bạn chỉ cần giám sát. Hệ thống tự quyết định tất cả.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

import redis

import config
from brain        import pick_best_entry
from risk_manager import RiskManager
from deriv_trade  import get_balance, place_and_wait
from logger       import TradeLogger, TradeRecord
from learner      import Learner
from predictor    import predict, Prediction
from simulator    import simulate
from memory       import MemoryBrain, TradeFeatures
from pipeline     import (
    TradeQueue, PermissionGate, LoadLimiter,
    PipelineMetrics, Orchestrator, QueuedTrade, TradeOutcome,
)
import deriv_data


# ──────────────────────────────────────────────────────────────────
# System state enum
# ──────────────────────────────────────────────────────────────────

class SystemMode(str, Enum):
    LIVE     = "LIVE"
    PAPER    = "PAPER"
    PAUSED   = "PAUSED"
    LEARNING = "LEARNING"


_REDIS_MODE_KEY       = "Deriv_System_Mode"
_REDIS_SCALE_KEY      = "Deriv_Active_Symbols"
_PAPER_LOG_KEY        = "Deriv_Paper_Log"
_MAX_CONSECUTIVE_ERRS = 5   # Lỗi liên tiếp trước khi self-heal


# ──────────────────────────────────────────────────────────────────
# Decision Engine
# ──────────────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Trung tâm điều phối toàn bộ hệ thống.

    Khởi động:
        engine = DecisionEngine()
        engine.run()   # vòng lặp vô tận
    """

    def __init__(self) -> None:
        self._r = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
        )
        self.risk    = RiskManager()
        self.logger  = TradeLogger()
        self.learner = Learner()

        # Redis Memory Brain — bộ não trung tâm ghi nhớ Win/Loss
        self.memory = MemoryBrain()

        # Pipeline components
        self._pipeline = Orchestrator(
            queue   = TradeQueue(),
            gate    = PermissionGate(),
            limiter = LoadLimiter(),
            metrics = PipelineMetrics(),
        )

        self._consecutive_errors = 0
        self._active_symbols     = self._load_active_symbols()
        self._mode               = self._load_mode()
        self._cycle_count        = 0

    # ── Persistence ───────────────────────────────────────────────

    def _load_mode(self) -> SystemMode:
        raw = self._r.get(_REDIS_MODE_KEY)
        if raw:
            try:
                return SystemMode(raw.decode())
            except Exception:
                pass
        return SystemMode.LIVE

    def _save_mode(self, mode: SystemMode) -> None:
        self._r.set(_REDIS_MODE_KEY, mode.value)
        self._mode = mode

    def _load_active_symbols(self) -> list[str]:
        raw = self._r.get(_REDIS_SCALE_KEY)
        if raw:
            try:
                syms = json.loads(raw)
                if isinstance(syms, list) and syms:
                    return syms
            except Exception:
                pass
        return list(config.SCAN_SYMBOLS)

    def _save_active_symbols(self) -> None:
        self._r.set(_REDIS_SCALE_KEY, json.dumps(self._active_symbols))

    # ── ① tự chọn việc ───────────────────────────────────────────

    def decide_work(self, balance: float) -> SystemMode:
        """
        Quyết định hệ thống nên làm gì trong chu kỳ này.

        Ưu tiên:
          PAUSED   : khi risk gate block giao dịch
          LEARNING : khi đủ history và đến kỳ học lại
          PAPER    : khi mode hiện tại là PAPER
          LIVE     : trạng thái bình thường
        """
        # Kiểm tra risk gate trước
        allowed, _ = self.risk.can_trade(balance=balance)
        if not allowed:
            return SystemMode.PAUSED

        # Kiểm tra có nên học lại không
        history_count = self._r.llen(config.REDIS_LOG_KEY)
        should_learn  = (
            history_count >= config.LEARNER_MIN_HISTORY
            and self._cycle_count > 0
            and self._cycle_count % config.LEARNER_INTERVAL_CYCLES == 0
        )
        if should_learn:
            return SystemMode.LEARNING

        # Giữ mode hiện tại (LIVE hoặc PAPER)
        return self._mode

    # ── ② tự quyết định ──────────────────────────────────────────

    def decide_entry(self,
                     signal,
                     df: Optional[object],
                     balance: float) -> Tuple[bool, Optional[Prediction]]:
        """
        Kết hợp brain score + predictor + learner để quyết định vào lệnh.

        Returns (should_enter: bool, prediction: Optional[Prediction])
        """
        wave_active = signal.wave.correction_active if signal.wave else False
        fib_zone    = signal.wave.fib_zone          if signal.wave else "NONE"

        # Kiểm tra điều kiện lịch sử yếu
        if self.learner.is_condition_weak(signal.score, fib_zone, wave_active):
            print("  [Engine] ⚠️  Điều kiện này có lịch sử yếu — bỏ qua lệnh.")
            return False, None

        # Chạy predictor
        if df is not None:
            pred = predict(signal, df, learner=self.learner, current_balance=balance)
            return pred.should_trade, pred

        # Fallback: chỉ dùng brain score
        params = self.learner.get_params()
        return signal.score >= params.effective_min_score, None

    # ── ③ tự mô phỏng ────────────────────────────────────────────

    def run_paper_cycle(self) -> None:
        """Chạy một chu kỳ mô phỏng — không đặt lệnh thật."""
        print("\n  📄 [PAPER] Đang mô phỏng...")
        try:
            best = pick_best_entry(symbols=self._active_symbols)
        except Exception as exc:
            print(f"  [PAPER] Scan thất bại: {exc}")
            return

        if best is None:
            print("  [PAPER] Không có tín hiệu đủ điều kiện.")
            return

        try:
            df = deriv_data.fetch_candles(symbol=best.symbol)
        except Exception:
            df = None

        balance = 100.0   # Balance ảo cho paper trading
        if df is not None:
            pred         = predict(best, df, learner=self.learner, current_balance=balance)
            stake        = pred.stake_suggestion or config.SIM_STAKE_USD
            pred_display = f"win_prob={pred.win_prob:.2f}  conf={pred.confidence:.2f}"
        else:
            stake        = config.SIM_STAKE_USD
            pred_display = "N/A"

        wave_active = bool(best.wave and best.wave.correction_active)
        fib_zone    = best.wave.fib_zone if best.wave else "NONE"

        print(
            f"  [PAPER] {best.symbol} {best.direction}  "
            f"score={best.score}  stake={stake:.2f}  {pred_display}"
        )

        # Ghi vào Redis paper log
        entry = {
            "timestamp"  : datetime.now().isoformat(),
            "symbol"     : best.symbol,
            "direction"  : best.direction,
            "score"      : best.score,
            "stake"      : stake,
            "wave_active": wave_active,
            "fib_zone"   : fib_zone,
            "pred"       : pred_display,
        }
        self._r.lpush(_PAPER_LOG_KEY, json.dumps(entry))
        self._r.ltrim(_PAPER_LOG_KEY, 0, 199)

    # ── ④ tự hành động ───────────────────────────────────────────

    def run_live_cycle(self, balance: float) -> None:
        """
        Chạy một chu kỳ giao dịch thật — sử dụng pipeline + Memory Brain.

        Chu kỳ chia thành 2 bước:
          A) SCAN + MEMORY CONSULT + SUBMIT: quét thị trường, tham vấn bộ nhớ,
             đưa tín hiệu đủ điều kiện vào hàng đợi
          B) DISPATCH: lấy từ hàng đợi, kiểm tra cổng quyền hạn + tải, thực thi

        Memory Brain là luật cứng: tham vấn BẮT BUỘC trước khi vào hàng đợi và
        một lần nữa qua Gate 4 trong dispatch.
        """
        ts = datetime.now()
        print(f"\n  [Pipeline] Scanning {len(self._active_symbols)} markets...")

        # ── Bước A: Quét thị trường, tạo tín hiệu ─────────────────
        try:
            best = pick_best_entry(symbols=self._active_symbols)
        except Exception as exc:
            print(f"  [LỖI] Scan thất bại: {exc}")
            self._consecutive_errors += 1
            return

        if best is not None:
            # Lấy dữ liệu nến để predictor phân tích
            try:
                df = deriv_data.fetch_candles(symbol=best.symbol)
            except Exception:
                df = None

            # Tạo TradeFeatures cho Memory Brain
            trade_features = MemoryBrain.features_from_signal(best)

            # ── Tham vấn Memory Brain trước khi xếp hàng (PRE-QUEUE) ──
            verdict = self.memory.consult(trade_features)
            print(
                f"  [Memory] 🧠 {verdict.reason}"
            )

            if verdict.hard_block:
                # Luật cứng kích hoạt — không đưa vào hàng đợi
                print(
                    f"  [Memory] ❌ LUẬT CỨNG — Không xếp hàng "
                    f"{best.symbol} {best.direction}"
                )
            else:
                # ② tự quyết định — tính win_prob + confidence
                should_enter, pred = self.decide_entry(best, df, balance)

                # Tính stake
                if pred and pred.stake_suggestion > 0:
                    stake = pred.stake_suggestion
                else:
                    params = self.learner.get_params()
                    stake  = round(
                        self.risk.compute_stake(best.score, balance) * params.stake_multiplier,
                        2,
                    )
                    stake = max(config.STAKE_MIN_USD, min(config.STAKE_MAX_USD, stake))

                win_prob    = pred.win_prob    if pred else 0.50
                confidence  = pred.confidence  if pred else 0.30
                wave_active = bool(best.wave and best.wave.correction_active)
                fib_zone    = best.wave.fib_zone if best.wave else "NONE"

                # Áp dụng memory priority_boost vào priority score
                priority = best.score * win_prob * confidence + verdict.priority_boost

                queued = QueuedTrade(
                    priority       = priority,
                    enqueued_at    = time.time(),
                    symbol         = best.symbol,
                    direction      = best.direction,
                    score          = best.score,
                    win_prob       = win_prob,
                    confidence     = confidence,
                    stake          = stake,
                    wave_active    = wave_active,
                    fib_zone       = fib_zone,
                    signal_ref     = best,
                    trade_features = trade_features,
                )

                submitted = self._pipeline.submit(queued)
                if submitted:
                    boost_str = f"  mem_boost={verdict.priority_boost:+.1f}" if verdict.priority_boost != 0 else ""
                    print(
                        f"  [Queue] ✅ Thêm vào hàng đợi: {best.symbol} {best.direction}  "
                        f"score={best.score:.0f}  wp={win_prob:.2f}  "
                        f"priority={priority:.1f}{boost_str}"
                    )
        else:
            print("  ⏳ Không có tín hiệu đủ điều kiện.")

        # ── Hiển thị trạng thái hàng đợi ─────────────────────────
        self._pipeline.print_queue_status()

        # ── Bước B: Dispatch — lấy + kiểm tra + thực thi ─────────
        allowed, _ = self.risk.can_trade(balance=balance)

        outcome = self._pipeline.dispatch(
            balance        = balance,
            risk_can_trade = allowed,
            executor_fn    = self._execute_trade,
            memory_brain   = self.memory,      # Gate 4 — memory hard veto
        )

        if outcome is None:
            print("  [Dispatch] Không có lệnh nào được thực thi chu kỳ này.")
            return

        if outcome.rejected_by:
            # Bị từ chối bởi gate — không thực thi
            return

        # Cập nhật risk + ghi log
        self.risk.update_after_trade(won=outcome.won, pnl=outcome.pnl)
        status    = "✅ THẮNG" if outcome.won else "❌ THUA"
        print(
            f"\n  {status}  {outcome.symbol} {outcome.direction}  "
            f"P&L: {outcome.pnl:+.2f} USD  latency={outcome.latency_ms:.0f}ms"
        )
        self._consecutive_errors = 0

    # ── Executor function (được truyền vào Orchestrator) ─────────

    def _execute_trade(self, trade: QueuedTrade) -> Optional[dict]:
        """
        Thực thi lệnh thật và ghi log.
        Được gọi bởi Orchestrator.dispatch() sau khi vượt qua tất cả cổng.

        Returns dict kết quả hoặc None nếu thất bại.
        """
        ts = datetime.now()

        if trade.wave_active:
            signal = trade.signal_ref
            wave   = signal.wave if signal else None
            print(
                f"  [Execute] {trade.symbol} {trade.direction}  "
                f"score={trade.score:.0f}  stake={trade.stake:.2f} USD  "
                + (
                    f"Fib={trade.fib_zone}  "
                    f"TP={wave.tp_price}  SL={wave.sl_price}"
                    if wave else ""
                )
            )
        else:
            print(
                f"  [Execute] {trade.symbol} {trade.direction}  "
                f"score={trade.score:.0f}  stake={trade.stake:.2f} USD"
            )

        try:
            result = place_and_wait(trade.direction, trade.symbol, trade.stake)
        except Exception as exc:
            print(f"  [LỖI] Đặt lệnh thất bại: {exc}")
            self._consecutive_errors += 1
            return None

        won    = result["won"]
        pnl    = result["pnl"]
        payout = result.get("payout", 0)

        # ── Ghi nhận kết quả vào Memory Brain (BẮT BUỘC) ─────────
        features = trade.trade_features
        if features is None:
            signal   = trade.signal_ref
            features = MemoryBrain.features_from_signal(signal) if signal else None

        if features is not None:
            self.memory.record_outcome(features, won=won, pnl=pnl)

        signal = trade.signal_ref
        record = TradeRecord(
            timestamp    = ts.isoformat(),
            symbol       = trade.symbol,
            direction    = trade.direction,
            signal_score = trade.score,
            stake        = trade.stake,
            payout       = payout,
            pnl          = pnl,
            won          = won,
            contract_id  = result.get("contract_id", ""),
            rsi          = signal.rsi          if signal else 0.0,
            momentum     = signal.momentum     if signal else 0.0,
            macd_hist    = signal.macd_hist    if signal else 0.0,
            bb_position  = signal.bb_position  if signal else 0.0,
        )
        self.logger.log(record)
        return result

    # ── ⑤ tự sửa lỗi ─────────────────────────────────────────────

    def self_heal(self) -> bool:
        """
        Kiểm tra sức khỏe hệ thống và tự phục hồi khi cần.

        Logic:
          - Nếu có quá nhiều lỗi liên tiếp → chuyển PAPER, nghỉ, thử kết nối lại
          - Nếu kết nối tốt → khôi phục LIVE

        Returns True nếu hệ thống khỏe mạnh, False nếu vẫn lỗi.
        """
        if self._consecutive_errors < _MAX_CONSECUTIVE_ERRS:
            return True

        print(
            f"\n  🔧 [Heal] {self._consecutive_errors} lỗi liên tiếp — "
            f"chuyển PAPER, nghỉ {config.HEAL_COOLDOWN_SECONDS}s..."
        )
        prev_mode = self._mode
        self._save_mode(SystemMode.PAPER)
        self._consecutive_errors = 0
        time.sleep(config.HEAL_COOLDOWN_SECONDS)

        # Thử kết nối lại
        try:
            bal = get_balance()
            if bal >= 0:
                print(f"  🔧 [Heal] Kết nối OK (balance={bal:.2f}) — khôi phục {prev_mode.value}")
                self._save_mode(prev_mode)
                return True
        except Exception as exc:
            print(f"  🔧 [Heal] Kết nối vẫn lỗi: {exc} — giữ PAPER mode")

        return False

    # ── ⑥ tự học ─────────────────────────────────────────────────

    def trigger_learning(self) -> None:
        """Kích hoạt learning cycle để cập nhật adaptive params."""
        history_len = self._r.llen(config.REDIS_LOG_KEY)
        print(f"\n  🧠 [Learning] Phân tích {history_len} lệnh lịch sử...")

        prev_mode = self._mode
        self._save_mode(SystemMode.LEARNING)
        try:
            new_params = self.learner.run_learning_cycle()
            print(
                f"  🧠 [Learning] ✅ Xong: "
                f"min_score={new_params.effective_min_score}  "
                f"stake_mult={new_params.stake_multiplier}×  "
                f"weak_conds={len(new_params.weak_conditions)}"
            )
        except Exception as exc:
            print(f"  🧠 [Learning] ⚠️  Lỗi: {exc}")
        finally:
            self._save_mode(prev_mode)

    # ── ⑦ tự mô phỏng on-demand ──────────────────────────────────

    def run_simulation(self, symbol: str) -> None:
        """Chạy walk-forward backtest nhanh cho một thị trường."""
        print(f"\n  🔬 [Sim] Backtest {symbol}...")
        try:
            df     = deriv_data.fetch_candles(symbol=symbol, count=config.SIM_CANDLE_COUNT)
            result = simulate(df, symbol=symbol)
            status = "✅ KHẢ THI" if result.is_viable() else "⚠️  KHÔNG KHẢ THI"
            print(
                f"  🔬 [Sim] {symbol}: {status}  "
                f"trades={result.total_trades}  WR={result.win_rate_pct:.1f}%  "
                f"PF={result.profit_factor:.2f}  PnL={result.total_pnl:+.2f}"
            )
        except Exception as exc:
            print(f"  🔬 [Sim] Lỗi backtest {symbol}: {exc}")

    # ── ⑧ tự scale ───────────────────────────────────────────────

    def self_scale(self) -> None:
        """
        Mở rộng hoặc thu hẹp pool thị trường dựa trên hiệu suất.

        - Win rate >= SCALE_HIGH_WIN_RATE và PF >= 1.3 → thêm market mới
        - Win rate < SCALE_LOW_WIN_RATE              → loại market cuối
        """
        stats = self.logger.get_stats()
        if "message" in stats or stats.get("total_trades", 0) < config.SCALE_MIN_TRADES:
            return

        wr        = stats["win_rate_pct"]
        pf        = stats["profit_factor"]
        full_pool = list(config.SCAN_SYMBOLS)
        current   = list(self._active_symbols)

        if wr >= config.SCALE_HIGH_WIN_RATE and pf >= 1.3:
            candidates = [s for s in full_pool if s not in current]
            if candidates:
                new_sym = candidates[0]
                current.append(new_sym)
                self._active_symbols = current
                self._save_active_symbols()
                print(f"  📈 [Scale] WR={wr:.1f}% PF={pf:.2f} → Thêm {new_sym} vào pool")

        elif wr < config.SCALE_LOW_WIN_RATE and len(current) > 1:
            removed              = current.pop(-1)
            self._active_symbols = current
            self._save_active_symbols()
            print(f"  📉 [Scale] WR={wr:.1f}% → Loại {removed} khỏi pool")

        else:
            print(
                f"  ⚖️  [Scale] WR={wr:.1f}%  PF={pf:.2f}  "
                f"Pool ổn định ({len(current)} thị trường)"
            )

    # ── Dashboard ─────────────────────────────────────────────────

    def print_dashboard(self, balance: float, mode: SystemMode) -> None:
        """In dashboard giám sát cho operator."""
        params    = self.learner.get_params()
        mode_icon = {"LIVE": "🟢", "PAPER": "📄", "PAUSED": "⏸️", "LEARNING": "🧠"}.get(mode.value, "⚪")
        metrics   = self._pipeline._metrics.snapshot()
        mem_rules = len(self.memory._hard_rules)

        print(f"\n{'='*65}")
        print(f"  👁️  DECISION ENGINE + PIPELINE + MEMORY  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*65}")
        print(f"  Mode     : {mode_icon} {mode.value}  |  Balance: {balance:.2f} USD")
        print(
            f"  Pool     : {', '.join(self._active_symbols)}"
            f"  ({len(self._active_symbols)}/{len(config.SCAN_SYMBOLS)})"
        )
        print(
            f"  Learned  : min_score={params.effective_min_score:.1f}  "
            f"stake_mult={params.stake_multiplier:.2f}×  "
            f"weak_conds={len(params.weak_conditions)}"
        )
        learn_in = config.LEARNER_INTERVAL_CYCLES - (self._cycle_count % config.LEARNER_INTERVAL_CYCLES)
        scale_in = config.SCALE_INTERVAL_CYCLES   - (self._cycle_count % config.SCALE_INTERVAL_CYCLES)
        print(
            f"  Cycle    : #{self._cycle_count}  |  "
            f"Errors: {self._consecutive_errors}  |  "
            f"Learn in: {learn_in}  |  Scale in: {scale_in}"
        )
        # Pipeline metrics summary
        print(
            f"  Pipeline : submitted={metrics['total_submitted']}  "
            f"executed={metrics['total_executed']}  "
            f"rejected={metrics['total_rejected']} ({metrics['rejection_rate_pct']:.0f}%)  "
            f"throughput={metrics['throughput_per_h']:.1f}/h"
        )
        # Memory Brain summary
        print(
            f"  Memory🧠 : hard_rules={mem_rules}  "
            f"(block_threshold≥{config.MEMORY_HARD_BLOCK_LOSS_RATE*100:.0f}% loss  "
            f"min_samples={config.MEMORY_MIN_SAMPLES_FOR_RULE})"
        )

    # ── Master run loop ───────────────────────────────────────────

    def run(self) -> None:
        """
        Vòng lặp chính — chạy mãi, tự điều khiển.
        Bạn chỉ cần giám sát.
        """
        print("\n" + "=" * 65)
        print("  🚀 DECISION ENGINE + PIPELINE + MEMORY BRAIN — ONLINE")
        print("  Hệ thống vận hành như tổ chức thật:")
        print("  có hàng đợi  |  có quyền hạn  |  có giới hạn tải")
        print("  có control   |  có đo lường   |  có điều phối")
        print("  có bộ nhớ Redis — luật cứng win/loss")
        print("=" * 65)
        print(f"  Pool ban đầu    : {', '.join(self._active_symbols)}")
        print(f"  Mode ban đầu    : {self._mode.value}")
        print(f"  Điểm tối thiểu  : {config.MIN_SIGNAL_SCORE}")
        print(f"  Sóng hồi biên   : {config.WAVE_CORRECTION_MIN*100:.0f}%–{config.WAVE_CORRECTION_MAX*100:.0f}%")
        print(f"  Predict min WP  : {config.PREDICT_MIN_WIN_PROB:.0%}")
        print(f"  Giới hạn lỗ ngày: {config.RISK_MAX_DAILY_LOSS_PCT*100:.0f}%")
        print(f"  Chu kỳ quét     : {config.SCAN_INTERVAL_SECONDS}s")
        print(f"  Hàng đợi max    : {config.PIPELINE_MAX_QUEUE_DEPTH} lệnh")
        print(
            f"  Rate limit      : {config.PIPELINE_RATE_MAX_TRADES} lệnh "
            f"/ {config.PIPELINE_RATE_WINDOW_SECONDS}s  "
            f"|  gap tối thiểu: {config.PIPELINE_MIN_TRADE_GAP_SECONDS}s"
        )
        print(f"  Authority gates : cần {config.PIPELINE_MIN_AUTHORITY_GATES}/3 cổng")
        print(
            f"  Memory Brain    : block≥{config.MEMORY_HARD_BLOCK_LOSS_RATE*100:.0f}% loss  "
            f"min_n={config.MEMORY_MIN_SAMPLES_FOR_RULE}  "
            f"rules={len(self.memory._hard_rules)}"
        )
        print("=" * 65)

        # Chạy simulation ban đầu trước khi vào vòng lặp
        if config.ENGINE_RUN_SIM_ON_START:
            print("\n  [Startup] Chạy backtest ban đầu...")
            for sym in self._active_symbols:
                self.run_simulation(sym)

        print(f"\n  [{datetime.now()}] Engine ONLINE. Nhấn Ctrl+C để dừng.\n")

        while True:
            try:
                self._cycle_count += 1

                # Lấy số dư
                try:
                    balance = get_balance()
                except Exception as exc:
                    print(f"  [LỖI] Không lấy được số dư: {exc}")
                    self._consecutive_errors += 1
                    self.self_heal()
                    time.sleep(config.SCAN_INTERVAL_SECONDS)
                    continue

                # Dashboard
                work_mode = self.decide_work(balance)
                self.print_dashboard(balance, work_mode)

                # ① tự chọn việc → dispatch
                if work_mode == SystemMode.PAUSED:
                    _, reason = self.risk.can_trade(balance=balance)
                    print(f"  ⏸️  PAUSED: {reason}")

                elif work_mode == SystemMode.LEARNING:
                    self.trigger_learning()

                elif work_mode == SystemMode.PAPER:
                    self.run_paper_cycle()

                else:   # LIVE
                    # ⑤ tự sửa lỗi
                    if not self.self_heal():
                        time.sleep(config.SCAN_INTERVAL_SECONDS)
                        continue

                    # ⑧ tự scale (mỗi SCALE_INTERVAL_CYCLES chu kỳ)
                    if self._cycle_count % config.SCALE_INTERVAL_CYCLES == 0:
                        self.self_scale()

                    # ④ tự hành động — qua pipeline
                    self.run_live_cycle(balance=balance)

                print(f"\n  {self.risk.summary()}")
                self.logger.print_stats()

                # In pipeline + memory report mỗi 10 chu kỳ
                if self._cycle_count % 10 == 0:
                    self._pipeline._metrics.print_report()
                    self.memory.report()

            except KeyboardInterrupt:
                print(f"\n  [{datetime.now()}] Operator ngắt hệ thống.")
                print(f"  {self.risk.summary()}")
                self.logger.print_stats()
                self._pipeline._metrics.print_report()
                self.memory.report()
                break

            except Exception as exc:
                self._consecutive_errors += 1
                print(f"  [LỖI nghiêm trọng] {exc}")
                self.self_heal()

            print(f"\n  ⏳ Nghỉ {config.SCAN_INTERVAL_SECONDS}s trước chu kỳ tiếp theo...\n")
            time.sleep(config.SCAN_INTERVAL_SECONDS)


# ──────────────────────────────────────────────────────────────────
# Chạy trực tiếp để kiểm tra
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = DecisionEngine()
    engine.run()

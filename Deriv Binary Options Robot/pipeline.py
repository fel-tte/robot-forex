"""
pipeline.py
===========
PIPELINE — Dây chuyền điều phối vận hành

Hệ thống không còn "vào lệnh liên tục" mà vận hành như một tổ chức thật:

┌─────────────────────────────────────────────────────────┐
│  SIGNAL  →  [QUEUE]  →  [AUTHORITY GATE]  →  [RATE      │
│            hàng đợi      quyền hạn           LIMITER]   │
│                                               giới hạn   │
│                                               tải        │
│                          ↓ pass                          │
│                        [EXECUTOR]  →  [METRICS]         │
│                         thực thi      đo lường          │
└─────────────────────────────────────────────────────────┘

Components:
  TradeQueue       — hàng đợi tín hiệu có ưu tiên
  PermissionGate   — quyền hạn: kiểm tra score, predictor, risk
  LoadLimiter      — giới hạn tải: rate window + gap giữa lệnh
  PipelineMetrics  — đo lường: throughput, latency, rejection rate
  Orchestrator     — điều phối: nhận signal → xếp hàng → kiểm tra → thực thi

Tất cả trạng thái được lưu vào Redis để bền vững qua restart.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Callable, TYPE_CHECKING

import redis

import config

if TYPE_CHECKING:
    from memory import MemoryBrain, TradeFeatures


# ──────────────────────────────────────────────────────────────────
# Dataclass đặc tả một lệnh trong hàng đợi
# ──────────────────────────────────────────────────────────────────

@dataclass(order=True)
class QueuedTrade:
    """Một tín hiệu đang chờ trong hàng đợi."""
    priority:        float    = field(compare=True)   # Điểm ưu tiên (cao hơn = trước)
    enqueued_at:     float    = field(compare=False)   # Thời điểm vào hàng (Unix timestamp)
    symbol:          str      = field(compare=False)
    direction:       str      = field(compare=False)
    score:           float    = field(compare=False)
    win_prob:        float    = field(compare=False)
    confidence:      float    = field(compare=False)
    stake:           float    = field(compare=False)
    wave_active:     bool     = field(compare=False)
    fib_zone:        str      = field(compare=False)
    signal_ref:      object   = field(compare=False, repr=False)   # MarketSignal gốc
    trade_features:  object   = field(compare=False, repr=False, default=None)  # TradeFeatures


@dataclass
class TradeOutcome:
    """Kết quả của một lệnh đã thực thi."""
    symbol:        str
    direction:     str
    score:         float
    won:           bool
    pnl:           float
    stake:         float
    executed_at:   float   # Unix timestamp
    latency_ms:    float   # Thời gian từ enqueue → execute (ms)
    rejected_by:   str     = ""   # Cổng từ chối (nếu bị từ chối)


# ──────────────────────────────────────────────────────────────────
# TradeQueue — hàng đợi tín hiệu có ưu tiên
# ──────────────────────────────────────────────────────────────────

class TradeQueue:
    """
    Hàng đợi tín hiệu có ưu tiên dựa trên điểm tín hiệu.

    - Ưu tiên = score × win_prob × confidence
    - Giới hạn tối đa PIPELINE_MAX_QUEUE_DEPTH phần tử
    - Loại bỏ phần tử cũ nhất nếu hàng đợi đầy và phần tử mới tốt hơn
    """

    def __init__(self, max_depth: int = config.PIPELINE_MAX_QUEUE_DEPTH) -> None:
        self._max_depth = max_depth
        self._queue: list[QueuedTrade] = []

    def push(self, trade: QueuedTrade) -> bool:
        """
        Thêm tín hiệu vào hàng đợi.

        Returns True nếu được chấp nhận, False nếu bị từ chối.
        """
        if len(self._queue) < self._max_depth:
            self._queue.append(trade)
            self._queue.sort(key=lambda t: t.priority, reverse=True)
            return True

        # Hàng đợi đầy — thay thế nếu phần tử mới tốt hơn phần tử cuối
        weakest = self._queue[-1]
        if trade.priority > weakest.priority:
            self._queue[-1] = trade
            self._queue.sort(key=lambda t: t.priority, reverse=True)
            print(
                f"  [Queue] ↔️  Thay thế {weakest.symbol} (priority={weakest.priority:.1f}) "
                f"bằng {trade.symbol} (priority={trade.priority:.1f})"
            )
            return True

        print(
            f"  [Queue] 🚫 Hàng đợi đầy ({len(self._queue)}/{self._max_depth}), "
            f"{trade.symbol} không đủ ưu tiên để thay thế."
        )
        return False

    def pop(self) -> Optional[QueuedTrade]:
        """Lấy phần tử có ưu tiên cao nhất."""
        return self._queue.pop(0) if self._queue else None

    def peek(self) -> Optional[QueuedTrade]:
        """Xem phần tử đầu hàng đợi mà không lấy ra."""
        return self._queue[0] if self._queue else None

    def size(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return not self._queue

    def snapshot(self) -> list[dict]:
        """Ảnh chụp hàng đợi hiện tại để hiển thị dashboard."""
        return [
            {
                "rank"     : i + 1,
                "symbol"   : t.symbol,
                "direction": t.direction,
                "score"    : t.score,
                "win_prob" : t.win_prob,
                "priority" : round(t.priority, 2),
                "wait_s"   : round(time.time() - t.enqueued_at, 1),
            }
            for i, t in enumerate(self._queue)
        ]


# ──────────────────────────────────────────────────────────────────
# PermissionGate — quyền hạn (3 cổng xác nhận)
# ──────────────────────────────────────────────────────────────────

class PermissionGate:
    """
    Kiểm tra quyền hạn trước khi thực thi lệnh.

    4 cổng độc lập:
      Gate 1 — Điểm tín hiệu (score) đủ ngưỡng tối thiểu
      Gate 2 — Predictor xác nhận (win_prob + confidence)
      Gate 3 — Risk manager cho phép (không paused, không vượt lỗ ngày)
      Gate 4 — Memory Brain: luật cứng Redis (HARD VETO — chặn tuyệt đối nếu kích hoạt)

    Gate 4 là HARD VETO: nếu kích hoạt → lệnh bị từ chối bất kể các cổng khác.
    Gates 1-3: cần PIPELINE_MIN_AUTHORITY_GATES cổng mở.
    """

    def check(
        self,
        trade:          QueuedTrade,
        balance:        float,
        risk_can_trade: bool,
        min_gates:      int = config.PIPELINE_MIN_AUTHORITY_GATES,
        memory_brain:   Optional["MemoryBrain"] = None,
    ) -> tuple[bool, list[str], list[str]]:
        """
        Kiểm tra tất cả cổng quyền hạn.

        Returns
        -------
        (approved: bool, passed: list[str], blocked: list[str])
        """
        passed  : list[str] = []
        blocked : list[str] = []

        # Gate 4 (HARD VETO): Memory Brain — kiểm tra trước, veto ngay nếu cần
        if memory_brain is not None and trade.trade_features is not None:
            verdict = memory_brain.consult(trade.trade_features)
            if verdict.hard_block:
                # Hard veto — không cần kiểm tra các cổng khác
                return False, [], [f"memory_hard_block({verdict.reason[:60]})"]
            else:
                passed.append(f"memory(WR={verdict.win_rate*100:.0f}%)")

        # Gate 1: Signal score
        learner_min = getattr(config, "_effective_min_score", config.MIN_SIGNAL_SCORE)
        if trade.score >= learner_min:
            passed.append("score")
        else:
            blocked.append(f"score({trade.score:.0f}<{learner_min:.0f})")

        # Gate 2: Predictor (win_prob + confidence)
        if (trade.win_prob  >= config.PREDICT_MIN_WIN_PROB
                and trade.confidence >= config.PREDICT_MIN_CONFIDENCE):
            passed.append("predictor")
        else:
            blocked.append(
                f"predictor(wp={trade.win_prob:.2f}<{config.PREDICT_MIN_WIN_PROB:.2f}"
                f" or conf={trade.confidence:.2f}<{config.PREDICT_MIN_CONFIDENCE:.2f})"
            )

        # Gate 3: Risk manager
        if risk_can_trade:
            passed.append("risk")
        else:
            blocked.append("risk(paused_or_daily_loss)")

        approved = len(passed) >= min_gates
        return approved, passed, blocked


# ──────────────────────────────────────────────────────────────────
# LoadLimiter — giới hạn tải (rate limiting + gap)
# ──────────────────────────────────────────────────────────────────

class LoadLimiter:
    """
    Giới hạn tải để ngăn hệ thống đặt lệnh liên tục.

    Hai lớp kiểm soát:
      1. Rate window  : không quá PIPELINE_RATE_MAX_TRADES lệnh
                        trong PIPELINE_RATE_WINDOW_SECONDS giây gần nhất
      2. Gap tối thiểu: khoảng cách ít nhất PIPELINE_MIN_TRADE_GAP_SECONDS
                        giữa 2 lệnh liên tiếp
    """

    def __init__(self) -> None:
        self._executed_times: deque[float] = deque()   # Unix timestamps của các lệnh đã thực thi
        self._last_executed: float = 0.0

    def can_execute(self) -> tuple[bool, str]:
        """
        Kiểm tra có được phép thực thi lệnh không.

        Returns (allowed: bool, reason: str)
        """
        now = time.time()

        # Xóa các timestamp cũ ngoài cửa sổ
        cutoff = now - config.PIPELINE_RATE_WINDOW_SECONDS
        while self._executed_times and self._executed_times[0] < cutoff:
            self._executed_times.popleft()

        # Kiểm tra gap tối thiểu
        gap_elapsed = now - self._last_executed
        if self._last_executed > 0 and gap_elapsed < config.PIPELINE_MIN_TRADE_GAP_SECONDS:
            wait_more = config.PIPELINE_MIN_TRADE_GAP_SECONDS - gap_elapsed
            return False, f"Gap tối thiểu chưa đủ (còn {wait_more:.0f}s)"

        # Kiểm tra rate window
        recent_count = len(self._executed_times)
        if recent_count >= config.PIPELINE_RATE_MAX_TRADES:
            oldest     = self._executed_times[0]
            reset_in   = int(config.PIPELINE_RATE_WINDOW_SECONDS - (now - oldest))
            return False, (
                f"Rate limit: {recent_count}/{config.PIPELINE_RATE_MAX_TRADES} lệnh "
                f"trong {config.PIPELINE_RATE_WINDOW_SECONDS}s "
                f"(reset sau {reset_in}s)"
            )

        return True, "OK"

    def record_execution(self) -> None:
        """Ghi nhận một lệnh vừa được thực thi."""
        now = time.time()
        self._executed_times.append(now)
        self._last_executed = now

    def status(self) -> dict:
        """Trạng thái hiện tại của load limiter."""
        now    = time.time()
        cutoff = now - config.PIPELINE_RATE_WINDOW_SECONDS
        while self._executed_times and self._executed_times[0] < cutoff:
            self._executed_times.popleft()

        gap_elapsed = now - self._last_executed if self._last_executed > 0 else float("inf")
        return {
            "recent_trades"    : len(self._executed_times),
            "rate_limit"       : config.PIPELINE_RATE_MAX_TRADES,
            "rate_window_s"    : config.PIPELINE_RATE_WINDOW_SECONDS,
            "last_trade_gap_s" : round(gap_elapsed, 1),
            "min_gap_s"        : config.PIPELINE_MIN_TRADE_GAP_SECONDS,
        }


# ──────────────────────────────────────────────────────────────────
# PipelineMetrics — đo lường
# ──────────────────────────────────────────────────────────────────

class PipelineMetrics:
    """
    Đo lường hiệu suất của pipeline trong thời gian thực.

    Chỉ số:
      - throughput       : số lệnh thực thi / giờ
      - rejection_rate   : % tín hiệu bị từ chối bởi các cổng
      - avg_queue_wait_s : thời gian chờ trung bình trong hàng đợi (ms)
      - gate_pass_rate   : % tín hiệu vượt qua từng cổng
      - win_rate         : win rate thực tế trong cửa sổ đo lường
      - avg_pnl          : P&L trung bình mỗi lệnh
    """

    def __init__(self, window_seconds: int = config.PIPELINE_METRICS_WINDOW_SECONDS) -> None:
        self._window_s     = window_seconds
        self._outcomes     : list[TradeOutcome] = []
        self._submissions  : list[float]        = []   # timestamps của mọi tín hiệu đưa vào pipeline
        self._rejections   : dict[str, int]     = {}   # reason → count
        self._gate_stats   : dict[str, int]     = {}   # gate_name → passes count

    def record_submission(self) -> None:
        """Ghi nhận một tín hiệu được đưa vào pipeline."""
        self._submissions.append(time.time())
        self._prune()

    def record_rejection(self, reason: str) -> None:
        """Ghi nhận một tín hiệu bị từ chối."""
        self._rejections[reason] = self._rejections.get(reason, 0) + 1

    def record_gate_result(self, gate: str, passed: bool) -> None:
        """Ghi nhận kết quả của một cổng quyền hạn."""
        key = f"gate_{gate}_pass" if passed else f"gate_{gate}_fail"
        self._gate_stats[key] = self._gate_stats.get(key, 0) + 1

    def record_outcome(self, outcome: TradeOutcome) -> None:
        """Ghi nhận kết quả của một lệnh đã thực thi."""
        self._outcomes.append(outcome)
        self._prune()

    def _prune(self) -> None:
        """Xóa dữ liệu cũ ngoài cửa sổ đo lường."""
        cutoff = time.time() - self._window_s
        self._outcomes    = [o for o in self._outcomes    if o.executed_at >= cutoff]
        self._submissions = [t for t in self._submissions if t                >= cutoff]

    def snapshot(self) -> dict:
        """Tóm tắt metrics hiện tại."""
        self._prune()
        total_sub  = len(self._submissions)
        total_exec = len(self._outcomes)
        rejected   = sum(self._rejections.values())
        wins       = sum(1 for o in self._outcomes if o.won)
        total_pnl  = sum(o.pnl for o in self._outcomes)
        latencies  = [o.latency_ms for o in self._outcomes if o.latency_ms > 0]

        win_rate  = wins / total_exec * 100 if total_exec > 0 else 0.0
        avg_pnl   = total_pnl / total_exec  if total_exec > 0 else 0.0
        avg_lat   = sum(latencies) / len(latencies) if latencies else 0.0

        # Throughput: lệnh/giờ dựa trên cửa sổ hiện tại
        elapsed_h = self._window_s / 3600
        throughput = total_exec / elapsed_h if elapsed_h > 0 else 0.0

        rej_rate = rejected / total_sub * 100 if total_sub > 0 else 0.0

        return {
            "window_hours"     : round(self._window_s / 3600, 2),
            "total_submitted"  : total_sub,
            "total_executed"   : total_exec,
            "total_rejected"   : rejected,
            "rejection_rate_pct": round(rej_rate, 1),
            "throughput_per_h" : round(throughput, 2),
            "win_rate_pct"     : round(win_rate, 1),
            "total_pnl"        : round(total_pnl, 2),
            "avg_pnl_per_trade": round(avg_pnl, 4),
            "avg_latency_ms"   : round(avg_lat, 1),
            "rejections"       : dict(self._rejections),
            "gate_stats"       : dict(self._gate_stats),
        }

    def print_report(self) -> None:
        m = self.snapshot()
        rej_breakdown = ", ".join(f"{k}={v}" for k, v in m["rejections"].items()) or "—"
        print(
            f"\n  {'─'*60}\n"
            f"  📏 PIPELINE METRICS  (cửa sổ {m['window_hours']:.1f}h)\n"
            f"  {'─'*60}\n"
            f"  Submitted  : {m['total_submitted']:>4}  "
            f"Executed: {m['total_executed']:>4}  "
            f"Rejected: {m['total_rejected']:>4} ({m['rejection_rate_pct']:.1f}%)\n"
            f"  Throughput : {m['throughput_per_h']:.1f} lệnh/giờ  "
            f"|  Win rate: {m['win_rate_pct']:.1f}%\n"
            f"  P&L tổng   : {m['total_pnl']:+.2f} USD  "
            f"|  Trung bình: {m['avg_pnl_per_trade']:+.4f} USD/lệnh\n"
            f"  Latency tb : {m['avg_latency_ms']:.0f}ms  "
            f"|  Từ chối do: {rej_breakdown}\n"
            f"  {'─'*60}"
        )


# ──────────────────────────────────────────────────────────────────
# Orchestrator — điều phối toàn bộ pipeline
# ──────────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Điều phối toàn bộ dây chuyền:
      submit()  — nhận tín hiệu, đưa vào hàng đợi
      dispatch() — lấy từ hàng đợi, kiểm tra cổng, thực thi

    Sử dụng trong DecisionEngine:
        orch = Orchestrator(risk=..., load=..., gate=..., metrics=...)
        orch.submit(signal, pred, stake)
        orch.dispatch(balance, executor_fn)
    """

    def __init__(
        self,
        queue   : TradeQueue,
        gate    : PermissionGate,
        limiter : LoadLimiter,
        metrics : PipelineMetrics,
    ) -> None:
        self._queue   = queue
        self._gate    = gate
        self._limiter = limiter
        self._metrics = metrics

    # ── Giai đoạn 1: Nhận tín hiệu vào hàng đợi ─────────────────

    def submit(self, trade: QueuedTrade) -> bool:
        """
        Nhận một tín hiệu và đưa vào hàng đợi.

        Returns True nếu được chấp nhận vào hàng đợi.
        """
        self._metrics.record_submission()
        accepted = self._queue.push(trade)
        if not accepted:
            self._metrics.record_rejection("queue_full")
        return accepted

    # ── Giai đoạn 2: Kiểm tra + thực thi lệnh đầu hàng đợi ──────

    def dispatch(
        self,
        balance        : float,
        risk_can_trade : bool,
        executor_fn    : Callable[[QueuedTrade], Optional[dict]],
        memory_brain   : Optional["MemoryBrain"] = None,
    ) -> Optional[TradeOutcome]:
        """
        Lấy lệnh đầu hàng đợi, kiểm tra cổng, thực thi.

        Parameters
        ----------
        balance        : số dư tài khoản
        risk_can_trade : kết quả risk.can_trade()
        executor_fn    : hàm thực thi lệnh (nhận QueuedTrade, trả về dict kết quả)
        memory_brain   : MemoryBrain để kiểm tra Gate 4 (hard veto)

        Returns TradeOutcome nếu thực thi, None nếu bị chặn hoặc hàng rỗng.
        """
        if self._queue.is_empty():
            return None

        # ── Kiểm tra giới hạn tải ─────────────────────────────────
        load_ok, load_reason = self._limiter.can_execute()
        if not load_ok:
            print(f"  [LoadLimiter] ⏳ {load_reason}")
            self._metrics.record_rejection(f"rate:{load_reason[:30]}")
            return None

        # ── Kiểm tra quyền hạn (bao gồm Gate 4 — memory hard veto) ──
        trade            = self._queue.pop()
        approved, passed, blocked = self._gate.check(
            trade, balance, risk_can_trade, memory_brain=memory_brain
        )

        for g in passed:
            self._metrics.record_gate_result(g, passed=True)
        for b in blocked:
            gate_name = b.split("(")[0]
            self._metrics.record_gate_result(gate_name, passed=False)

        if not approved:
            reason = ", ".join(blocked)
            print(f"  [Gate] 🚫 Từ chối {trade.symbol}: {reason}")
            self._metrics.record_rejection(f"gate:{blocked[0].split('(')[0]}")
            return TradeOutcome(
                symbol      = trade.symbol,
                direction   = trade.direction,
                score       = trade.score,
                won         = False,
                pnl         = 0.0,
                stake       = trade.stake,
                executed_at = time.time(),
                latency_ms  = 0.0,
                rejected_by = reason,
            )

        # ── Thực thi ─────────────────────────────────────────────
        print(
            f"  [Gate] ✅ Thông qua {len(passed)}/{len(passed)+len(blocked)} cổng "
            f"({', '.join(passed)})"
        )

        execute_start = time.time()
        raw_result    = executor_fn(trade)
        latency_ms    = (time.time() - execute_start) * 1000

        if raw_result is None:
            self._metrics.record_rejection("executor_failed")
            return None

        self._limiter.record_execution()

        outcome = TradeOutcome(
            symbol      = trade.symbol,
            direction   = trade.direction,
            score       = trade.score,
            won         = raw_result["won"],
            pnl         = raw_result["pnl"],
            stake       = trade.stake,
            executed_at = time.time(),
            latency_ms  = latency_ms,
        )
        self._metrics.record_outcome(outcome)
        return outcome

    # ── Dashboard hàng đợi ───────────────────────────────────────

    def print_queue_status(self) -> None:
        snap   = self._queue.snapshot()
        ls     = self._limiter.status()
        n      = self._queue.size()

        print(f"\n  📋 [QUEUE]  {n}/{config.PIPELINE_MAX_QUEUE_DEPTH} lệnh đang chờ")
        for item in snap:
            print(
                f"     #{item['rank']} {item['symbol']} {item['direction']}  "
                f"score={item['score']:.0f}  wp={item['win_prob']:.2f}  "
                f"priority={item['priority']:.1f}  wait={item['wait_s']}s"
            )

        print(
            f"  ⚡ [LOAD]   {ls['recent_trades']}/{ls['rate_limit']} lệnh "
            f"trong {ls['rate_window_s']}s  "
            f"|  gap từ lệnh cuối: {ls['last_trade_gap_s']}s "
            f"(min {ls['min_gap_s']}s)"
        )


# ──────────────────────────────────────────────────────────────────
# Chạy trực tiếp để kiểm tra logic
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    queue   = TradeQueue()
    gate    = PermissionGate()
    limiter = LoadLimiter()
    metrics = PipelineMetrics(window_seconds=3600)
    orch    = Orchestrator(queue, gate, limiter, metrics)

    # Mô phỏng một số tín hiệu
    symbols = ["R_10", "R_25", "R_50", "R_75", "R_100"]
    for i in range(5):
        score    = random.uniform(55, 95)
        win_prob = random.uniform(0.48, 0.72)
        conf     = random.uniform(0.25, 0.80)
        t = QueuedTrade(
            priority    = score * win_prob * conf,
            enqueued_at = time.time(),
            symbol      = random.choice(symbols),
            direction   = random.choice(["CALL", "PUT"]),
            score       = score,
            win_prob    = win_prob,
            confidence  = conf,
            stake       = 10.0,
            wave_active = random.choice([True, False]),
            fib_zone    = random.choice(["F382", "F618", "NONE"]),
            signal_ref  = None,
        )
        accepted = orch.submit(t)
        print(f"  Submit {t.symbol} {t.direction} score={t.score:.0f}: {'✅' if accepted else '🚫'}")

    orch.print_queue_status()

    # Mô phỏng dispatch
    def fake_executor(trade: QueuedTrade) -> dict:
        won = random.random() > 0.45
        return {"won": won, "pnl": 8.5 if won else -10.0, "payout": 18.5}

    print("\n  Dispatching...")
    for _ in range(3):
        outcome = orch.dispatch(
            balance        = 100.0,
            risk_can_trade = True,
            executor_fn    = fake_executor,
        )
        if outcome and not outcome.rejected_by:
            print(f"  → {outcome.symbol} {outcome.direction}: {'WIN' if outcome.won else 'LOSS'} P&L={outcome.pnl:+.2f}")

    metrics.print_report()

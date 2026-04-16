"""
memory.py
=========
MEMORY BRAIN — Redis là bộ não trung tâm ghi nhớ Win/Loss

Vai trò:
  Redis lưu trữ TOÀN BỘ trạng thái win/loss theo feature fingerprint.
  Trước mỗi lệnh, hệ thống BẮT BUỘC tham vấn MemoryBrain.
  Nếu fingerprint bị xếp vào luật cứng (hard rule) → lệnh bị CHẶN tuyệt đối.

Fingerprint là tổ hợp các yếu tố:
  {symbol}:{direction}:{score_band}:{fib_zone}:{wave}:{rsi_band}:{momentum_sign}:{hour_bucket}

Mỗi fingerprint tích lũy:
  win_count, loss_count, total_pnl, last_seen

Luật cứng được xây dựng khi:
  loss_rate >= MEMORY_HARD_BLOCK_LOSS_RATE  và  n >= MEMORY_MIN_SAMPLES_FOR_RULE

Cơ chế tra vấn nhiều cấp (multi-level lookup):
  Level 1 — Full fingerprint (chi tiết nhất)
  Level 2 — Bỏ hour_bucket (chỉ theo thị trường + hướng + điều kiện kỹ thuật)
  Level 3 — Chỉ {symbol}:{direction}:{score_band} (thị trường + hướng + điểm)

Nếu bất kỳ level nào kích hoạt hard rule → BLOCK tuyệt đối.

API:
  brain = MemoryBrain()
  brain.record_outcome(features, won, pnl)   # Ghi sau khi lệnh kết thúc
  verdict = brain.consult(features)          # Tham vấn trước khi đặt lệnh
  brain.rebuild_rules()                      # Cập nhật danh sách luật cứng
  brain.report()                             # In báo cáo tổng hợp
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import redis

import config


# ──────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────

@dataclass
class TradeFeatures:
    """
    Đặc tả đầy đủ bối cảnh của một lệnh để tạo fingerprint.

    Được tạo từ MarketSignal + thông tin thị trường hiện tại.
    """
    symbol:          str
    direction:       str      # CALL / PUT
    score:           float
    fib_zone:        str      # F236, F382, F500, F618, F786, NONE
    wave_active:     bool
    rsi:             float    # Giá trị RSI hiện tại
    momentum:        float    # Giá trị momentum hiện tại
    hour:            int      # Giờ UTC hiện tại (0-23)


@dataclass
class MemoryVerdict:
    """
    Kết quả tham vấn từ MemoryBrain.

    hard_block  : True → lệnh bị chặn tuyệt đối (luật cứng)
    approved    : True → lệnh được phép
    win_rate    : tỉ lệ thắng lịch sử của fingerprint (-1 nếu không có data)
    sample_count: số lệnh lịch sử phù hợp
    priority_boost: điều chỉnh ưu tiên (+/-)
    fingerprint : fingerprint đầy đủ được dùng
    matched_level: 1=full, 2=no_hour, 3=broad, 0=no_data
    reason      : mô tả quyết định
    """
    hard_block:      bool
    approved:        bool
    win_rate:        float
    sample_count:    int
    priority_boost:  float
    fingerprint:     str
    matched_level:   int
    reason:          str


# ──────────────────────────────────────────────────────────────────
# MemoryBrain
# ──────────────────────────────────────────────────────────────────

class MemoryBrain:
    """
    Bộ não trung tâm Redis — ghi nhớ và tra vấn Win/Loss patterns.

    Khởi tạo:
        brain = MemoryBrain()

    Trước khi đặt lệnh (BẮT BUỘC):
        verdict = brain.consult(features)
        if verdict.hard_block:
            return  # Không đặt lệnh

    Sau khi lệnh kết thúc:
        brain.record_outcome(features, won=True, pnl=8.5)
    """

    def __init__(self) -> None:
        self._r = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
        )
        # Tải hard rules vào memory để tra vấn nhanh
        self._hard_rules: set[str] = self._load_hard_rules()

    # ── Helpers: xây dựng fingerprint ────────────────────────────

    @staticmethod
    def _score_band(score: float) -> str:
        if score >= 91:
            return "91_100"
        elif score >= 81:
            return "81_90"
        elif score >= 71:
            return "71_80"
        return "60_70"

    @staticmethod
    def _rsi_band(rsi: float) -> str:
        if rsi < 35:
            return "oversold"
        elif rsi > 65:
            return "overbought"
        return "neutral"

    @staticmethod
    def _momentum_sign(momentum: float) -> str:
        if momentum > 0.001:
            return "pos"
        elif momentum < -0.001:
            return "neg"
        return "zero"

    @staticmethod
    def _hour_bucket(hour: int) -> str:
        if hour < 6:
            return "night"
        elif hour < 12:
            return "morning"
        elif hour < 18:
            return "afternoon"
        return "evening"

    def _make_fingerprint(self, f: TradeFeatures) -> str:
        """Full fingerprint (Level 1)."""
        return ":".join([
            f.symbol,
            f.direction,
            self._score_band(f.score),
            f.fib_zone,
            "wave" if f.wave_active else "nowave",
            self._rsi_band(f.rsi),
            self._momentum_sign(f.momentum),
            self._hour_bucket(f.hour),
        ])

    def _make_fingerprint_l2(self, f: TradeFeatures) -> str:
        """Level 2 — bỏ hour_bucket."""
        return ":".join([
            f.symbol,
            f.direction,
            self._score_band(f.score),
            f.fib_zone,
            "wave" if f.wave_active else "nowave",
            self._rsi_band(f.rsi),
            self._momentum_sign(f.momentum),
        ])

    def _make_fingerprint_l3(self, f: TradeFeatures) -> str:
        """Level 3 — chỉ symbol + direction + score_band."""
        return ":".join([
            f.symbol,
            f.direction,
            self._score_band(f.score),
        ])

    # ── Redis key ─────────────────────────────────────────────────

    def _key(self, fingerprint: str) -> str:
        return config.REDIS_MEMORY_PREFIX + fingerprint

    # ── Ghi nhận kết quả lệnh ────────────────────────────────────

    def record_outcome(
        self,
        features: TradeFeatures,
        won:      bool,
        pnl:      float,
    ) -> None:
        """
        Ghi nhận kết quả win/loss cho một lệnh đã thực thi.

        Lưu vào TẤT CẢ 3 level fingerprint để tra vấn đa cấp.
        Tự động rebuild hard rules sau khi ghi.
        """
        fp1 = self._make_fingerprint(f=features)
        fp2 = self._make_fingerprint_l2(f=features)
        fp3 = self._make_fingerprint_l3(f=features)

        for fp in (fp1, fp2, fp3):
            self._update_pattern(fp, won, pnl)

        # Rebuild rules sau mỗi lần ghi để cập nhật luật cứng
        self.rebuild_rules()

        status = "WIN" if won else "LOSS"
        print(
            f"  [Memory] 📝 {status}  {features.symbol} {features.direction}  "
            f"fp={fp1[:40]}  pnl={pnl:+.2f}"
        )

    def _update_pattern(self, fingerprint: str, won: bool, pnl: float) -> None:
        """Cập nhật hoặc khởi tạo pattern trong Redis."""
        key = self._key(fingerprint)
        pipe = self._r.pipeline()

        if won:
            pipe.hincrby(key, "win_count", 1)
        else:
            pipe.hincrby(key, "loss_count", 1)

        # Lưu total_pnl dưới dạng int (cents) để tránh float precision
        pipe.hincrbyfloat(key, "total_pnl", round(pnl, 4))
        pipe.hset(key, "last_seen", datetime.now().isoformat())
        pipe.execute()

    def _get_pattern(self, fingerprint: str) -> Optional[dict]:
        """Lấy thống kê của một fingerprint từ Redis."""
        key  = self._key(fingerprint)
        data = self._r.hgetall(key)
        if not data:
            return None
        return {
            "win_count"  : int(data.get(b"win_count",  0)),
            "loss_count" : int(data.get(b"loss_count", 0)),
            "total_pnl"  : float(data.get(b"total_pnl", 0)),
            "last_seen"  : data.get(b"last_seen", b"").decode(),
        }

    # ── Xây dựng luật cứng ───────────────────────────────────────

    def rebuild_rules(self) -> list[str]:
        """
        Quét tất cả pattern trong Redis, xây dựng danh sách luật cứng.

        Điều kiện thành luật cứng:
          - n >= MEMORY_MIN_SAMPLES_FOR_RULE
          - loss_rate >= MEMORY_HARD_BLOCK_LOSS_RATE

        Lưu vào Redis và cập nhật cache nội bộ.

        Returns danh sách fingerprints bị chặn.
        """
        blocked: list[str] = []
        prefix  = config.REDIS_MEMORY_PREFIX

        # Tìm tất cả keys của memory
        pattern = prefix + "*"
        for key_bytes in self._r.scan_iter(match=pattern):
            key_str     = key_bytes.decode()
            fingerprint = key_str[len(prefix):]
            data        = self._r.hgetall(key_bytes)
            if not data:
                continue

            wins   = int(data.get(b"win_count",  0))
            losses = int(data.get(b"loss_count", 0))
            n      = wins + losses

            if n < config.MEMORY_MIN_SAMPLES_FOR_RULE:
                continue

            loss_rate = losses / n
            if loss_rate >= config.MEMORY_HARD_BLOCK_LOSS_RATE:
                blocked.append(fingerprint)

        # Lưu vào Redis
        self._r.set(config.REDIS_MEMORY_RULES_KEY, json.dumps(blocked))

        # Cập nhật stats
        stats = {
            "total_patterns" : self._count_patterns(),
            "hard_rules"     : len(blocked),
            "last_rebuilt"   : datetime.now().isoformat(),
        }
        self._r.set(config.REDIS_MEMORY_STATS_KEY, json.dumps(stats))

        # Cập nhật cache
        self._hard_rules = set(blocked)
        return blocked

    def _load_hard_rules(self) -> set[str]:
        """Tải hard rules từ Redis."""
        raw = self._r.get(config.REDIS_MEMORY_RULES_KEY)
        if raw:
            try:
                return set(json.loads(raw))
            except Exception:
                pass
        return set()

    def _count_patterns(self) -> int:
        count   = 0
        pattern = config.REDIS_MEMORY_PREFIX + "*"
        for _ in self._r.scan_iter(match=pattern):
            count += 1
        return count

    # ── Tra vấn trước khi đặt lệnh (BẮT BUỘC) ───────────────────

    def consult(self, features: TradeFeatures) -> MemoryVerdict:
        """
        Tham vấn bộ nhớ trước khi đặt lệnh.

        Kiểm tra theo 3 level fingerprint:
          Level 1 → Full fingerprint
          Level 2 → Bỏ hour_bucket
          Level 3 → symbol + direction + score_band

        Hard block nếu bất kỳ level nào kích hoạt luật cứng.
        Trả về MemoryVerdict với đầy đủ thông tin để pipeline quyết định.
        """
        fp1 = self._make_fingerprint(f=features)
        fp2 = self._make_fingerprint_l2(f=features)
        fp3 = self._make_fingerprint_l3(f=features)

        # ── Kiểm tra luật cứng (hard block) ──────────────────────
        for level, fp in enumerate([fp1, fp2, fp3], start=1):
            if fp in self._hard_rules:
                data    = self._get_pattern(fp) or {}
                wins    = data.get("win_count",  0)
                losses  = data.get("loss_count", 0)
                n       = wins + losses
                wr      = wins / n if n > 0 else 0.0
                return MemoryVerdict(
                    hard_block     = True,
                    approved       = False,
                    win_rate       = round(wr, 4),
                    sample_count   = n,
                    priority_boost = 0.0,
                    fingerprint    = fp,
                    matched_level  = level,
                    reason         = (
                        f"🚫 LUẬT CỨNG L{level}: {fp[:50]}  "
                        f"n={n}  WR={wr*100:.1f}%  "
                        f"(loss_rate≥{config.MEMORY_HARD_BLOCK_LOSS_RATE*100:.0f}%)"
                    ),
                )

        # ── Tra vấn dữ liệu lịch sử (soft — không block) ─────────
        best_fp    = fp1
        best_level = 1
        best_data  = self._get_pattern(fp1)

        if best_data is None:
            best_data  = self._get_pattern(fp2)
            best_fp    = fp2
            best_level = 2

        if best_data is None:
            best_data  = self._get_pattern(fp3)
            best_fp    = fp3
            best_level = 3

        if best_data is None:
            # Không có dữ liệu — cho phép nhưng không boost
            return MemoryVerdict(
                hard_block     = False,
                approved       = True,
                win_rate       = -1.0,
                sample_count   = 0,
                priority_boost = 0.0,
                fingerprint    = fp1,
                matched_level  = 0,
                reason         = "✅ Không có dữ liệu lịch sử — cho phép",
            )

        wins  = best_data["win_count"]
        losses= best_data["loss_count"]
        n     = wins + losses
        wr    = wins / n if n > 0 else 0.5

        # Tính priority boost dựa trên lịch sử
        if wr >= config.MEMORY_STRONG_WIN_RATE and n >= config.MEMORY_MIN_SAMPLES_FOR_RULE:
            boost  = round((wr - 0.5) * 20, 2)   # Tối đa +10 khi WR=100%
            reason = (
                f"✅ L{best_level}: WR={wr*100:.1f}%  n={n}  "
                f"boost={boost:+.1f}"
            )
        elif wr < 0.50 and n >= config.MEMORY_MIN_SAMPLES_FOR_RULE:
            boost  = round((wr - 0.5) * 20, 2)   # Âm khi WR < 50%
            reason = (
                f"⚠️  L{best_level}: WR={wr*100:.1f}%  n={n}  "
                f"boost={boost:.1f} (yếu nhưng chưa đủ để chặn cứng)"
            )
        else:
            boost  = 0.0
            reason = f"✅ L{best_level}: WR={wr*100:.1f}%  n={n}  (trung tính)"

        return MemoryVerdict(
            hard_block     = False,
            approved       = True,
            win_rate       = round(wr, 4),
            sample_count   = n,
            priority_boost = boost,
            fingerprint    = best_fp,
            matched_level  = best_level,
            reason         = reason,
        )

    # ── Tạo features từ MarketSignal ─────────────────────────────

    @staticmethod
    def features_from_signal(signal: object) -> TradeFeatures:
        """
        Tạo TradeFeatures từ MarketSignal (brain.MarketSignal).

        Dùng khi cần tham vấn memory trước khi đặt lệnh.
        """
        wave     = getattr(signal, "wave", None)
        fib_zone = wave.fib_zone if wave else "NONE"
        wave_act = bool(wave and wave.correction_active)
        rsi      = float(getattr(signal, "rsi",      50.0) or 50.0)
        momentum = float(getattr(signal, "momentum",  0.0) or 0.0)
        hour     = datetime.utcnow().hour

        return TradeFeatures(
            symbol      = signal.symbol,
            direction   = signal.direction,
            score       = signal.score,
            fib_zone    = fib_zone,
            wave_active = wave_act,
            rsi         = rsi,
            momentum    = momentum,
            hour        = hour,
        )

    # ── Báo cáo tổng hợp ─────────────────────────────────────────

    def report(self) -> None:
        """In báo cáo tổng hợp bộ nhớ win/loss."""
        raw_stats = self._r.get(config.REDIS_MEMORY_STATS_KEY)
        stats     = json.loads(raw_stats) if raw_stats else {}

        hard_rules   = list(self._hard_rules)
        total_pats   = stats.get("total_patterns", self._count_patterns())
        last_rebuilt = stats.get("last_rebuilt", "—")

        print(f"\n  {'─'*62}")
        print(f"  🧠 REDIS MEMORY BRAIN — Win/Loss Pattern Report")
        print(f"  {'─'*62}")
        print(f"  Tổng patterns : {total_pats}")
        print(f"  Luật cứng     : {len(hard_rules)}")
        print(f"  Cập nhật lần : {last_rebuilt}")

        if hard_rules:
            print(f"\n  ❌ LUẬT CỨNG — Các fingerprint bị chặn tuyệt đối:")
            for fp in sorted(hard_rules)[:20]:
                data = self._get_pattern(fp)
                if data:
                    n  = data["win_count"] + data["loss_count"]
                    wr = data["win_count"] / n * 100 if n > 0 else 0.0
                    print(f"     {fp[:55]}  n={n}  WR={wr:.1f}%")

        # Top win patterns
        top_wins = self._get_top_patterns(win=True, top_n=5)
        if top_wins:
            print(f"\n  ✅ TOP 5 PATTERNS WIN RATE CAO NHẤT:")
            for fp, data in top_wins:
                n  = data["win_count"] + data["loss_count"]
                wr = data["win_count"] / n * 100 if n > 0 else 0.0
                print(f"     {fp[:55]}  n={n:>3}  WR={wr:.1f}%")

        # Top loss patterns
        top_loss = self._get_top_patterns(win=False, top_n=5)
        if top_loss:
            print(f"\n  ❌ TOP 5 PATTERNS LOSS RATE CAO NHẤT:")
            for fp, data in top_loss:
                n  = data["win_count"] + data["loss_count"]
                lr = data["loss_count"] / n * 100 if n > 0 else 0.0
                print(f"     {fp[:55]}  n={n:>3}  LR={lr:.1f}%")

        print(f"  {'─'*62}")

    def _get_top_patterns(
        self,
        win: bool,
        top_n: int = 5,
        min_samples: int = 3,
    ) -> list[tuple[str, dict]]:
        """Lấy top N patterns theo win rate hoặc loss rate."""
        prefix  = config.REDIS_MEMORY_PREFIX
        results = []

        for key_bytes in self._r.scan_iter(match=prefix + "*"):
            key_str     = key_bytes.decode()
            fingerprint = key_str[len(prefix):]
            data        = self._r.hgetall(key_bytes)
            if not data:
                continue

            wins   = int(data.get(b"win_count",  0))
            losses = int(data.get(b"loss_count", 0))
            n      = wins + losses
            if n < min_samples:
                continue

            rate = wins / n if win else losses / n
            results.append((fingerprint, {
                "win_count" : wins,
                "loss_count": losses,
                "total_pnl" : float(data.get(b"total_pnl", 0)),
                "rate"      : rate,
            }))

        results.sort(key=lambda x: x[1]["rate"], reverse=True)
        return results[:top_n]


# ──────────────────────────────────────────────────────────────────
# Chạy trực tiếp để kiểm tra
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    brain = MemoryBrain()

    # Mô phỏng ghi nhận 30 lệnh ngẫu nhiên
    symbols   = ["R_10", "R_25", "R_50", "R_75", "R_100"]
    fib_zones = ["F382", "F618", "NONE"]
    print("Mô phỏng ghi nhận 30 lệnh...")

    for i in range(30):
        f = TradeFeatures(
            symbol      = random.choice(symbols),
            direction   = random.choice(["CALL", "PUT"]),
            score       = random.uniform(60, 95),
            fib_zone    = random.choice(fib_zones),
            wave_active = random.choice([True, False]),
            rsi         = random.uniform(25, 75),
            momentum    = random.uniform(-0.01, 0.01),
            hour        = random.randint(0, 23),
        )
        # Tạo bias: R_50 PUT thường thua
        if f.symbol == "R_50" and f.direction == "PUT":
            won = random.random() > 0.80   # 80% thua
        else:
            won = random.random() > 0.45
        pnl = 8.5 if won else -10.0
        brain.record_outcome(f, won=won, pnl=pnl)

    # Tham vấn bộ nhớ
    test = TradeFeatures(
        symbol="R_50", direction="PUT", score=70, fib_zone="NONE",
        wave_active=False, rsi=50, momentum=0.0, hour=10,
    )
    verdict = brain.consult(test)
    print(f"\nTham vấn R_50 PUT:")
    print(f"  hard_block={verdict.hard_block}")
    print(f"  approved={verdict.approved}")
    print(f"  win_rate={verdict.win_rate:.2%}")
    print(f"  reason={verdict.reason}")

    brain.report()

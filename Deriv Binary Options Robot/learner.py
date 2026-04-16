"""
learner.py
==========
Tự học — Self-Learn

Phân tích lịch sử giao dịch để nhận diện điều kiện tín hiệu tốt/xấu.
Tự động điều chỉnh ngưỡng chấm điểm và hệ số kích thước lệnh.

Phân tích theo chiều:
  - score_band    : 60-70, 71-80, 81-90, 91-100
  - fib_zone      : F236, F382, F500, F618, F786, NONE
  - wave_active   : True / False

Tham số tự điều chỉnh:
  - effective_min_score : ngưỡng điểm thực tế cho chu kỳ hiện tại
  - stake_multiplier    : hệ số nhân stake dựa trên performance gần đây (0.5×–1.5×)
  - weak_conditions     : list điều kiện có win_rate < ngưỡng

Các tham số được lưu vào Redis để bền vững qua restart.
"""

from __future__ import annotations

import json
import redis
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional

import config

_REDIS_LEARNED_KEY = "Deriv_Learned_Params"
_MIN_SAMPLES       = 5   # Cần ít nhất N mẫu để đánh giá một điều kiện


# ──────────────────────────────────────────────────────────────────
# Dataclass tham số đã học
# ──────────────────────────────────────────────────────────────────

@dataclass
class LearnedParams:
    effective_min_score: float      = config.MIN_SIGNAL_SCORE
    stake_multiplier:    float      = 1.0
    weak_conditions:     list[str]  = field(default_factory=list)
    last_updated:        str        = ""


# ──────────────────────────────────────────────────────────────────
# Learner
# ──────────────────────────────────────────────────────────────────

class Learner:
    """
    Self-learning từ lịch sử giao dịch.

    Sử dụng:
        learner = Learner()
        params  = learner.get_params()          # Lấy params hiện tại
        learner.run_learning_cycle()            # Học lại từ lịch sử
        weak = learner.is_condition_weak(...)   # Kiểm tra điều kiện
    """

    def __init__(self) -> None:
        self._r      = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
        )
        self._params = self._load_params()

    # ── Persistence ───────────────────────────────────────────────

    def _load_params(self) -> LearnedParams:
        raw = self._r.get(_REDIS_LEARNED_KEY)
        if raw:
            try:
                data = json.loads(raw)
                fields = LearnedParams.__dataclass_fields__
                return LearnedParams(**{k: v for k, v in data.items() if k in fields})
            except Exception:
                pass
        return LearnedParams()

    def _save_params(self, params: LearnedParams) -> None:
        self._r.set(_REDIS_LEARNED_KEY, json.dumps(asdict(params)))

    def get_params(self) -> LearnedParams:
        return self._params

    # ── Load history ──────────────────────────────────────────────

    def _load_trade_history(self) -> list[dict]:
        """Tải lịch sử giao dịch từ Redis."""
        raw_list = self._r.lrange(config.REDIS_LOG_KEY, 0, -1)
        records  = []
        for raw in raw_list:
            try:
                records.append(json.loads(raw))
            except Exception:
                pass
        return records

    # ── Condition key ─────────────────────────────────────────────

    @staticmethod
    def _condition_key(record: dict) -> str:
        """Tạo key điều kiện từ một trade record."""
        score = float(record.get("signal_score", 0))
        if score >= 91:
            band = "91_100"
        elif score >= 81:
            band = "81_90"
        elif score >= 71:
            band = "71_80"
        else:
            band = "60_70"

        indicators = record.get("indicators", {})
        fib   = indicators.get("fib_zone", "NONE") if isinstance(indicators, dict) else "NONE"
        wave  = "wave" if (isinstance(indicators, dict) and indicators.get("correction")) else "nowave"
        return f"{band}:{fib}:{wave}"

    # ── Learning cycle ────────────────────────────────────────────

    def run_learning_cycle(self) -> LearnedParams:
        """
        Phân tích lịch sử và cập nhật LearnedParams.

        Quy trình:
          1. Tải lịch sử từ Redis
          2. Nhóm theo điều kiện, tính win rate mỗi nhóm
          3. Điều chỉnh effective_min_score theo win rate gần đây
          4. Điều chỉnh stake_multiplier theo win rate gần đây
          5. Lưu kết quả vào Redis

        Returns LearnedParams mới sau khi học.
        """
        records = self._load_trade_history()
        if len(records) < config.LEARNER_MIN_HISTORY:
            print(
                f"  [Learner] Chưa đủ dữ liệu "
                f"({len(records)}/{config.LEARNER_MIN_HISTORY} lệnh) — giữ params cũ."
            )
            return self._params

        # ── Phân nhóm theo điều kiện ──────────────────────────────
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in records:
            key = self._condition_key(r)
            groups[key].append(bool(r.get("won", False)))

        # ── Xác định điều kiện yếu ────────────────────────────────
        weak_conditions: list[str] = []
        print("\n  [Learner] Phân tích theo điều kiện tín hiệu:")
        for cond, results in sorted(groups.items()):
            n    = len(results)
            wins = sum(results)
            wr   = wins / n * 100 if n > 0 else 0.0
            tag  = "⚠️ " if (n >= _MIN_SAMPLES and wr < config.LEARNER_WEAK_WIN_RATE * 100) else "✅"
            print(f"    {tag} [{cond}] n={n:>3}  WR={wr:>5.1f}%")
            if n >= _MIN_SAMPLES and wr < config.LEARNER_WEAK_WIN_RATE * 100:
                weak_conditions.append(cond)

        # ── Win rate 20 lệnh gần nhất ─────────────────────────────
        recent      = records[: min(20, len(records))]
        recent_wins = sum(1 for r in recent if r.get("won"))
        recent_wr   = recent_wins / len(recent) * 100 if recent else 0.0

        # ── Điều chỉnh effective_min_score ────────────────────────
        current_min = self._params.effective_min_score
        if recent_wr >= 60.0:
            new_min_score = max(config.MIN_SIGNAL_SCORE, current_min - 2.0)
        elif recent_wr < 45.0:
            new_min_score = min(85.0, current_min + 5.0)
        else:
            new_min_score = current_min

        # ── Điều chỉnh stake_multiplier ───────────────────────────
        current_mult = self._params.stake_multiplier
        if recent_wr >= 65.0:
            stake_mult = min(1.5, current_mult + 0.1)
        elif recent_wr < 40.0:
            stake_mult = max(0.5, current_mult - 0.2)
        else:
            stake_mult = current_mult

        new_params = LearnedParams(
            effective_min_score = round(new_min_score, 1),
            stake_multiplier    = round(stake_mult, 2),
            weak_conditions     = weak_conditions,
            last_updated        = datetime.now().isoformat(),
        )

        self._save_params(new_params)
        self._params = new_params

        print(
            f"\n  [Learner] ✅ Học xong:\n"
            f"    Min score  : {new_params.effective_min_score}\n"
            f"    Stake mult : {new_params.stake_multiplier}×\n"
            f"    Weak conds : {len(weak_conditions)}\n"
            f"    Recent WR  : {recent_wr:.1f}% (20 lệnh gần nhất)"
        )
        return new_params

    # ── Condition check ───────────────────────────────────────────

    def is_condition_weak(self,
                          score: float,
                          fib_zone: str,
                          wave_active: bool) -> bool:
        """
        Kiểm tra điều kiện hiện tại có trong danh sách yếu không.

        Returns True → nên bỏ qua lệnh này.
        """
        if score >= 91:
            band = "91_100"
        elif score >= 81:
            band = "81_90"
        elif score >= 71:
            band = "71_80"
        else:
            band = "60_70"

        wave = "wave" if wave_active else "nowave"
        key  = f"{band}:{fib_zone}:{wave}"
        return key in self._params.weak_conditions


# ──────────────────────────────────────────────────────────────────
# Chạy trực tiếp để kiểm tra
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    learner = Learner()
    print("Params hiện tại:")
    p = learner.get_params()
    print(f"  min_score={p.effective_min_score}  stake_mult={p.stake_multiplier}×")
    print(f"  weak_conds={p.weak_conditions}")
    print("\nĐang chạy learning cycle...")
    new_p = learner.run_learning_cycle()
    print(f"\nParams mới: {new_p}")

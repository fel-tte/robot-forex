"""
predictor.py
============
Tự dự đoán — Self-Predict

Ước tính xác suất thắng của một MarketSignal dựa trên:
  1. Điểm confluence (score từ brain.py)
  2. Trạng thái sóng (wave depth + Fibonacci zone)
  3. Win rate lịch sử của điều kiện tương tự (từ Learner)
  4. Biến động thị trường hiện tại (ATR tương đối)

Trả về Prediction chứa:
  - win_prob        : xác suất thắng [0-1]
  - confidence      : mức tự tin [0-1]
  - stake_suggestion: stake được điều chỉnh theo confidence + stake_multiplier
  - should_trade    : bool — có nên vào lệnh không

Công thức cơ bản:
  win_prob = base_from_score + wave_boost + history_adj + vol_adj
  base: score=60 → ~52%, score=80 → ~61%, score=100 → ~72%
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import config

if TYPE_CHECKING:
    from brain   import MarketSignal
    from learner import Learner


# ──────────────────────────────────────────────────────────────────
# Dataclass kết quả dự đoán
# ──────────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    win_prob:            float   # Xác suất thắng [0-1]
    confidence:          float   # Mức tự tin [0-1]
    base_score:          float   # Điểm gốc từ brain
    wave_boost:          float   # Điều chỉnh từ sóng
    history_adjustment:  float   # Điều chỉnh từ lịch sử học
    vol_adjustment:      float   # Điều chỉnh từ biến động
    stake_suggestion:    float   # Stake gợi ý (USD)
    should_trade:        bool
    reason:              str


# ──────────────────────────────────────────────────────────────────
# ATR helper
# ──────────────────────────────────────────────────────────────────

def _relative_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Tính ATR tương đối (ATR / close) để đo biến động."""
    high  = df["high"]  if "high"  in df.columns else df["close"]
    low   = df["low"]   if "low"   in df.columns else df["close"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr   = tr.rolling(period).mean().iloc[-1]
    price = float(close.iloc[-1])
    return float(atr / price) if price > 0 else 0.0


# ──────────────────────────────────────────────────────────────────
# Predictor chính
# ──────────────────────────────────────────────────────────────────

def predict(
    signal: "MarketSignal",
    df: pd.DataFrame,
    learner: Optional["Learner"] = None,
    current_balance: float = 0.0,
) -> Prediction:
    """
    Dự đoán kết quả cho một MarketSignal.

    Parameters
    ----------
    signal          : MarketSignal từ brain.py
    df              : DataFrame nến hiện tại (để tính ATR, biến động)
    learner         : Learner instance — cung cấp dữ liệu lịch sử
    current_balance : Số dư hiện tại — dùng để tính stake gợi ý

    Returns
    -------
    Prediction
    """
    base_score = signal.score   # 0-100

    # ── 1. Xác suất cơ bản từ điểm tín hiệu ─────────────────────
    # score=60 → 52%, score=80 → 61%, score=100 → 72%
    # Công thức tuyến tính trên [60, 100] → [0.52, 0.72]
    base_win_prob = float(np.clip(0.50 + (base_score - 60) / 40 * 0.22, 0.45, 0.75))

    # ── 2. Điều chỉnh từ sóng ────────────────────────────────────
    wave_boost  = 0.0
    wave_active = False
    fib_zone    = "NONE"

    if signal.wave and signal.wave.correction_active:
        wave_active = True
        fib_zone    = signal.wave.fib_zone
        depth       = signal.wave.correction_depth_pct
        depth_ideal = 38.0 <= depth <= 62.0
        fib_strong  = fib_zone in ("F618", "F500")
        at_sr       = signal.wave.at_support_resistance

        if fib_strong and at_sr and depth_ideal:
            wave_boost = 0.08   # Tất cả xác nhận → tăng mạnh
        elif fib_strong and (at_sr or depth_ideal):
            wave_boost = 0.05
        elif fib_zone in ("F382", "F786") and depth_ideal:
            wave_boost = 0.03
        elif depth > 70.0:
            wave_boost = -0.05  # Sóng hồi quá sâu → rủi ro cao hơn

    # ── 3. Điều chỉnh từ lịch sử học ────────────────────────────
    history_adj = 0.0
    stake_mult  = 1.0

    if learner is not None:
        params     = learner.get_params()
        stake_mult = params.stake_multiplier
        if learner.is_condition_weak(base_score, fib_zone, wave_active):
            history_adj = -0.07   # Điều kiện từng yếu → hạ xác suất

    # ── 4. Điều chỉnh từ biến động ───────────────────────────────
    vol_adj = 0.0
    try:
        rel_atr = _relative_atr(df)
        if rel_atr > config.PREDICT_HIGH_VOLATILITY_ATR:
            vol_adj = -0.04   # Biến động cao → kém ổn định
        elif rel_atr < config.PREDICT_LOW_VOLATILITY_ATR:
            vol_adj = +0.02   # Biến động thấp → ổn định hơn
    except Exception:
        pass

    # ── Tổng hợp xác suất ────────────────────────────────────────
    win_prob   = float(np.clip(base_win_prob + wave_boost + history_adj + vol_adj, 0.40, 0.80))

    # ── Mức tự tin ───────────────────────────────────────────────
    # Kết hợp: khoảng cách từ 0.5 (uncertainty floor) + điểm tín hiệu
    confidence = float(np.clip(
        (win_prob - 0.40) / 0.35 * (base_score / 100.0),
        0.0, 1.0,
    ))

    # ── Stake gợi ý ──────────────────────────────────────────────
    stake_suggestion = 0.0
    if current_balance > 0:
        if base_score >= 80:
            pct = config.STAKE_PCT_HIGH
        elif base_score >= 60:
            pct = config.STAKE_PCT_MEDIUM
        else:
            pct = config.STAKE_PCT_LOW

        raw              = current_balance * pct * stake_mult * confidence
        stake_suggestion = float(np.clip(raw, config.STAKE_MIN_USD, config.STAKE_MAX_USD))

    # ── Quyết định có nên vào lệnh không ─────────────────────────
    eff_min_score = learner.get_params().effective_min_score if learner else config.MIN_SIGNAL_SCORE
    should_trade  = (
        win_prob   >= config.PREDICT_MIN_WIN_PROB
        and base_score >= eff_min_score
        and confidence >= config.PREDICT_MIN_CONFIDENCE
    )

    reason = (
        f"win_prob={win_prob:.3f}  conf={confidence:.3f}  "
        f"score={base_score}  wave={wave_boost:+.3f}  "
        f"hist={history_adj:+.3f}  vol={vol_adj:+.3f}"
    )

    return Prediction(
        win_prob           = round(win_prob, 4),
        confidence         = round(confidence, 4),
        base_score         = base_score,
        wave_boost         = round(wave_boost, 4),
        history_adjustment = round(history_adj, 4),
        vol_adjustment     = round(vol_adj, 4),
        stake_suggestion   = round(stake_suggestion, 2),
        should_trade       = should_trade,
        reason             = reason,
    )


# ──────────────────────────────────────────────────────────────────
# Chạy trực tiếp để kiểm tra
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import deriv_data
    from brain   import pick_best_entry
    from learner import Learner

    df      = deriv_data.fetch_candles()
    learner = Learner()
    best    = pick_best_entry()

    if best:
        pred = predict(best, df, learner=learner, current_balance=100.0)
        print("\n─── Kết quả dự đoán ───")
        print(f"  Tín hiệu : {best.symbol} {best.direction} score={best.score}")
        print(f"  Win prob  : {pred.win_prob:.1%}")
        print(f"  Confidence: {pred.confidence:.1%}")
        print(f"  Stake gợi ý: {pred.stake_suggestion:.2f} USD")
        print(f"  Nên vào  : {'✅ CÓ' if pred.should_trade else '🚫 KHÔNG'}")
        print(f"  Lý do    : {pred.reason}")
    else:
        print("Không có tín hiệu.")

"""
brain.py
========
Bộ não tự vận hành của Robot Deriv — Operator System.

Nhiệm vụ:
  1. Tự quét nhiều thị trường cùng lúc
  2. Tính điểm chất lượng tín hiệu (0-100) cho từng thị trường
     Tầng 1 — Chỉ báo kỹ thuật cơ bản (tối đa 60 điểm):
       RSI crossover     : 0–18 điểm
       Momentum strength : 0–12 điểm
       MACD confirmation : 0–15 điểm
       Bollinger position: 0–15 điểm
     Tầng 2 — Phân tích sóng (tối đa 40 điểm):
       Fibonacci zone    : 0–20 điểm
       Correction depth  : 0–10 điểm
       S/R cluster       : 0–10 điểm
  3. Tự chọn thị trường + hướng lệnh tốt nhất
  4. Trả về MarketSignal bao gồm cả WaveContext

Ghi chú: Khi sóng hồi active và hướng sóng khớp hướng kỹ thuật →
         điểm tổng có thể đạt 80-100. Nếu không có sóng hồi →
         điểm tối đa chỉ là 60.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import config
import deriv_data
from wave_analyzer import WaveContext, analyze_waves


# ------------------------------------------------------------------
# Dataclass kết quả phân tích
# ------------------------------------------------------------------

@dataclass
class MarketSignal:
    symbol: str
    direction: str          # 'CALL', 'PUT', hoặc 'NONE'
    score: float            # 0–100
    rsi: float
    momentum: float
    macd_hist: float
    bb_position: float      # 0=dưới dải dưới, 1=trên dải trên, 0.5=giữa
    wave: Optional[WaveContext] = None
    indicators: dict = field(default_factory=dict)

    def is_tradeable(self) -> bool:
        return self.direction != "NONE" and self.score >= config.MIN_SIGNAL_SCORE


# ------------------------------------------------------------------
# Tính chỉ báo kỹ thuật (Tầng 1 — tối đa 60 điểm)
# ------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_g / avg_l
    return 100 - (100 / (1 + rs))


def _momentum(series: pd.Series, period: int = config.MOMENTUM_PERIOD) -> pd.Series:
    return series - series.shift(period)


def _macd(series: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """Trả về (macd_line, histogram)."""
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_line - signal_line


def _bollinger(series: pd.Series,
               period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Trả về (upper_band, middle_band, lower_band)."""
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ------------------------------------------------------------------
# Tính điểm tín hiệu cho một thị trường
# ------------------------------------------------------------------

def _score_signal(df: pd.DataFrame) -> MarketSignal:
    """
    Phân tích DataFrame nến và tính điểm tín hiệu (0-100).

    Tầng 1 (kỹ thuật cơ bản) tối đa 60 điểm.
    Tầng 2 (phân tích sóng)   tối đa 40 điểm.
    Tổng được giới hạn tại 100.
    """
    close = df["close"]

    rsi_series             = _rsi(close)
    mom_series             = _momentum(close)
    _, macd_hist           = _macd(close)
    bb_upper, bb_mid, bb_lower = _bollinger(close)

    rsi_now   = rsi_series.iloc[-1]
    rsi_prev  = rsi_series.iloc[-2]
    mom_now   = mom_series.iloc[-1]
    hist_now  = macd_hist.iloc[-1]
    hist_prev = macd_hist.iloc[-2]
    price     = close.iloc[-1]
    bb_up     = bb_upper.iloc[-1]
    bb_dn     = bb_lower.iloc[-1]
    bb_range  = bb_up - bb_dn if (bb_up - bb_dn) != 0 else 1

    call_score = 0.0
    put_score  = 0.0

    # ── Tầng 1a: RSI crossover (max 18 điểm) ──────────────────────
    rsi_cross_up   = rsi_prev < config.RSI_OVERSOLD  and rsi_now >= config.RSI_OVERSOLD
    rsi_cross_down = rsi_prev > config.RSI_OVERBOUGHT and rsi_now <= config.RSI_OVERBOUGHT
    rsi_near_os    = rsi_now < 35
    rsi_near_ob    = rsi_now > 65

    if rsi_cross_up:
        call_score += 18
    elif rsi_near_os:
        call_score += 9 * (35 - rsi_now) / 35

    if rsi_cross_down:
        put_score += 18
    elif rsi_near_ob:
        put_score += 9 * (rsi_now - 65) / 35

    # ── Tầng 1b: Momentum (max 12 điểm) ───────────────────────────
    mom_std = mom_series.rolling(20).std().iloc[-1]
    if mom_std and not np.isnan(mom_std) and mom_std > 0:
        mom_z   = abs(mom_now) / mom_std
        mom_pts = min(12, 6 * mom_z)
        if mom_now > 0:
            call_score += mom_pts
        elif mom_now < 0:
            put_score  += mom_pts

    # ── Tầng 1c: MACD histogram (max 15 điểm) ─────────────────────
    macd_cross_up      = hist_prev < 0 and hist_now >= 0
    macd_cross_down    = hist_prev > 0 and hist_now <= 0
    macd_trending_up   = hist_now > 0 and hist_now > hist_prev
    macd_trending_down = hist_now < 0 and hist_now < hist_prev

    if macd_cross_up:
        call_score += 15
    elif macd_trending_up:
        call_score += 6

    if macd_cross_down:
        put_score += 15
    elif macd_trending_down:
        put_score += 6

    # ── Tầng 1d: Bollinger Bands (max 15 điểm) ────────────────────
    bb_pos = (price - bb_dn) / bb_range
    if bb_pos <= 0.1:
        call_score += 15 * (1 - bb_pos / 0.1)
    elif bb_pos <= 0.3:
        call_score += 6

    if bb_pos >= 0.9:
        put_score += 15 * ((bb_pos - 0.9) / 0.1)
    elif bb_pos >= 0.7:
        put_score += 6

    # ── Tầng 2: Phân tích sóng (max 40 điểm) ─────────────────────
    wave_ctx: Optional[WaveContext] = None
    try:
        wave_ctx = analyze_waves(df)
        if wave_ctx.is_wave_entry():
            wave_score = wave_ctx.entry_score   # 0-40
            if wave_ctx.entry_direction == "CALL":
                call_score += wave_score
            elif wave_ctx.entry_direction == "PUT":
                put_score  += wave_score
    except Exception:
        pass   # Không để lỗi phân tích sóng làm gián đoạn chấm điểm

    # ── Quyết định hướng ──────────────────────────────────────────
    if call_score >= put_score and call_score >= config.MIN_SIGNAL_SCORE:
        direction = "CALL"
        score     = min(100.0, call_score)
    elif put_score > call_score and put_score >= config.MIN_SIGNAL_SCORE:
        direction = "PUT"
        score     = min(100.0, put_score)
    else:
        direction = "NONE"
        score     = max(call_score, put_score)

    return MarketSignal(
        symbol      = "",
        direction   = direction,
        score       = round(score, 2),
        rsi         = round(rsi_now, 4),
        momentum    = round(mom_now, 4),
        macd_hist   = round(hist_now, 6),
        bb_position = round(bb_pos, 4),
        wave        = wave_ctx,
        indicators  = {
            "call_score"  : round(call_score, 2),
            "put_score"   : round(put_score, 2),
            "rsi_prev"    : round(rsi_prev, 4),
            "bb_upper"    : round(bb_up, 6),
            "bb_lower"    : round(bb_dn, 6),
            "wave_score"  : wave_ctx.entry_score if wave_ctx else 0,
            "wave_dir"    : wave_ctx.entry_direction if wave_ctx else "NONE",
            "fib_zone"    : wave_ctx.fib_zone if wave_ctx else "NONE",
            "correction"  : wave_ctx.correction_active if wave_ctx else False,
        },
    )


# ------------------------------------------------------------------
# Quét nhiều thị trường và chọn điểm vào lệnh tốt nhất
# ------------------------------------------------------------------

def scan_all_markets(symbols: list[str] = config.SCAN_SYMBOLS) -> list[MarketSignal]:
    """
    Quét tất cả thị trường trong danh sách, tính điểm từng thị trường.

    Returns danh sách MarketSignal đã sắp xếp theo điểm giảm dần.
    """
    results = []
    for sym in symbols:
        try:
            df  = deriv_data.fetch_candles(symbol=sym)
            sig = _score_signal(df)
            sig.symbol = sym

            wave_info = ""
            if sig.wave and sig.wave.correction_active:
                wave_info = (
                    f" | 🌊 {sig.wave.main_direction} hồi {sig.wave.correction_depth_pct:.1f}%"
                    f" [{sig.wave.fib_zone}] wave={sig.wave.entry_score}"
                )

            results.append(sig)
            print(
                f"  [{sym}] dir={sig.direction:<4} score={sig.score:>5.1f} "
                f"RSI={sig.rsi:.1f} mom={sig.momentum:.4f}"
                f"{wave_info}"
            )
        except Exception as exc:
            print(f"  [{sym}] ⚠️  Không lấy được dữ liệu: {exc}")

    results.sort(key=lambda s: s.score, reverse=True)
    return results


def pick_best_entry(symbols: list[str] = config.SCAN_SYMBOLS) -> Optional[MarketSignal]:
    """
    Quét tất cả thị trường và trả về tín hiệu tốt nhất.

    Returns None nếu không có thị trường nào đủ điều kiện.
    """
    candidates = scan_all_markets(symbols)
    tradeable  = [s for s in candidates if s.is_tradeable()]
    if not tradeable:
        return None
    best = tradeable[0]
    wave_summary = ""
    if best.wave and best.wave.correction_active:
        wave_summary = (
            f"\n   🌊 Sóng: {best.wave.description}"
            f"\n   📍 TP={best.wave.tp_price} | SL={best.wave.sl_price}"
        )
    print(
        f"\n🏆 Thị trường tốt nhất: {best.symbol} | "
        f"{best.direction} | score={best.score}"
        f"{wave_summary}"
    )
    return best


# ------------------------------------------------------------------
# Chạy trực tiếp để kiểm tra
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Đang quét {len(config.SCAN_SYMBOLS)} thị trường...\n")
    best = pick_best_entry()
    if best:
        print(f"\nKết quả: {best}")
    else:
        print("\nKhông có tín hiệu đủ điều kiện giao dịch.")

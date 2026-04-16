"""
wave_analyzer.py
================
Phân tích sóng tự động — trung tâm của hệ thống Operator.

Khái niệm:
  • Sóng chính (Main Wave) : xu hướng chủ đạo được xác định bằng chuỗi
    đỉnh/đáy tăng dần (uptrend) hoặc giảm dần (downtrend).

  • Sóng hồi (Correction Wave) : khi gặp cản mạnh, giá kéo ngược lại
    sóng chính. Sóng hồi kết thúc khi giá chạm vùng Fibonacci hỗ trợ
    và bắt đầu hồi phục theo hướng sóng chính.

Hệ thống tự:
  ① Tự phát hiện sóng hồi   – detect_swings() + analyze_waves()
  ② Tự đo lường              – Fibonacci 23.6 / 38.2 / 50 / 61.8 / 78.6 %
  ③ Tự giới hạn              – chỉ vào lệnh khi sóng hồi trong biên [20%, 80%]
  ④ Tự điều phối             – entry_score 0-40, chỉ xét khi correction_active
  ⑤ Tự tìm điểm vào/ra an toàn – entry tại cuối sóng hồi, TP tại đỉnh/đáy cũ

Công thức điểm sóng (tổng tối đa 40):
  Fibonacci zone      : 0-20 điểm  (61.8% = 20pt, 50% = 15pt, 38.2% = 10pt)
  Correction trong biên: 0-10 điểm (depth 38-62% → 10pt)
  S/R cluster         : 0-10 điểm  (giá ở gần pivot cluster → +10pt)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import config


# ──────────────────────────────────────────────────────────────────
# Dataclass kết quả phân tích sóng
# ──────────────────────────────────────────────────────────────────

@dataclass
class WaveContext:
    # Sóng chính
    main_direction: str        # 'UP' | 'DOWN' | 'RANGING'
    main_wave_size: float      # Kích thước sóng chính (price units)
    swing_high: float          # Đỉnh sóng chính gần nhất
    swing_low: float           # Đáy sóng chính gần nhất

    # Sóng hồi
    correction_active: bool         # Đang có sóng hồi?
    correction_depth_pct: float     # Độ sâu sóng hồi (% của sóng chính, 0-100)
    fib_zone: str                   # 'F236','F382','F500','F618','F786','NONE','DEEP'
    at_support_resistance: bool     # Giá ở gần vùng S/R?

    # Khuyến nghị giao dịch
    entry_direction: str       # 'CALL' | 'PUT' | 'NONE'
    entry_score: float         # 0–40 điểm (đóng góp vào điểm tổng của brain.py)
    tp_price: float            # Suggested take-profit (tiếp tục sóng chính)
    sl_price: float            # Suggested stop-loss (vỡ cấu trúc sóng hồi)

    # Mô tả
    description: str = ""
    sr_levels: list[float] = field(default_factory=list)

    def is_wave_entry(self) -> bool:
        """Có tín hiệu vào lệnh từ phân tích sóng không?"""
        return self.correction_active and self.entry_direction != "NONE" and self.entry_score >= 10


# ──────────────────────────────────────────────────────────────────
# Phát hiện đỉnh / đáy (Swing Detection — ZigZag)
# ──────────────────────────────────────────────────────────────────

def detect_swings(close: pd.Series,
                  order: int = config.WAVE_SWING_ORDER) -> pd.DataFrame:
    """
    Tìm các đỉnh (Swing High) và đáy (Swing Low) trong chuỗi giá.

    Thuật toán: rolling-window — nến i là đỉnh nếu nó là cao nhất trong
    cửa sổ [i-order, i+order].

    Returns
    -------
    DataFrame: columns=['datetime', 'price', 'type']
               type = 'H' (high) hoặc 'L' (low)
    """
    n    = len(close)
    rows = []
    for i in range(order, n - order):
        window = close.iloc[i - order: i + order + 1]
        p = close.iloc[i]
        if p == window.max():
            rows.append((close.index[i], p, "H"))
        elif p == window.min():
            rows.append((close.index[i], p, "L"))

    if not rows:
        return pd.DataFrame(columns=["datetime", "price", "type"])

    # Sắp xếp theo thời gian
    rows.sort(key=lambda x: x[0])

    # Loại bỏ đỉnh/đáy liên tiếp cùng loại — giữ cực trị mạnh hơn
    filtered: list[tuple] = [rows[0]]
    for r in rows[1:]:
        last = filtered[-1]
        if r[2] == last[2]:
            # Cùng loại → giữ cái cực trị hơn
            if r[2] == "H" and r[1] >= last[1]:
                filtered[-1] = r
            elif r[2] == "L" and r[1] <= last[1]:
                filtered[-1] = r
        else:
            filtered.append(r)

    return pd.DataFrame(filtered, columns=["datetime", "price", "type"])


# ──────────────────────────────────────────────────────────────────
# Tính mức Fibonacci
# ──────────────────────────────────────────────────────────────────

_FIB_RATIOS = {
    "F236": 0.236,
    "F382": 0.382,
    "F500": 0.500,
    "F618": 0.618,
    "F786": 0.786,
}


def fibonacci_levels(swing_start: float, swing_end: float) -> dict[str, float]:
    """
    Tính các mức Fibonacci retracement.

    swing_start → điểm BẮT ĐẦU sóng chính (đáy nếu UP, đỉnh nếu DOWN)
    swing_end   → điểm KẾT THÚC sóng chính (đỉnh nếu UP, đáy nếu DOWN)

    Returns dict {'F236': price, 'F382': price, ...}
    """
    diff = swing_end - swing_start
    return {
        name: round(swing_end - ratio * diff, 8)
        for name, ratio in _FIB_RATIOS.items()
    }


def nearest_fib_zone(price: float,
                     levels: dict[str, float],
                     tolerance: float = config.WAVE_FIB_TOLERANCE) -> str:
    """
    Trả về tên vùng Fibonacci mà giá đang ở gần nhất.

    tolerance : ±% của khoảng cách sóng coi là "tại vùng"
    """
    swing_range = max(levels.values()) - min(levels.values())
    tol_abs     = swing_range * tolerance if swing_range > 0 else 0

    for name, level in sorted(_FIB_RATIOS.items(), key=lambda x: x[1], reverse=True):
        fib_price = levels[name]
        if abs(price - fib_price) <= tol_abs:
            return name
    return "NONE"


def fib_zone_score(zone: str) -> float:
    """Điểm Fibonacci theo vùng (tối đa 20)."""
    return {
        "F618": 20.0,
        "F500": 15.0,
        "F382": 10.0,
        "F236": 5.0,
        "F786": 8.0,   # Retracement sâu — vẫn có giá trị nhưng rủi ro hơn
        "NONE": 0.0,
    }.get(zone, 0.0)


# ──────────────────────────────────────────────────────────────────
# Xác định vùng hỗ trợ / kháng cự (S/R clusters)
# ──────────────────────────────────────────────────────────────────

def find_sr_levels(swings: pd.DataFrame,
                   cluster_pct: float = 0.005) -> list[float]:
    """
    Gộp các đỉnh/đáy gần nhau thành vùng S/R.

    cluster_pct : nếu hai swing cách nhau < cluster_pct*price → cùng cluster.
    Returns danh sách mức giá S/R (trung bình của từng cluster).
    """
    prices = sorted(swings["price"].tolist())
    if not prices:
        return []

    clusters: list[list[float]] = [[prices[0]]]
    for p in prices[1:]:
        ref = clusters[-1][-1]
        if abs(p - ref) / ref <= cluster_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    return [round(float(np.mean(c)), 8) for c in clusters]


def at_sr_zone(price: float,
               sr_levels: list[float],
               tolerance: float = config.WAVE_FIB_TOLERANCE) -> bool:
    """Kiểm tra giá có đang ở vùng S/R hay không."""
    for level in sr_levels:
        if abs(price - level) / level <= tolerance:
            return True
    return False


# ──────────────────────────────────────────────────────────────────
# Phân tích xu hướng chính
# ──────────────────────────────────────────────────────────────────

def _main_direction(swings: pd.DataFrame) -> str:
    """
    Xác định xu hướng chính từ chuỗi đỉnh/đáy.

    Uptrend  : đỉnh sau cao hơn đỉnh trước VÀ đáy sau cao hơn đáy trước
    Downtrend: ngược lại
    """
    highs = swings[swings["type"] == "H"]["price"].values
    lows  = swings[swings["type"] == "L"]["price"].values

    if len(highs) < 2 or len(lows) < 2:
        return "RANGING"

    hh = highs[-1] > highs[-2]   # Higher High
    hl = lows[-1]  > lows[-2]    # Higher Low
    lh = highs[-1] < highs[-2]   # Lower High
    ll = lows[-1]  < lows[-2]    # Lower Low

    if hh and hl:
        return "UP"
    if lh and ll:
        return "DOWN"
    return "RANGING"


# ──────────────────────────────────────────────────────────────────
# Hàm chính: phân tích toàn bộ sóng
# ──────────────────────────────────────────────────────────────────

def analyze_waves(df: pd.DataFrame) -> WaveContext:
    """
    Phân tích sóng chính và sóng hồi từ DataFrame nến.

    Parameters
    ----------
    df : DataFrame với cột 'close' (và tùy chọn 'high', 'low')

    Returns
    -------
    WaveContext chứa đầy đủ thông tin sóng + khuyến nghị giao dịch
    """
    close  = df["close"]
    price  = float(close.iloc[-1])

    # ── 1. Phát hiện đỉnh/đáy ──────────────────────────────────────
    swings = detect_swings(close, order=config.WAVE_SWING_ORDER)

    _empty = WaveContext(
        main_direction="RANGING", main_wave_size=0, swing_high=price, swing_low=price,
        correction_active=False, correction_depth_pct=0, fib_zone="NONE",
        at_support_resistance=False, entry_direction="NONE", entry_score=0,
        tp_price=price, sl_price=price, description="Không đủ dữ liệu sóng",
    )
    if len(swings) < 4:
        return _empty

    # ── 2. Xu hướng chính ─────────────────────────────────────────
    main_dir  = _main_direction(swings)
    sr_levels = find_sr_levels(swings)

    last_swing  = swings.iloc[-1]
    # Tìm đỉnh và đáy gần nhất trong swings
    highs_df = swings[swings["type"] == "H"]
    lows_df  = swings[swings["type"] == "L"]

    if highs_df.empty or lows_df.empty:
        return _empty

    swing_high = float(highs_df.iloc[-1]["price"])
    swing_low  = float(lows_df.iloc[-1]["price"])

    main_wave_size = swing_high - swing_low
    min_wave_size  = price * config.WAVE_MIN_SIZE_PCT
    if main_wave_size < min_wave_size:
        return _empty

    # ── 3. Phát hiện sóng hồi ─────────────────────────────────────
    correction_active   = False
    correction_depth_pct = 0.0
    fib_levels: dict[str, float] = {}
    fib_zone = "NONE"
    entry_direction = "NONE"

    if main_dir == "UP":
        # Tìm điểm bắt đầu sóng hồi: đỉnh cuối cùng trong uptrend
        # Sóng hồi = giá đã kéo xuống từ đỉnh về phía đáy
        if price < swing_high and (swing_high - price) / main_wave_size > config.WAVE_CORRECTION_MIN:
            correction_depth = swing_high - price
            correction_depth_pct = correction_depth / main_wave_size * 100
            if correction_depth_pct <= config.WAVE_CORRECTION_MAX * 100:
                correction_active = True
                fib_levels = fibonacci_levels(swing_low, swing_high)
                fib_zone   = nearest_fib_zone(price, fib_levels)
                entry_direction = "CALL"   # Sóng hồi kết thúc → tiếp tục UP

    elif main_dir == "DOWN":
        # Sóng hồi = giá đã hồi lên từ đáy về phía đỉnh
        if price > swing_low and (price - swing_low) / main_wave_size > config.WAVE_CORRECTION_MIN:
            correction_depth = price - swing_low
            correction_depth_pct = correction_depth / main_wave_size * 100
            if correction_depth_pct <= config.WAVE_CORRECTION_MAX * 100:
                correction_active = True
                fib_levels = fibonacci_levels(swing_high, swing_low)
                fib_zone   = nearest_fib_zone(price, fib_levels)
                entry_direction = "PUT"    # Sóng hồi kết thúc → tiếp tục DOWN

    # ── 4. Kiểm tra S/R ───────────────────────────────────────────
    at_sr = at_sr_zone(price, sr_levels)

    # ── 5. Tính điểm sóng (0-40) ──────────────────────────────────
    entry_score = 0.0
    if correction_active:
        # a) Fibonacci zone score (0-20)
        entry_score += fib_zone_score(fib_zone)

        # b) Correction depth in ideal range 38-62% (0-10)
        ideal_low, ideal_high = 38.0, 62.0
        if ideal_low <= correction_depth_pct <= ideal_high:
            # Bão hoà tại 50% → 10 điểm
            dist_from_50 = abs(correction_depth_pct - 50.0)
            entry_score += max(0, 10 - dist_from_50 / 6)
        elif correction_depth_pct < ideal_low:
            entry_score += 5.0 * (correction_depth_pct / ideal_low)
        else:
            entry_score += 5.0 * ((100 - correction_depth_pct) / (100 - ideal_high))

        # c) S/R cluster confirmation (0-10)
        if at_sr:
            entry_score += 10.0

    entry_score = round(min(40.0, entry_score), 2)

    # ── 6. TP / SL ────────────────────────────────────────────────
    sl_buffer = main_wave_size * 0.05   # 5% của sóng chính làm buffer SL

    if main_dir == "UP" and correction_active:
        tp_price = round(swing_high, 8)                    # Đỉnh cũ
        sl_price = round(swing_low - sl_buffer, 8)        # Dưới đáy sóng chính
    elif main_dir == "DOWN" and correction_active:
        tp_price = round(swing_low, 8)                     # Đáy cũ
        sl_price = round(swing_high + sl_buffer, 8)       # Trên đỉnh sóng chính
    else:
        tp_price = price
        sl_price = price

    # ── 7. Mô tả ──────────────────────────────────────────────────
    if correction_active:
        desc = (
            f"Xu hướng {main_dir} | Sóng hồi {correction_depth_pct:.1f}% "
            f"({fib_zone}) | S/R={'✓' if at_sr else '✗'} | "
            f"Điểm sóng={entry_score}"
        )
    else:
        desc = f"Xu hướng {main_dir} | Không có sóng hồi | Giá={price:.6f}"

    return WaveContext(
        main_direction       = main_dir,
        main_wave_size       = round(main_wave_size, 8),
        swing_high           = swing_high,
        swing_low            = swing_low,
        correction_active    = correction_active,
        correction_depth_pct = round(correction_depth_pct, 2),
        fib_zone             = fib_zone,
        at_support_resistance= at_sr,
        entry_direction      = entry_direction if correction_active else "NONE",
        entry_score          = entry_score,
        tp_price             = tp_price,
        sl_price             = sl_price,
        description          = desc,
        sr_levels            = sr_levels,
    )


# ──────────────────────────────────────────────────────────────────
# Chạy trực tiếp để kiểm tra
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import deriv_data
    print(f"Đang lấy dữ liệu {config.SYMBOL}...")
    df  = deriv_data.fetch_candles()
    ctx = analyze_waves(df)
    print("\n─── Kết quả phân tích sóng ───")
    print(f"  Xu hướng chính : {ctx.main_direction}")
    print(f"  Kích thước sóng: {ctx.main_wave_size:.6f}")
    print(f"  Swing High     : {ctx.swing_high}")
    print(f"  Swing Low      : {ctx.swing_low}")
    print(f"  Sóng hồi       : {'CÓ' if ctx.correction_active else 'KHÔNG'}")
    if ctx.correction_active:
        print(f"  Độ sâu hồi     : {ctx.correction_depth_pct:.1f}%")
        print(f"  Vùng Fibonacci : {ctx.fib_zone}")
        print(f"  Tại S/R        : {ctx.at_support_resistance}")
        print(f"  Hướng vào lệnh : {ctx.entry_direction}")
        print(f"  Điểm sóng      : {ctx.entry_score}/40")
        print(f"  TP / SL        : {ctx.tp_price} / {ctx.sl_price}")
    print(f"\n  📝 {ctx.description}")

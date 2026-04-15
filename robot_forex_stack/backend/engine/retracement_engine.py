"""
Retracement Engine — Operator System tự vận hành.

Chức năng (5 trụ cột):
  1. **Tự phát hiện** sóng hồi nằm trong sóng chính khi giá gặp cản mạnh.
  2. **Tự đo lường** độ sâu theo tỷ lệ Fibonacci (23.6% → 78.6%).
  3. **Tự giới hạn** — Golden Zone 38.2%–61.8%, reject nếu nông/sâu quá.
  4. **Tự điều phối** — cung cấp context chất lượng cho AutoPilot scorer.
  5. **Tự tìm entry/SL/TP an toàn nhất** dựa trên hội tụ Fibonacci + S/R.

Quy trình nội bộ
----------------
  Main Wave BULL_MAIN → sub_wave SUB_WAVE_DOWN (hoặc price đang retrace)
    ├─ Xác định impulse leg (start_price → end_price của sóng chính)
    ├─ Tính % pullback hiện tại: retrace_pct = (end - now) / (end - start)
    ├─ Ánh xạ sang mức Fibonacci gần nhất
    ├─ Phát hiện S/R mạnh bằng cluster swing points (multi-touch)
    ├─ Kiểm tra nến đảo chiều (rejection candle) tại fib/S/R
    └─ Xuất RetracementMeasure: quality, safest_entry, safest_sl, safest_tp

S/R Detection
-------------
  – Cluster tất cả fractal swing highs/lows trong 150 bar gần nhất
  – Nhóm các level nằm trong ±0.3×ATR của nhau
  – Đếm số lần giá "chạm" level (came within 0.5×ATR)
  – Strength = min(touch_count / 3, 1.0) × recency_factor

Self-limit rules
----------------
  retrace_pct < 0.236 → NOT_RETRACING (vẫn đang impulse)
  retrace_pct > 0.786 → STRUCTURE_BROKEN (quá sâu, trend có thể đảo)
  0.236 ≤ pct < 0.382 → SHALLOW (valid nhưng low quality)
  0.382 ≤ pct ≤ 0.618 → GOLDEN_ZONE (tốt nhất)
  0.618 < pct ≤ 0.786 → DEEP (valid nhưng higher risk)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .wave_detector import WaveAnalysis, WaveState

logger = logging.getLogger(__name__)

# ── Fibonacci ratios ───────────────────────────────────────────────────── #

# String keys are kept intentionally for JSON serialisation compatibility
# (API returns {"0.382": 1.2345, ...}) — never use for numeric comparison.
FIB_RATIOS: Dict[str, float] = {
    "0.236": 0.236,
    "0.382": 0.382,
    "0.500": 0.500,
    "0.618": 0.618,
    "0.786": 0.786,
}

# Fibonacci extension levels (TP targets beyond impulse end)
FIB_EXTENSIONS: Dict[str, float] = {
    "1.000": 1.000,
    "1.272": 1.272,
    "1.618": 1.618,
}

# ── Enums ──────────────────────────────────────────────────────────────── #

# Fraction of total-bar-count subtracted from recency for the oldest bar.
# e.g. 0.5 means oldest bar = 50% strength reduction, newest bar = no reduction.
_RECENCY_DECAY_FACTOR = 0.5

# Minimum body size relative to ATR for a candle to qualify as a bounce signal.
_MIN_BOUNCE_BODY_ATR_RATIO = 0.25


class RetracementZone(str, Enum):
    NOT_RETRACING    = "NOT_RETRACING"    # pct < 23.6%
    SHALLOW          = "SHALLOW"          # 23.6% ≤ pct < 38.2%
    GOLDEN_ZONE      = "GOLDEN_ZONE"      # 38.2% ≤ pct ≤ 61.8%
    DEEP             = "DEEP"             # 61.8% < pct ≤ 78.6%
    STRUCTURE_BROKEN = "STRUCTURE_BROKEN" # pct > 78.6%


# ── Data classes ───────────────────────────────────────────────────────── #

@dataclass
class SupportResistanceLevel:
    """Mức hỗ trợ / kháng cự đã được xác nhận."""
    price:       float
    strength:    float        # 0.0 – 1.0
    sr_type:     str          # "SUPPORT" | "RESISTANCE" | "BOTH"
    touch_count: int
    origin_bar:  int          # bar index where level was first formed


@dataclass
class RetracementMeasure:
    """
    Kết quả đo lường đầy đủ của một sóng hồi.
    Được cập nhật mỗi tick bởi RetracementEngine.
    """
    in_retracement:   bool
    main_direction:   str              # "BUY" | "SELL"
    zone:             RetracementZone

    impulse_start:    float            # giá bắt đầu impulse leg
    impulse_end:      float            # đỉnh/đáy impulse (nơi sóng hồi bắt đầu)
    current_price:    float

    retrace_pct:      float            # 0.0 – 1.0 (e.g. 0.382 = 38.2%)
    nearest_fib:      str              # "0.382", "0.500", etc.
    fib_levels:       Dict[str, float] # {"0.382": 1.2345, ...}

    nearest_sr:       Optional[SupportResistanceLevel]
    sr_levels:        List[SupportResistanceLevel]

    quality:          float            # 0.0 – 1.0 tổng thể
    bounce_detected:  bool             # nến đảo chiều tại fib/S/R

    safest_entry:     float
    safest_sl:        float
    safest_tp:        float
    tp_extension:     float            # Fibonacci extension TP (127.2% or 161.8%)

    description:      str

    @property
    def risk_reward(self) -> float:
        if abs(self.safest_entry - self.safest_sl) < 1e-9:
            return 0.0
        return round(
            abs(self.safest_tp - self.safest_entry)
            / abs(self.safest_entry - self.safest_sl),
            2,
        )


# ── RetracementEngine ──────────────────────────────────────────────────── #

class RetracementEngine:
    """
    Operator System — Tự phát hiện, tự đo lường, tự giới hạn, tự điều phối
    điểm vào/thoát an toàn nhất.

    Parameters
    ----------
    fractal_period : int   Fractal lookback để phát hiện swing points
    sr_cluster_atr : float S/R cluster radius = sr_cluster_atr × ATR
    sr_touch_atr   : float "Touch" radius = sr_touch_atr × ATR
    min_quality    : float Ngưỡng quality tối thiểu để coi là valid setup
    sl_buffer_atr  : float SL được đặt cách fib level = sl_buffer_atr × ATR
    """

    def __init__(
        self,
        fractal_period: int = 2,
        sr_cluster_atr: float = 0.3,
        sr_touch_atr: float = 0.5,
        min_quality: float = 0.35,
        sl_buffer_atr: float = 0.8,
    ) -> None:
        self.fractal_period = fractal_period
        self.sr_cluster_atr = sr_cluster_atr
        self.sr_touch_atr = sr_touch_atr
        self.min_quality = min_quality
        self.sl_buffer_atr = sl_buffer_atr

        self._last_measure: Optional[RetracementMeasure] = None

    # ── Public API ─────────────────────────────────────────────────────── #

    def measure(
        self,
        df: pd.DataFrame,
        wave_analysis: WaveAnalysis,
        atr: float,
        pip_size: float = 0.0001,
    ) -> RetracementMeasure:
        """
        Phân tích đầy đủ sóng hồi trên df hiện tại.
        Trả về RetracementMeasure với tất cả thông tin cần thiết.
        """
        if len(df) < 20 or atr <= 0:
            return self._empty_measure(wave_analysis)

        main_wave = wave_analysis.main_wave
        if main_wave == WaveState.SIDEWAYS:
            return self._empty_measure(wave_analysis)

        current_price = float(df["close"].iloc[-1])
        is_bull = main_wave == WaveState.BULL_MAIN
        main_dir = "BUY" if is_bull else "SELL"

        # ── Bước 1: Tìm impulse leg ─────────────────────────────────── #
        impulse_start, impulse_end = self._find_impulse_leg(
            wave_analysis.swing_highs,
            wave_analysis.swing_lows,
            is_bull,
        )
        if impulse_start <= 0 or impulse_end <= 0:
            return self._empty_measure(wave_analysis)

        # ── Bước 2: Đo % pullback ───────────────────────────────────── #
        leg_size = abs(impulse_end - impulse_start)
        if leg_size < pip_size * 5:
            return self._empty_measure(wave_analysis)

        if is_bull:
            retrace_pct = (impulse_end - current_price) / leg_size
        else:
            retrace_pct = (current_price - impulse_end) / leg_size

        retrace_pct = round(retrace_pct, 4)

        # ── Bước 3: Xác định zone ───────────────────────────────────── #
        zone = self._classify_zone(retrace_pct)

        # ── Bước 4: Fibonacci levels ────────────────────────────────── #
        fib_levels = self._calc_fib_levels(impulse_start, impulse_end, is_bull)
        nearest_fib = self._nearest_fib(current_price, fib_levels)

        # ── Bước 5: S/R levels ──────────────────────────────────────── #
        sr_levels = self.detect_sr_levels(df, atr)
        nearest_sr = self._nearest_sr(current_price, sr_levels, atr)

        # ── Bước 6: Bounce candle detection ─────────────────────────── #
        bounce = self._detect_bounce(df, is_bull, atr)

        # ── Bước 7: Quality scoring (tự đo lường + tự giới hạn) ──── #
        in_retrace = zone not in (
            RetracementZone.NOT_RETRACING, RetracementZone.STRUCTURE_BROKEN
        )
        quality = self._calc_quality(
            retrace_pct, zone, nearest_sr, bounce,
            wave_analysis.sub_wave is not None,
        )

        # ── Bước 8: Safest entry / SL / TP ─────────────────────────── #
        safest_entry, safest_sl, safest_tp, tp_ext = self._find_safest_entry(
            current_price, impulse_start, impulse_end,
            fib_levels, nearest_fib, atr, is_bull, pip_size,
        )

        description = self._build_description(
            zone, retrace_pct, nearest_fib, quality, bounce, nearest_sr
        )

        measure = RetracementMeasure(
            in_retracement=in_retrace and quality >= self.min_quality,
            main_direction=main_dir,
            zone=zone,
            impulse_start=impulse_start,
            impulse_end=impulse_end,
            current_price=current_price,
            retrace_pct=retrace_pct,
            nearest_fib=nearest_fib,
            fib_levels=fib_levels,
            nearest_sr=nearest_sr,
            sr_levels=sr_levels[:5],
            quality=quality,
            bounce_detected=bounce,
            safest_entry=safest_entry,
            safest_sl=safest_sl,
            safest_tp=safest_tp,
            tp_extension=tp_ext,
            description=description,
        )
        self._last_measure = measure
        return measure

    def detect_sr_levels(
        self, df: pd.DataFrame, atr: float, lookback: int = 150
    ) -> List[SupportResistanceLevel]:
        """
        Phát hiện các mức S/R mạnh bằng cách cluster swing points.
        Levels được sắp xếp theo strength giảm dần.
        """
        if len(df) < 10 or atr <= 0:
            return []

        data = df.iloc[-min(len(df), lookback) :].reset_index(drop=True)
        fp = self.fractal_period
        n = len(data)

        raw_levels: List[Tuple[float, str, int]] = []  # (price, type, bar_idx)
        for i in range(fp, n - fp):
            win_h = data["high"].iloc[i - fp : i + fp + 1]
            if data["high"].iloc[i] == win_h.max():
                raw_levels.append((float(data["high"].iloc[i]), "RESISTANCE", i))
            win_l = data["low"].iloc[i - fp : i + fp + 1]
            if data["low"].iloc[i] == win_l.min():
                raw_levels.append((float(data["low"].iloc[i]), "SUPPORT", i))

        if not raw_levels:
            return []

        # Cluster nearby levels
        clusters = self._cluster_levels(raw_levels, atr)

        # Count touches for each cluster
        result: List[SupportResistanceLevel] = []
        close_arr = data["close"].values
        high_arr  = data["high"].values
        low_arr   = data["low"].values
        touch_radius = self.sr_touch_atr * atr

        for cluster_price, cluster_type, origin_bar in clusters:
            touches = 0
            for j in range(n):
                if abs(high_arr[j] - cluster_price) <= touch_radius or \
                   abs(low_arr[j]  - cluster_price) <= touch_radius or \
                   abs(close_arr[j] - cluster_price) <= touch_radius:
                    touches += 1

            # Recency factor (more recent = stronger)
            recency = 1.0 - (n - 1 - origin_bar) / max(n, 1) * _RECENCY_DECAY_FACTOR
            strength = round(min(touches / 4.0, 1.0) * recency, 3)

            result.append(SupportResistanceLevel(
                price=cluster_price,
                strength=strength,
                sr_type=cluster_type,
                touch_count=touches,
                origin_bar=origin_bar,
            ))

        result.sort(key=lambda x: x.strength, reverse=True)
        return result

    @property
    def last_measure(self) -> Optional[RetracementMeasure]:
        return self._last_measure

    # ── Internal helpers ───────────────────────────────────────────────── #

    @staticmethod
    def _find_impulse_leg(
        swing_highs: list,
        swing_lows: list,
        is_bull: bool,
    ) -> Tuple[float, float]:
        """
        Tìm chân impulse gần nhất:
          Bull: last_swing_low → last_swing_high
          Bear: last_swing_high → last_swing_low
        """
        if is_bull:
            if len(swing_highs) < 1 or len(swing_lows) < 1:
                return 0.0, 0.0
            end   = swing_highs[-1].price   # đỉnh impulse
            # Tìm low trước đỉnh đó
            prev_lows = [s for s in swing_lows if s.index < swing_highs[-1].index]
            if not prev_lows:
                return 0.0, 0.0
            start = prev_lows[-1].price     # đáy trước đỉnh
            if end <= start:
                return 0.0, 0.0
            return start, end
        else:
            if len(swing_lows) < 1 or len(swing_highs) < 1:
                return 0.0, 0.0
            end   = swing_lows[-1].price    # đáy impulse
            prev_highs = [s for s in swing_highs if s.index < swing_lows[-1].index]
            if not prev_highs:
                return 0.0, 0.0
            start = prev_highs[-1].price    # đỉnh trước đáy
            if end >= start:
                return 0.0, 0.0
            return start, end

    @staticmethod
    def _calc_fib_levels(
        impulse_start: float, impulse_end: float, is_bull: bool
    ) -> Dict[str, float]:
        """
        Tính giá trị price tại từng mức Fibonacci.
        Bull: levels đi xuống từ impulse_end (đỉnh)
        Bear: levels đi lên từ impulse_end (đáy)
        """
        leg = abs(impulse_end - impulse_start)
        levels: Dict[str, float] = {}
        for label, ratio in FIB_RATIOS.items():
            if is_bull:
                levels[label] = round(impulse_end - leg * ratio, 5)
            else:
                levels[label] = round(impulse_end + leg * ratio, 5)
        return levels

    @staticmethod
    def _nearest_fib(price: float, fib_levels: Dict[str, float]) -> str:
        """Trả về label của mức Fibonacci gần current_price nhất."""
        if not fib_levels:
            return "0.500"
        return min(fib_levels, key=lambda k: abs(fib_levels[k] - price))

    @staticmethod
    def _nearest_sr(
        price: float,
        sr_levels: List[SupportResistanceLevel],
        atr: float,
    ) -> Optional[SupportResistanceLevel]:
        """Tìm S/R level gần current_price nhất, trong vòng 2×ATR."""
        if not sr_levels:
            return None
        best = min(sr_levels, key=lambda s: abs(s.price - price))
        if abs(best.price - price) <= 2.0 * atr:
            return best
        return None

    @staticmethod
    def _detect_bounce(
        df: pd.DataFrame, is_bull: bool, atr: float
    ) -> bool:
        """
        Phát hiện nến đảo chiều (rejection candle) để xác nhận
        rằng sóng hồi đã kết thúc và giá đang quay về hướng chính.

        Bull retrace → cần bullish rejection:
          – Nến close > open (xanh)
          – Lower wick >= body (giá đã bị đẩy lên từ phía dưới)
          – Body >= 0.25 × ATR (đủ mạnh)

        Bear retrace → cần bearish rejection:
          – Nến close < open (đỏ)
          – Upper wick >= body
          – Body >= 0.25 × ATR
        """
        if len(df) < 2:
            return False
        c = df.iloc[-1]
        op, hi, lo, cl = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"])
        body = abs(cl - op)

        if body < _MIN_BOUNCE_BODY_ATR_RATIO * atr:
            return False

        if is_bull:
            lower_wick = min(op, cl) - lo
            return cl > op and lower_wick >= body * 0.8
        else:
            upper_wick = hi - max(op, cl)
            return cl < op and upper_wick >= body * 0.8

    @staticmethod
    def _classify_zone(retrace_pct: float) -> RetracementZone:
        if retrace_pct < 0.236:
            return RetracementZone.NOT_RETRACING
        if retrace_pct > 0.786:
            return RetracementZone.STRUCTURE_BROKEN
        if retrace_pct <= 0.382:
            return RetracementZone.SHALLOW
        if retrace_pct <= 0.618:
            return RetracementZone.GOLDEN_ZONE
        return RetracementZone.DEEP

    def _calc_quality(
        self,
        retrace_pct: float,
        zone: RetracementZone,
        nearest_sr: Optional[SupportResistanceLevel],
        bounce: bool,
        sub_wave_active: bool,
    ) -> float:
        """
        Tự đo lường + tự giới hạn:
          – Zone score:        0.0 / 0.35 / 0.55 / 0.40
          – S/R confluence:    +0.25 × sr_strength
          – Bounce candle:     +0.20
          – Sub-wave confirmed:+0.10
        Quality < min_quality → RetracementMeasure.in_retracement = False
        """
        if zone in (RetracementZone.NOT_RETRACING, RetracementZone.STRUCTURE_BROKEN):
            return 0.0

        zone_score = {
            RetracementZone.SHALLOW:     0.35,
            RetracementZone.GOLDEN_ZONE: 0.55,
            RetracementZone.DEEP:        0.40,
        }.get(zone, 0.0)

        sr_score  = (nearest_sr.strength * 0.25) if nearest_sr else 0.0
        bounce_sc = 0.20 if bounce else 0.0
        sub_sc    = 0.10 if sub_wave_active else 0.0

        # Small penalty for being too close to boundary of zone
        boundary_penalty = 0.0
        if zone == RetracementZone.GOLDEN_ZONE:
            dist_from_edge = min(
                abs(retrace_pct - 0.382),
                abs(retrace_pct - 0.618),
            )
            if dist_from_edge < 0.02:
                boundary_penalty = -0.05

        quality = zone_score + sr_score + bounce_sc + sub_sc + boundary_penalty
        return round(min(max(quality, 0.0), 1.0), 4)

    def _find_safest_entry(
        self,
        current_price: float,
        impulse_start: float,
        impulse_end: float,
        fib_levels: Dict[str, float],
        nearest_fib: str,
        atr: float,
        is_bull: bool,
        pip_size: float,
    ) -> Tuple[float, float, float, float]:
        """
        Tính điểm vào lệnh an toàn nhất:
          Entry  = current_price (đặt ngay khi bounce xác nhận)
          SL     = just beyond fib_0.786 level (+ sl_buffer×ATR)
          TP     = impulse_end (target = đỉnh/đáy sóng chính)
          TP_ext = Fibonacci extension 127.2% (nếu momentum mạnh)
        """
        buf = self.sl_buffer_atr * atr
        leg = abs(impulse_end - impulse_start)

        if is_bull:
            # SL: below fib 0.786 level
            sl_level = fib_levels.get("0.786", impulse_start)
            safest_sl = round(sl_level - buf, 5)
            safest_sl = max(safest_sl, impulse_start - buf)

            # TP: impulse_end (previous swing high)
            safest_tp = round(impulse_end, 5)

            # Extension TP
            tp_ext = round(impulse_end + leg * (FIB_EXTENSIONS["1.272"] - 1.0), 5)
        else:
            # SL: above fib 0.786 level
            sl_level = fib_levels.get("0.786", impulse_start)
            safest_sl = round(sl_level + buf, 5)
            safest_sl = min(safest_sl, impulse_start + buf)

            # TP: impulse_end (previous swing low)
            safest_tp = round(impulse_end, 5)

            # Extension TP
            tp_ext = round(impulse_end - leg * (FIB_EXTENSIONS["1.272"] - 1.0), 5)

        return current_price, safest_sl, safest_tp, tp_ext

    def _cluster_levels(
        self,
        raw: List[Tuple[float, str, int]],
        atr: float,
    ) -> List[Tuple[float, str, int]]:
        """
        Gộp các level gần nhau (< sr_cluster_atr × ATR) thành một cluster.
        Trả về list (avg_price, dominant_type, most_recent_bar).
        """
        if not raw:
            return []
        radius = self.sr_cluster_atr * atr
        sorted_raw = sorted(raw, key=lambda x: x[0])
        clusters: List[List[Tuple[float, str, int]]] = []
        current: List[Tuple[float, str, int]] = [sorted_raw[0]]

        for item in sorted_raw[1:]:
            if item[0] - current[-1][0] <= radius:
                current.append(item)
            else:
                clusters.append(current)
                current = [item]
        clusters.append(current)

        result = []
        for cl in clusters:
            avg_price = float(np.mean([x[0] for x in cl]))
            types = [x[1] for x in cl]
            dom_type = "BOTH" if len(set(types)) > 1 else types[0]
            latest_bar = max(x[2] for x in cl)
            result.append((round(avg_price, 5), dom_type, latest_bar))
        return result

    @staticmethod
    def _build_description(
        zone: RetracementZone,
        retrace_pct: float,
        nearest_fib: str,
        quality: float,
        bounce: bool,
        nearest_sr: Optional[SupportResistanceLevel],
    ) -> str:
        parts = [
            f"Zone: {zone.value}",
            f"Retrace: {retrace_pct:.1%}",
            f"Near fib: {nearest_fib}",
            f"Quality: {quality:.0%}",
        ]
        if bounce:
            parts.append("✓ Bounce")
        if nearest_sr:
            parts.append(f"S/R: {nearest_sr.price:.5f} (str={nearest_sr.strength:.2f})")
        return " | ".join(parts)

    @staticmethod
    def _empty_measure(wave_analysis: WaveAnalysis) -> RetracementMeasure:
        is_bull = wave_analysis.main_wave == WaveState.BULL_MAIN
        return RetracementMeasure(
            in_retracement=False,
            main_direction="BUY" if is_bull else "SELL",
            zone=RetracementZone.NOT_RETRACING,
            impulse_start=0.0,
            impulse_end=0.0,
            current_price=0.0,
            retrace_pct=0.0,
            nearest_fib="0.500",
            fib_levels={},
            nearest_sr=None,
            sr_levels=[],
            quality=0.0,
            bounce_detected=False,
            safest_entry=0.0,
            safest_sl=0.0,
            safest_tp=0.0,
            tp_extension=0.0,
            description="No data / Sideways",
        )

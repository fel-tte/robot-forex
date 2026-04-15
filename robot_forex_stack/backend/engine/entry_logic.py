"""
Entry Logic — Opening Range, SL/TP calculation, Entry mode checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class EntryMode(str, Enum):
    BREAKOUT = "BREAKOUT"
    INSTANT_BREAKOUT = "INSTANT_BREAKOUT"
    RETRACE = "RETRACE"
    INSTANT_RETRACE = "INSTANT_RETRACE"
    RETEST_SAME = "RETEST_SAME"
    RETEST_OPPOSITE = "RETEST_OPPOSITE"
    RETEST_LEVEL_X = "RETEST_LEVEL_X"


class SLMode(str, Enum):
    POINTS = "POINTS"
    ATR = "ATR"
    RANGE_SIZE = "RANGE_SIZE"
    PREV_CANDLE_POINTS = "PREV_CANDLE_POINTS"
    PREV_CANDLE_ATR = "PREV_CANDLE_ATR"
    LAST_SWING_POINTS = "LAST_SWING_POINTS"
    LAST_SWING_ATR = "LAST_SWING_ATR"
    RANGE_OPPOSITE_POINTS = "RANGE_OPPOSITE_POINTS"
    RANGE_OPPOSITE_ATR = "RANGE_OPPOSITE_ATR"


class TPMode(str, Enum):
    SL_RATIO = "SL_RATIO"
    ATR = "ATR"
    POINTS = "POINTS"


@dataclass
class EntrySignal:
    signal_id: str
    symbol: str
    direction: str        # BUY / SELL
    entry_price: float
    sl: float
    tp: float
    lot_size: float
    entry_mode: str
    sl_distance: float    # in price units
    tp_distance: float
    atr: float
    timestamp: float = field(default_factory=lambda: __import__("time").time())
    meta: Dict = field(default_factory=dict)

    @property
    def risk_reward(self) -> float:
        if self.sl_distance <= 0:
            return 0.0
        return round(self.tp_distance / self.sl_distance, 2)


@dataclass
class OpeningRange:
    high: float = 0.0
    low: float = 0.0
    mid: float = 0.0
    size: float = 0.0
    formed: bool = False

    def update(self, high: float, low: float) -> None:
        self.high = max(self.high, high)
        self.low = min(self.low if self.low > 0 else low, low)
        self.mid = (self.high + self.low) / 2
        self.size = self.high - self.low
        self.formed = self.size > 0


class RangeManager:
    """Tracks the Opening Range formed during a monitoring period."""

    def __init__(self) -> None:
        self._range = OpeningRange()
        self._candle_count = 0
        self._monitoring_limit = 12   # default: 12 candles = 1h on M5

    def set_monitoring_candles(self, n: int) -> None:
        self._monitoring_limit = max(1, n)

    def add_candle(self, high: float, low: float) -> bool:
        """Returns True when range is fully formed."""
        if self._candle_count < self._monitoring_limit:
            self._range.update(high, low)
            self._candle_count += 1
        return self._range.formed and self._candle_count >= self._monitoring_limit

    def reset(self) -> None:
        self._range = OpeningRange()
        self._candle_count = 0

    @property
    def range(self) -> OpeningRange:
        return self._range

    @property
    def is_complete(self) -> bool:
        return self._range.formed and self._candle_count >= self._monitoring_limit


class EntryLogic:
    """
    Calculates SL/TP and checks entry conditions for all supported modes.
    """

    def __init__(
        self,
        sl_mode: SLMode = SLMode.POINTS,
        sl_value: float = 200,
        tp_mode: TPMode = TPMode.SL_RATIO,
        tp_value: float = 2.0,
        entry_mode: EntryMode = EntryMode.BREAKOUT,
        retrace_atr_mult: float = 0.5,
        min_body_atr: float = 0.3,
        retest_level_x: float = 0.5,     # 0.5 = 50% of range
    ) -> None:
        self.sl_mode = sl_mode
        self.sl_value = sl_value
        self.tp_mode = tp_mode
        self.tp_value = tp_value
        self.entry_mode = entry_mode
        self.retrace_atr_mult = retrace_atr_mult
        self.min_body_atr = min_body_atr
        self.retest_level_x = retest_level_x

    # ------------------------------------------------------------------ #
    #  SL / TP Calculation                                                 #
    # ------------------------------------------------------------------ #

    def calculate_sl(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        swing_high: float = 0.0,
        swing_low: float = 0.0,
        range_high: float = 0.0,
        range_low: float = 0.0,
        prev_high: float = 0.0,
        prev_low: float = 0.0,
        pip_size: float = 0.0001,
    ) -> Tuple[float, float]:
        """Returns (sl_price, sl_distance)."""
        is_buy = direction.upper() in ("BUY", "LONG")

        if self.sl_mode == SLMode.POINTS:
            dist = self.sl_value * pip_size
        elif self.sl_mode == SLMode.ATR:
            dist = self.sl_value * atr
        elif self.sl_mode == SLMode.RANGE_SIZE:
            dist = (range_high - range_low) if range_high > range_low else atr
        elif self.sl_mode == SLMode.PREV_CANDLE_POINTS:
            base = prev_low if is_buy else prev_high
            dist = abs(entry_price - base) + self.sl_value * pip_size
        elif self.sl_mode == SLMode.PREV_CANDLE_ATR:
            base = prev_low if is_buy else prev_high
            dist = abs(entry_price - base) + self.sl_value * atr
        elif self.sl_mode == SLMode.LAST_SWING_POINTS:
            base = swing_low if is_buy else swing_high
            dist = abs(entry_price - base) + self.sl_value * pip_size if base > 0 else self.sl_value * pip_size
        elif self.sl_mode == SLMode.LAST_SWING_ATR:
            base = swing_low if is_buy else swing_high
            dist = abs(entry_price - base) + self.sl_value * atr if base > 0 else self.sl_value * atr
        elif self.sl_mode == SLMode.RANGE_OPPOSITE_POINTS:
            base = range_low if is_buy else range_high
            dist = abs(entry_price - base) + self.sl_value * pip_size if base > 0 else self.sl_value * pip_size
        elif self.sl_mode == SLMode.RANGE_OPPOSITE_ATR:
            base = range_low if is_buy else range_high
            dist = abs(entry_price - base) + self.sl_value * atr if base > 0 else self.sl_value * atr
        else:
            dist = self.sl_value * pip_size

        dist = max(dist, pip_size * 5)   # minimum SL = 5 pips
        sl_price = entry_price - dist if is_buy else entry_price + dist
        return round(sl_price, 5), round(dist, 5)

    def calculate_tp(
        self,
        direction: str,
        entry_price: float,
        sl_distance: float,
        atr: float,
        pip_size: float = 0.0001,
    ) -> Tuple[float, float]:
        """Returns (tp_price, tp_distance)."""
        is_buy = direction.upper() in ("BUY", "LONG")

        if self.tp_mode == TPMode.SL_RATIO:
            dist = sl_distance * self.tp_value
        elif self.tp_mode == TPMode.ATR:
            dist = self.tp_value * atr
        elif self.tp_mode == TPMode.POINTS:
            dist = self.tp_value * pip_size
        else:
            dist = sl_distance * 2.0

        dist = max(dist, pip_size * 5)
        tp_price = entry_price + dist if is_buy else entry_price - dist
        return round(tp_price, 5), round(dist, 5)

    # ------------------------------------------------------------------ #
    #  Entry condition checks                                              #
    # ------------------------------------------------------------------ #

    def check_entry(
        self,
        candle: pd.Series,
        range_high: float,
        range_low: float,
        atr: float,
        prev_close: float = 0.0,
    ) -> Optional[str]:
        """
        Returns 'BUY', 'SELL', or None depending on entry mode.
        candle: Series with fields open, high, low, close
        """
        close = float(candle["close"])
        open_ = float(candle["open"])
        high = float(candle["high"])
        low = float(candle["low"])
        body = abs(close - open_)

        if self.entry_mode == EntryMode.BREAKOUT:
            return self._check_breakout(close, range_high, range_low, body, atr)

        elif self.entry_mode == EntryMode.INSTANT_BREAKOUT:
            # Trigger as soon as price touches the range boundary (no close needed)
            if high > range_high:
                return "BUY"
            if low < range_low:
                return "SELL"

        elif self.entry_mode == EntryMode.RETRACE:
            return self._check_retrace(close, open_, range_high, range_low, atr)

        elif self.entry_mode == EntryMode.INSTANT_RETRACE:
            # Retrace to EMA / mid-range, no close confirmation
            mid = (range_high + range_low) / 2
            if prev_close > range_high and low <= mid + atr * self.retrace_atr_mult:
                return "BUY"
            if prev_close < range_low and high >= mid - atr * self.retrace_atr_mult:
                return "SELL"

        elif self.entry_mode == EntryMode.RETEST_SAME:
            return self._check_retest_same(close, high, low, range_high, range_low)

        elif self.entry_mode == EntryMode.RETEST_OPPOSITE:
            return self._check_retest_opposite(close, high, low, range_high, range_low)

        elif self.entry_mode == EntryMode.RETEST_LEVEL_X:
            level = range_low + (range_high - range_low) * self.retest_level_x
            if low <= level and close > level:
                return "BUY"
            if high >= level and close < level:
                return "SELL"

        return None

    def _check_breakout(
        self,
        close: float,
        range_high: float,
        range_low: float,
        body: float,
        atr: float,
    ) -> Optional[str]:
        if range_high <= 0 or range_low <= 0:
            return None
        min_body = self.min_body_atr * atr
        if close > range_high and body >= min_body:
            return "BUY"
        if close < range_low and body >= min_body:
            return "SELL"
        return None

    def _check_retrace(
        self,
        close: float,
        open_: float,
        range_high: float,
        range_low: float,
        atr: float,
    ) -> Optional[str]:
        retrace_dist = self.retrace_atr_mult * atr
        # Price broke above range, then pulled back into range
        if close > range_high - retrace_dist and open_ > range_high:
            return "BUY"
        if close < range_low + retrace_dist and open_ < range_low:
            return "SELL"
        return None

    @staticmethod
    def _check_retest_same(
        close: float,
        high: float,
        low: float,
        range_high: float,
        range_low: float,
    ) -> Optional[str]:
        # Price retests the same side of the range after breakout
        if low <= range_high and close > range_high:
            return "BUY"
        if high >= range_low and close < range_low:
            return "SELL"
        return None

    @staticmethod
    def _check_retest_opposite(
        close: float,
        high: float,
        low: float,
        range_high: float,
        range_low: float,
    ) -> Optional[str]:
        # Price retests the opposite boundary
        mid = (range_high + range_low) / 2
        if close > mid and low <= (range_high + range_low) / 2:
            return "BUY"
        if close < mid and high >= (range_high + range_low) / 2:
            return "SELL"
        return None

    def build_entry_signal(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        lot_size: float,
        atr: float,
        swing_high: float = 0.0,
        swing_low: float = 0.0,
        range_high: float = 0.0,
        range_low: float = 0.0,
        prev_high: float = 0.0,
        prev_low: float = 0.0,
        pip_size: float = 0.0001,
    ) -> EntrySignal:
        sl_price, sl_dist = self.calculate_sl(
            direction, entry_price, atr,
            swing_high, swing_low, range_high, range_low, prev_high, prev_low, pip_size
        )
        tp_price, tp_dist = self.calculate_tp(direction, entry_price, sl_dist, atr, pip_size)
        return EntrySignal(
            signal_id=signal_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            sl=sl_price,
            tp=tp_price,
            lot_size=lot_size,
            entry_mode=self.entry_mode.value,
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            atr=atr,
        )

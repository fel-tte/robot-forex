"""
Mock Data Provider — Synthetic OHLCV generation + indicator calculations.

Produces realistic Forex-like price series with:
  - Identifiable bull / bear trends
  - Sideways consolidation periods
  - Corrections (sub-waves) within trends
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OHLCVBar:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


class MockDataProvider:
    """
    Generates synthetic OHLCV data simulating Forex price action.

    The price series alternates between:
      - Strong trend (50 candles)
      - Correction / sub-wave (20 candles)
      - Sideways consolidation (30 candles)
    Cycle repeats indefinitely.
    """

    PHASE_TREND = "TREND"
    PHASE_CORRECTION = "CORRECTION"
    PHASE_SIDEWAYS = "SIDEWAYS"

    _PHASES = [
        (PHASE_TREND, 50),
        (PHASE_CORRECTION, 20),
        (PHASE_TREND, 50),
        (PHASE_SIDEWAYS, 30),
    ]

    def __init__(
        self,
        symbol: str = "EURUSD",
        start_price: float = 1.10000,
        pip_size: float = 0.0001,
        spread_pips: float = 1.5,
        noise_factor: float = 0.3,
        seed: Optional[int] = 42,
        candle_seconds: int = 300,   # M5
    ) -> None:
        self.symbol = symbol
        self.pip_size = pip_size
        self.spread_pips = spread_pips
        self.noise_factor = noise_factor
        self.candle_seconds = candle_seconds

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._price = start_price
        self._bars: List[OHLCVBar] = []
        self._ts = time.time() - 500 * candle_seconds  # pre-history starts 500 bars ago
        self._phase_idx = 0
        self._phase_candle = 0
        self._trend_direction = 1   # +1 bull, -1 bear
        self._atr_base = pip_size * 120  # ~12 pips ATR

        # Pre-generate 500 historical candles
        self._generate_bulk(500)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_candles(
        self, limit: int = 200, timeframe: str = "M5"
    ) -> pd.DataFrame:
        """Returns DataFrame of most recent `limit` candles."""
        self._generate_new_candles()
        bars = self._bars[-limit:]
        df = pd.DataFrame(
            [
                {
                    "timestamp": b.timestamp,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in bars
            ]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    def get_current_price(self) -> Tuple[float, float]:
        """Returns (bid, ask)."""
        if self._bars:
            mid = self._bars[-1].close
        else:
            mid = self._price
        half_spread = self.spread_pips * self.pip_size / 2
        return round(mid - half_spread, 5), round(mid + half_spread, 5)

    def get_spread_points(self) -> float:
        return self.spread_pips

    def advance(self) -> OHLCVBar:
        """Generate one new candle and return it."""
        bar = self._generate_one_candle()
        return bar

    # ------------------------------------------------------------------ #
    #  Indicator calculations (static / class-level)                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr_series = tr.rolling(period).mean()
        val = atr_series.iloc[-1]
        return float(val) if not math.isnan(val) else 0.0

    @staticmethod
    def calculate_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
        return df[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_fractals(
        df: pd.DataFrame, period: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Returns (fractal_highs, fractal_lows) — price at fractal bars, NaN elsewhere.
        """
        n = len(df)
        frac_highs = pd.Series(np.nan, index=df.index)
        frac_lows = pd.Series(np.nan, index=df.index)

        for i in range(period, n - period):
            window_h = df["high"].iloc[i - period : i + period + 1]
            if df["high"].iloc[i] == window_h.max():
                frac_highs.iloc[i] = df["high"].iloc[i]
            window_l = df["low"].iloc[i - period : i + period + 1]
            if df["low"].iloc[i] == window_l.min():
                frac_lows.iloc[i] = df["low"].iloc[i]

        return frac_highs, frac_lows

    @staticmethod
    def calculate_sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
        return df[column].rolling(period).mean()

    # ------------------------------------------------------------------ #
    #  Candle generation                                                   #
    # ------------------------------------------------------------------ #

    def _generate_bulk(self, n: int) -> None:
        for _ in range(n):
            self._generate_one_candle()

    def _generate_new_candles(self) -> None:
        """Generate any candles that should have formed since last call."""
        now = time.time()
        while self._ts + self.candle_seconds <= now:
            self._generate_one_candle()

    def _generate_one_candle(self) -> OHLCVBar:
        phase_name, phase_len = self._PHASES[self._phase_idx]

        # Trend parameters per phase
        if phase_name == self.PHASE_TREND:
            bias = self._trend_direction * self._atr_base * 0.4
            volatility = self._atr_base
        elif phase_name == self.PHASE_CORRECTION:
            bias = -self._trend_direction * self._atr_base * 0.25
            volatility = self._atr_base * 0.8
        else:  # SIDEWAYS
            bias = 0.0
            volatility = self._atr_base * 0.4

        noise = np.random.normal(0, volatility * self.noise_factor)
        move = bias + noise

        open_ = self._price
        close = open_ + move

        # Wicks
        candle_range = abs(move) + np.random.exponential(volatility * 0.3)
        body_hi = max(open_, close)
        body_lo = min(open_, close)
        upper_wick = np.random.uniform(0, candle_range * 0.4)
        lower_wick = np.random.uniform(0, candle_range * 0.4)
        high = body_hi + upper_wick
        low = body_lo - lower_wick

        # Clamp price above zero
        low = max(low, self.pip_size * 10)
        close = max(close, self.pip_size * 10)
        high = max(high, close, open_)
        low = min(low, close, open_)

        volume = abs(np.random.normal(1000, 300))

        bar = OHLCVBar(
            timestamp=self._ts,
            open=round(open_, 5),
            high=round(high, 5),
            low=round(low, 5),
            close=round(close, 5),
            volume=round(volume, 0),
        )
        self._bars.append(bar)
        self._price = close

        # Advance phase
        self._phase_candle += 1
        if self._phase_candle >= phase_len:
            self._phase_candle = 0
            self._phase_idx = (self._phase_idx + 1) % len(self._PHASES)
            # Flip trend at end of correction cycle
            if phase_name == self.PHASE_CORRECTION and random.random() < 0.4:
                self._trend_direction *= -1

        self._ts += self.candle_seconds
        return bar

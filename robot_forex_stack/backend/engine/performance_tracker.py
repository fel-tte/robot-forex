"""
Performance Tracker — Statistical learning foundation.

Records per-(mode, wave_state) trade outcomes in a sliding window.
Provides: win_rate, profit_factor, expectancy, avg_rr per segment.
AdaptiveController reads these stats to adjust scoring weights.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sliding window size per segment
_SEGMENT_WINDOW = 30


# ── Data classes ───────────────────────────────────────────────────────── #

@dataclass
class TradeOutcome:
    """A single completed trade outcome for learning."""
    mode:         str    # entry mode (BREAKOUT, RETRACEMENT, etc.)
    wave_state:   str    # BULL_MAIN | BEAR_MAIN | SIDEWAYS
    direction:    str    # BUY | SELL
    retrace_zone: str    # NOT_RETRACING | GOLDEN_ZONE | etc.
    pnl:          float  # raw PnL in account currency
    rr_achieved:  float  # actual R:R (pnl / initial_risk, signed)
    initial_risk: float  # initial risk in account currency (absolute)
    timestamp:    float  = field(default_factory=time.time)


@dataclass
class SegmentStats:
    """Statistics for a specific (mode, wave_state) segment."""
    win_rate:      float = 0.0
    profit_factor: float = 1.0
    avg_rr:        float = 0.0
    expectancy:    float = 0.0   # average PnL per trade
    sample_size:   int   = 0
    last_updated:  float = 0.0


# Segment key = (mode, wave_state)
_SegKey = Tuple[str, str]


# ── PerformanceTracker ─────────────────────────────────────────────────── #

class PerformanceTracker:
    """
    Sliding-window statistical tracker.

    Segments: (entry_mode, wave_state) pairs.
    Also maintains a global rolling window for overall system health.
    """

    def __init__(self, window: int = _SEGMENT_WINDOW) -> None:
        self.window = window
        self._segments: Dict[_SegKey, Deque[TradeOutcome]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._global: Deque[TradeOutcome] = deque(maxlen=200)
        self._total_recorded: int = 0

    def record(self, outcome: TradeOutcome) -> None:
        """Record a completed trade outcome."""
        key = (outcome.mode, outcome.wave_state)
        self._segments[key].append(outcome)
        self._global.append(outcome)
        self._total_recorded += 1
        logger.debug(
            "PerformanceTracker: %s/%s pnl=%.2f rr=%.2f (total=%d)",
            outcome.mode, outcome.wave_state,
            outcome.pnl, outcome.rr_achieved, self._total_recorded,
        )

    def get_stats(self, mode: str, wave_state: str) -> SegmentStats:
        """Statistics for a specific (mode, wave_state) segment."""
        key = (mode, wave_state)
        return self._compute_stats(list(self._segments.get(key, [])))

    def get_global_stats(self) -> SegmentStats:
        """Global stats across all modes and wave states."""
        return self._compute_stats(list(self._global))

    def get_consecutive_losses(self) -> int:
        """Count of consecutive losses in most recent global trades."""
        losses = 0
        for o in reversed(list(self._global)):
            if o.pnl < 0:
                losses += 1
            else:
                break
        return losses

    def get_all_segment_stats(self) -> Dict[str, SegmentStats]:
        """All segment stats keyed by 'mode/wave_state'."""
        result: Dict[str, SegmentStats] = {}
        for (mode, wave_state), outcomes in self._segments.items():
            result[f"{mode}/{wave_state}"] = self._compute_stats(list(outcomes))
        return result

    def get_recent_outcomes(self, n: int = 20) -> List[TradeOutcome]:
        """Most recent n outcomes (newest first)."""
        return list(reversed(list(self._global)))[:n]

    @property
    def total_recorded(self) -> int:
        return self._total_recorded

    # ── Internal helpers ───────────────────────────────────────────────── #

    @staticmethod
    def _compute_stats(outcomes: List[TradeOutcome]) -> SegmentStats:
        if not outcomes:
            return SegmentStats()
        wins   = [o.pnl for o in outcomes if o.pnl > 0]
        losses = [o.pnl for o in outcomes if o.pnl < 0]
        win_rate = len(wins) / len(outcomes)
        gross_profit = sum(wins) if wins else 0.0
        gross_loss   = abs(sum(losses)) if losses else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 10.0
        else:
            profit_factor = 1.0
        avg_rr     = float(sum(o.rr_achieved for o in outcomes) / len(outcomes))
        expectancy = float(sum(o.pnl for o in outcomes) / len(outcomes))
        return SegmentStats(
            win_rate=round(win_rate, 4),
            profit_factor=round(min(profit_factor, 10.0), 3),
            avg_rr=round(avg_rr, 3),
            expectancy=round(expectancy, 4),
            sample_size=len(outcomes),
            last_updated=time.time(),
        )

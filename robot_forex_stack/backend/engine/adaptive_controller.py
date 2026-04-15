"""
Adaptive Controller — Self-learning parameter adjustment.

Reads PerformanceTracker and adjusts:
  1. mode_weight_adj   : per-(mode, wave_state) multiplier delta (−0.5 → +0.5)
  2. min_score_adj     : added to base _MIN_SCORE (−0.10 → +0.25)
  3. lot_scale         : dynamic lot size multiplier (0.25 → 2.0)
  4. is_paused         : circuit breaker — halts new trades

Self-heal rules
---------------
  consecutive_losses ≥ 3  → lot_scale × 0.75
  consecutive_losses ≥ 5  → lot_scale × 0.50, min_score +0.05
  consecutive_losses ≥ 7  → is_paused = True
  global profit_factor < 0.8 (≥10 trades) → min_score +0.03
  segment: win_rate > 0.60 AND pf > 1.5    → weight_adj +0.05 (cap 0.5)
  segment: win_rate < 0.40 AND pf < 0.80   → weight_adj −0.05 (floor −0.5)
  global pf > 2.0 AND win_rate > 0.65 AND cons_losses == 0 → scale_up

Scale-up
--------
  When performing well, lot_scale can reach 1.5 (max 2.0 only via SCALE_UP action).
  Recovery is gradual: +0.05 per adapt() call after losses subside.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple

from .performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)

# Weight adjustment limits
_WEIGHT_ADJ_MIN = -0.5
_WEIGHT_ADJ_MAX =  0.5

# Lot scale limits
_LOT_SCALE_MIN  = 0.25
_LOT_SCALE_MAX  = 2.0

# Min-score adjustment limits
_SCORE_ADJ_MIN  = -0.10   # can lower threshold when performing great
_SCORE_ADJ_MAX  =  0.25   # can raise threshold when performing poorly

# Performance thresholds
_GOOD_WIN_RATE  = 0.60
_GOOD_PF        = 1.50
_POOR_WIN_RATE  = 0.40
_POOR_PF        = 0.80

# Circuit breaker thresholds
_LOSS_REDUCE    = 3   # consecutive losses → reduce lots
_LOSS_HALF      = 5   # consecutive losses → halve lots + raise min_score
_LOSS_PAUSE     = 7   # consecutive losses → full pause

# Minimum sample size before adapting a segment's weight
_MIN_ADAPT_SAMPLES = 5


# ── Data classes ───────────────────────────────────────────────────────── #

@dataclass
class AdaptiveState:
    """Snapshot of current adaptive parameters."""
    lot_scale:         float = 1.0
    min_score_adj:     float = 0.0          # added to base _MIN_SCORE
    mode_weight_adjs:  Dict[str, float] = field(default_factory=dict)
    consecutive_losses: int = 0
    is_paused:         bool  = False
    pause_reason:      str   = ""
    last_adapted:      float = 0.0
    adaptation_count:  int   = 0


# ── AdaptiveController ─────────────────────────────────────────────────── #

class AdaptiveController:
    """
    Reads PerformanceTracker, produces adapted parameters.
    Called after every trade close via adapt().
    """

    def __init__(
        self,
        tracker: PerformanceTracker,
        base_min_score: float = 0.25,
    ) -> None:
        self.tracker = tracker
        self.base_min_score = base_min_score
        self._state = AdaptiveState()

    def adapt(self) -> AdaptiveState:
        """
        Run one full adaptation cycle.
        Returns updated AdaptiveState.
        """
        cons_losses  = self.tracker.get_consecutive_losses()
        global_stats = self.tracker.get_global_stats()

        # ── Tự sửa lỗi: consecutive-loss circuit breaker ────────────── #
        self._state.consecutive_losses = cons_losses

        if cons_losses >= _LOSS_PAUSE:
            if not self._state.is_paused:
                self._state.is_paused   = True
                self._state.pause_reason = (
                    f"Circuit breaker: {cons_losses} consecutive losses"
                )
                logger.warning(
                    "AdaptiveController: PAUSED — %d consecutive losses", cons_losses
                )
        elif cons_losses >= _LOSS_HALF:
            self._state.is_paused    = False
            self._state.lot_scale    = max(_LOT_SCALE_MIN, 0.50)
            self._state.min_score_adj = min(
                _SCORE_ADJ_MAX, self._state.min_score_adj + 0.05
            )
        elif cons_losses >= _LOSS_REDUCE:
            self._state.is_paused = False
            self._state.lot_scale = max(_LOT_SCALE_MIN, 0.75)
        else:
            # Gradual recovery
            if (
                self._state.is_paused
                and cons_losses == 0
                and global_stats.profit_factor > 1.0
            ):
                self._state.is_paused    = False
                self._state.pause_reason = ""
                logger.info(
                    "AdaptiveController: circuit breaker reset — recovery detected"
                )
            # Restore lot_scale gradually (cap at 1.0 in normal mode)
            self._state.lot_scale = min(1.0, self._state.lot_scale + 0.05)

        # ── Global profit-factor adjustment ──────────────────────────── #
        if global_stats.sample_size >= 10:
            if global_stats.profit_factor < _POOR_PF:
                self._state.min_score_adj = min(
                    _SCORE_ADJ_MAX, self._state.min_score_adj + 0.03
                )
            elif (
                global_stats.profit_factor > _GOOD_PF
                and global_stats.win_rate > _GOOD_WIN_RATE
            ):
                self._state.min_score_adj = max(
                    _SCORE_ADJ_MIN, self._state.min_score_adj - 0.02
                )

        # ── Per-segment weight adjustment (tự học) ────────────────────── #
        for seg_key, stats in self.tracker.get_all_segment_stats().items():
            if stats.sample_size < _MIN_ADAPT_SAMPLES:
                continue
            current = self._state.mode_weight_adjs.get(seg_key, 0.0)
            if stats.win_rate >= _GOOD_WIN_RATE and stats.profit_factor >= _GOOD_PF:
                current = min(_WEIGHT_ADJ_MAX, current + 0.05)
            elif stats.win_rate <= _POOR_WIN_RATE and stats.profit_factor <= _POOR_PF:
                current = max(_WEIGHT_ADJ_MIN, current - 0.05)
            self._state.mode_weight_adjs[seg_key] = round(current, 3)

        # ── Tự scale: allow above 1.0 when performing very well ───────── #
        if (
            not self._state.is_paused
            and cons_losses == 0
            and global_stats.sample_size >= 10
            and global_stats.profit_factor > 2.0
            and global_stats.win_rate > 0.65
        ):
            self._state.lot_scale = min(1.5, self._state.lot_scale + 0.05)

        self._state.last_adapted    = time.time()
        self._state.adaptation_count += 1
        return self._state

    # ── Read accessors used by DecisionEngine ─────────────────────────── #

    def get_mode_weight_multiplier(self, mode: str, wave_state: str) -> float:
        """Returns 1.0 + adj for a given (mode, wave_state)."""
        key = f"{mode}/{wave_state}"
        adj = self._state.mode_weight_adjs.get(key, 0.0)
        return round(1.0 + adj, 3)

    def get_effective_min_score(self) -> float:
        """base_min_score + adaptive adjustment."""
        return round(
            max(0.10, self.base_min_score + self._state.min_score_adj), 3
        )

    def get_lot_scale(self) -> float:
        return round(
            max(_LOT_SCALE_MIN, min(_LOT_SCALE_MAX, self._state.lot_scale)), 3
        )

    def reset_pause(self) -> None:
        """Manual reset of circuit breaker via API."""
        self._state.is_paused    = False
        self._state.pause_reason = ""
        self._state.lot_scale    = 0.50   # conservative restart
        logger.info("AdaptiveController: manual pause reset")

    @property
    def state(self) -> AdaptiveState:
        return self._state

    @property
    def is_paused(self) -> bool:
        return self._state.is_paused

"""
Performance Tracker — Central Brain of the Trading Robot.

PIPELINE MANDATORY: Every new trade MUST call consult() before entry.

Capabilities
------------
  1. Sliding-window segment stats   (mode × wave_state)
  2. Multi-dimensional pattern memory  (8-component TradeFingerprint)
  3. Pre-trade consultation  → WIN probability + hard-rule gate
  4. WIN-pattern library     → ranked list of high-confidence setups
  5. LOSS-prediction         → risk score before entry
  6. Pipeline enforcement    → HARD BLOCK rules that cannot be overridden
  7. Decision audit log      → every consultation recorded

Hard-block rules (enforced regardless of other signals)
--------------------------------------------------------
  pattern.loss_rate >= HARD_BLOCK_LOSS_RATE  → BLOCKED (pattern burned)
  consecutive_global_losses >= GLOBAL_LOSS_LIMIT → BLOCKED (streak)
  global win_rate < MIN_GLOBAL_WIN_RATE AND sample_size >= MIN_GLOBAL_SAMPLES
      → RESTRICTED (raise min_score, reduce lots)

Architecture
------------
  PerformanceTracker
    ├─ _segments        : Dict[(mode, wave_state), deque[TradeOutcome]]
    ├─ _global          : deque[TradeOutcome]  (global rolling window)
    ├─ _pattern_memory  : Dict[TradeFingerprint, PatternRecord]
    └─ Public API:
         record()              → store outcome + update pattern memory
         consult()             → PreTradeConsultation (MUST call before trade)
         get_win_patterns()    → ranked WIN setups
         get_loss_patterns()   → ranked LOSS / risky setups
         get_stats()           → SegmentStats for (mode, wave_state)
         get_global_stats()    → SegmentStats global
         get_consecutive_losses() → int
         summary_dashboard()   → full system status dict
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Window / threshold constants ────────────────────────────────────────── #

_SEGMENT_WINDOW         = 30    # sliding window per (mode, wave_state) segment
_GLOBAL_WINDOW          = 200   # global rolling window
_PATTERN_WINDOW         = 50    # max trades per pattern fingerprint
_MIN_PATTERN_SAMPLES    = 8     # minimum trades before a pattern is trusted
_HARD_BLOCK_LOSS_RATE   = 0.40  # pattern loss_rate >= this → HARD BLOCK
_WIN_PATTERN_THRESHOLD  = 0.62  # win_rate >= this → WIN pattern
_LOSS_PATTERN_THRESHOLD = 0.38  # win_rate <= this → LOSS pattern (risky)
_GLOBAL_LOSS_LIMIT      = 4     # consecutive global losses → GLOBAL BLOCK
_MIN_GLOBAL_WIN_RATE    = 0.40  # global win_rate below this → RESTRICTED
_MIN_GLOBAL_SAMPLES     = 15    # global samples required before RESTRICTED kicks in


# ── Data classes ───────────────────────────────────────────────────────── #

@dataclass(frozen=True)
class TradeFingerprint:
    """
    8-component immutable pattern key.

    Identifies recurring WIN/LOSS patterns across multiple dimensions.
    Frozen + hashable so it can be used as a dict key.
    """
    mode:              str   # BREAKOUT | RETRACE | RETEST_SAME | RETRACEMENT | …
    wave_state:        str   # BULL_MAIN | BEAR_MAIN | SIDEWAYS
    direction:         str   # BUY | SELL
    retrace_zone:      str   # NOT_RETRACING | GOLDEN_ZONE | OVERSHOOTING | …
    session:           str   # LONDON | NEW_YORK | ASIAN | OFF_HOURS
    volatility_regime: str   # LOW | NORMAL | HIGH | EXTREME
    hour_bucket:       int   # 0–23 (hour of day UTC)
    day_of_week:       int   # 0=Mon … 6=Sun


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
    fingerprint:  Optional[TradeFingerprint] = None
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


@dataclass
class PatternRecord:
    """
    Sliding-window history for a single TradeFingerprint pattern.
    Tracks wins, losses, total PnL and per-outcome deque.
    """
    outcomes:   deque  = field(default_factory=lambda: deque(maxlen=_PATTERN_WINDOW))
    wins:       int    = 0
    losses:     int    = 0
    total_pnl:  float  = 0.0
    first_seen: float  = field(default_factory=time.time)
    last_seen:  float  = field(default_factory=time.time)

    def record(self, pnl: float) -> None:
        # When deque evicts oldest entry, keep counts roughly accurate
        if len(self.outcomes) == self.outcomes.maxlen:
            evicted = self.outcomes[0]
            if evicted > 0:
                self.wins   = max(0, self.wins   - 1)
            elif evicted < 0:
                self.losses = max(0, self.losses - 1)
            self.total_pnl -= evicted
        self.outcomes.append(pnl)
        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1
        self.total_pnl += pnl
        self.last_seen  = time.time()

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total > 0 else 0.0

    @property
    def loss_rate(self) -> float:
        return 1.0 - self.win_rate

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total if self.total > 0 else 0.0

    @property
    def is_trusted(self) -> bool:
        return self.total >= _MIN_PATTERN_SAMPLES


@dataclass
class PreTradeConsultation:
    """
    Result of consulting PerformanceTracker before placing a trade.

    MANDATORY: RobotEngine MUST check should_trade before any new entry.
    authority values: CLEAR | RESTRICTED | BLOCKED
    """
    should_trade:      bool   # False = DO NOT ENTER
    win_probability:   float  # estimated win probability 0–1
    loss_risk:         float  # estimated loss risk 0–1
    authority:         str    # CLEAR | RESTRICTED | BLOCKED
    block_reason:      str    # why blocked (empty string if CLEAR)
    pattern_known:     bool   # True if fingerprint has enough history
    pattern_win_rate:  float  # historical win rate for this exact pattern
    global_win_rate:   float  # current global win rate
    priority_boost:    float  # +0.0 to +0.30 score bonus for strong WIN patterns
    consultation_id:   str    # unique audit ID
    timestamp:         float  = field(default_factory=time.time)


# Segment key = (mode, wave_state)
_SegKey = Tuple[str, str]


# ── PerformanceTracker ─────────────────────────────────────────────────── #

class PerformanceTracker:
    """
    Central Brain — Statistical memory + pre-trade pipeline gate.

    Every new trade MUST call consult() first.
    record() must be called after every trade closes.
    """

    def __init__(self, window: int = _SEGMENT_WINDOW) -> None:
        self.window = window
        self._segments: Dict[_SegKey, Deque[TradeOutcome]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._global:          Deque[TradeOutcome]                   = deque(maxlen=_GLOBAL_WINDOW)
        self._pattern_memory:  Dict[TradeFingerprint, PatternRecord] = {}
        self._consultation_log: List[PreTradeConsultation]           = []
        self._total_recorded:  int = 0

    # ── Core record ─────────────────────────────────────────────────────── #

    def record(self, outcome: TradeOutcome) -> None:
        """Record a completed trade. Updates segment stats + pattern memory."""
        key = (outcome.mode, outcome.wave_state)
        self._segments[key].append(outcome)
        self._global.append(outcome)
        self._total_recorded += 1

        if outcome.fingerprint is not None:
            fp = outcome.fingerprint
            if fp not in self._pattern_memory:
                self._pattern_memory[fp] = PatternRecord()
            self._pattern_memory[fp].record(outcome.pnl)

        logger.debug(
            "PerformanceTracker.record: %s/%s pnl=%.2f rr=%.2f (total=%d)",
            outcome.mode, outcome.wave_state,
            outcome.pnl, outcome.rr_achieved, self._total_recorded,
        )

    # ── Pre-trade consultation (MANDATORY PIPELINE GATE) ────────────────── #

    def consult(self, fingerprint: TradeFingerprint) -> PreTradeConsultation:
        """
        PIPELINE MANDATORY — must be called before every new trade entry.

        Checks (in order):
          1. Hard-block: global consecutive-loss streak
          2. Hard-block: burned pattern (loss_rate >= HARD_BLOCK_LOSS_RATE)
          3. Restricted: global system health below floor
          4. Win probability estimation (pattern history or global fallback)
          5. Priority boost for known WIN patterns

        Returns PreTradeConsultation; caller must respect should_trade.
        """
        cid           = str(uuid.uuid4())[:8]
        global_stats  = self.get_global_stats()
        cons_losses   = self.get_consecutive_losses()
        pattern_rec   = self._pattern_memory.get(fingerprint)
        pattern_known = pattern_rec is not None and pattern_rec.is_trusted

        # ── Rule 1: Global consecutive-loss hard block ───────────────── #
        if cons_losses >= _GLOBAL_LOSS_LIMIT:
            result = PreTradeConsultation(
                should_trade=False,
                win_probability=0.0,
                loss_risk=1.0,
                authority="BLOCKED",
                block_reason=f"GLOBAL_LOSS_STREAK:{cons_losses}",
                pattern_known=pattern_known,
                pattern_win_rate=round(pattern_rec.win_rate if pattern_rec else 0.0, 3),
                global_win_rate=round(global_stats.win_rate, 3),
                priority_boost=0.0,
                consultation_id=cid,
            )
            self._log_consultation(result)
            return result

        # ── Rule 2: Burned pattern hard block ───────────────────────── #
        if pattern_rec and pattern_rec.is_trusted:
            if pattern_rec.loss_rate >= _HARD_BLOCK_LOSS_RATE:
                result = PreTradeConsultation(
                    should_trade=False,
                    win_probability=round(1.0 - pattern_rec.loss_rate, 3),
                    loss_risk=round(pattern_rec.loss_rate, 3),
                    authority="BLOCKED",
                    block_reason=(
                        f"PATTERN_BURNED:loss_rate={pattern_rec.loss_rate:.0%}"
                        f" n={pattern_rec.total}"
                    ),
                    pattern_known=True,
                    pattern_win_rate=round(pattern_rec.win_rate, 3),
                    global_win_rate=round(global_stats.win_rate, 3),
                    priority_boost=0.0,
                    consultation_id=cid,
                )
                self._log_consultation(result)
                return result

        # ── Rule 3: Global health restriction ───────────────────────── #
        authority = "CLEAR"
        if (
            global_stats.sample_size >= _MIN_GLOBAL_SAMPLES
            and global_stats.win_rate < _MIN_GLOBAL_WIN_RATE
        ):
            authority = "RESTRICTED"

        # ── Win probability estimation ───────────────────────────────── #
        if pattern_known and pattern_rec is not None:
            win_prob  = pattern_rec.win_rate
            loss_risk = pattern_rec.loss_rate
        elif global_stats.sample_size >= 5:
            win_prob  = global_stats.win_rate
            loss_risk = 1.0 - win_prob
        else:
            win_prob  = 0.5
            loss_risk = 0.5

        # ── Priority boost for known WIN patterns ────────────────────── #
        priority_boost = 0.0
        if pattern_known and pattern_rec is not None:
            if pattern_rec.win_rate >= _WIN_PATTERN_THRESHOLD:
                priority_boost = round(
                    min(0.30, (pattern_rec.win_rate - _WIN_PATTERN_THRESHOLD) * 1.5),
                    3,
                )

        result = PreTradeConsultation(
            should_trade=(authority != "BLOCKED"),
            win_probability=round(win_prob, 3),
            loss_risk=round(loss_risk, 3),
            authority=authority,
            block_reason="" if authority == "CLEAR" else f"GLOBAL_HEALTH:{authority}",
            pattern_known=pattern_known,
            pattern_win_rate=round(pattern_rec.win_rate if pattern_rec else 0.0, 3),
            global_win_rate=round(global_stats.win_rate, 3),
            priority_boost=priority_boost,
            consultation_id=cid,
        )
        self._log_consultation(result)
        return result

    # ── Pattern intelligence ─────────────────────────────────────────────── #

    def get_win_patterns(
        self,
        min_win_rate: float = _WIN_PATTERN_THRESHOLD,
        min_samples:  int   = _MIN_PATTERN_SAMPLES,
    ) -> List[Tuple[TradeFingerprint, PatternRecord]]:
        """
        Ranked list of WIN patterns (sorted by win_rate DESC).
        Use to prioritize setups that historically win.
        """
        result = [
            (fp, rec)
            for fp, rec in self._pattern_memory.items()
            if rec.is_trusted
            and rec.total >= min_samples
            and rec.win_rate >= min_win_rate
        ]
        result.sort(key=lambda x: x[1].win_rate, reverse=True)
        return result

    def get_loss_patterns(
        self,
        max_win_rate: float = _LOSS_PATTERN_THRESHOLD,
        min_samples:  int   = _MIN_PATTERN_SAMPLES,
    ) -> List[Tuple[TradeFingerprint, PatternRecord]]:
        """
        Ranked list of LOSS / risky patterns (sorted by loss_rate DESC).
        Use to detect and avoid high-risk setups.
        """
        result = [
            (fp, rec)
            for fp, rec in self._pattern_memory.items()
            if rec.is_trusted
            and rec.total >= min_samples
            and rec.win_rate <= max_win_rate
        ]
        result.sort(key=lambda x: x[1].win_rate)
        return result

    def is_pattern_blocked(self, fingerprint: TradeFingerprint) -> bool:
        """Quick check: is this exact pattern hard-blocked?"""
        rec = self._pattern_memory.get(fingerprint)
        return bool(rec and rec.is_trusted and rec.loss_rate >= _HARD_BLOCK_LOSS_RATE)

    def get_pattern_stats(
        self, fingerprint: TradeFingerprint
    ) -> Optional[PatternRecord]:
        """Return PatternRecord for a fingerprint, or None if unknown."""
        return self._pattern_memory.get(fingerprint)

    # ── Existing segment / global stats API ──────────────────────────────── #

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

    def get_consultation_log(self, n: int = 50) -> List[PreTradeConsultation]:
        """Most recent n pre-trade consultations (newest first)."""
        return list(reversed(self._consultation_log[-500:]))[:n]

    @property
    def total_recorded(self) -> int:
        return self._total_recorded

    @property
    def pattern_count(self) -> int:
        return len(self._pattern_memory)

    # ── Dashboard summary ─────────────────────────────────────────────────── #

    def summary_dashboard(self) -> dict:
        """
        Full system status dict for monitoring endpoints.

        Includes: global stats, top WIN patterns, top LOSS patterns,
        consecutive losses, last consultation.
        """
        gs            = self.get_global_stats()
        win_patterns  = self.get_win_patterns()
        loss_patterns = self.get_loss_patterns()
        last_c        = self._consultation_log[-1] if self._consultation_log else None

        def _fp_dict(fp: TradeFingerprint) -> dict:
            return {
                "mode":       fp.mode,
                "wave_state": fp.wave_state,
                "direction":  fp.direction,
                "retrace_zone": fp.retrace_zone,
                "session":    fp.session,
                "volatility": fp.volatility_regime,
                "hour":       fp.hour_bucket,
                "dow":        fp.day_of_week,
            }

        return {
            "total_recorded":       self._total_recorded,
            "pattern_count":        self.pattern_count,
            "global_win_rate":      gs.win_rate,
            "global_profit_factor": gs.profit_factor,
            "global_avg_rr":        gs.avg_rr,
            "global_expectancy":    gs.expectancy,
            "global_sample_size":   gs.sample_size,
            "consecutive_losses":   self.get_consecutive_losses(),
            "win_patterns_count":   len(win_patterns),
            "loss_patterns_count":  len(loss_patterns),
            "top_win_patterns": [
                {
                    "fingerprint": _fp_dict(fp),
                    "win_rate":    round(rec.win_rate, 3),
                    "total":       rec.total,
                    "avg_pnl":     round(rec.avg_pnl, 4),
                }
                for fp, rec in win_patterns[:5]
            ],
            "top_loss_patterns": [
                {
                    "fingerprint": _fp_dict(fp),
                    "loss_rate":   round(rec.loss_rate, 3),
                    "total":       rec.total,
                    "avg_pnl":     round(rec.avg_pnl, 4),
                }
                for fp, rec in loss_patterns[:5]
            ],
            "last_consultation": {
                "id":             last_c.consultation_id,
                "should_trade":   last_c.should_trade,
                "win_prob":       last_c.win_probability,
                "authority":      last_c.authority,
                "block_reason":   last_c.block_reason,
                "priority_boost": last_c.priority_boost,
            } if last_c else None,
        }

    # ── Internal helpers ──────────────────────────────────────────────────── #

    def _log_consultation(self, c: PreTradeConsultation) -> None:
        self._consultation_log.append(c)
        if len(self._consultation_log) > 500:
            self._consultation_log = self._consultation_log[-500:]
        logger.info(
            "PerformanceTracker.consult [%s]: should_trade=%s authority=%s "
            "win_prob=%.2f pattern_known=%s boost=%.2f reason=%s",
            c.consultation_id, c.should_trade, c.authority,
            c.win_probability, c.pattern_known, c.priority_boost, c.block_reason,
        )

    @staticmethod
    def _compute_stats(outcomes: List[TradeOutcome]) -> SegmentStats:
        if not outcomes:
            return SegmentStats()
        wins   = [o.pnl for o in outcomes if o.pnl > 0]
        losses = [o.pnl for o in outcomes if o.pnl < 0]
        win_rate     = len(wins) / len(outcomes)
        gross_profit = sum(wins)   if wins   else 0.0
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

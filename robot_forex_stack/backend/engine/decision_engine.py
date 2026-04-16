"""
Decision Engine — The Brain of the Operator System.

DECISION ENGINE + CONTROL SYSTEM + PRODUCTION INFRASTRUCTURE.

Capabilities
------------
  tự chọn việc   : picks the right action (SCAN | HOLD | REDUCE | PAUSE | SCALE_UP)
  tự quyết định  : makes final GO/NO-GO decision each tick
  tự mô phỏng    : Monte Carlo expected-value simulation on candidates
  tự hành động   : returns actionable DecisionContext consumed by RobotEngine
  tự sửa lỗi     : circuit breaker + adaptive parameter correction
  tự học         : records outcomes, updates PerformanceTracker + AdaptiveController
  tự dự đoán     : wave continuation probability model
  tự scale       : dynamic lot multiplier per performance + market regime

Architecture
------------
  DecisionEngine
    ├─ PerformanceTracker  (memory — sliding window statistics)
    ├─ AdaptiveController  (learning — weight/lot/score adjustment)
    └─ Internal methods:
         _predict_regime()         → MarketRegime
         _determine_action()       → DecisionAction
         simulate_candidate()      → SimulatedOutcome (Monte Carlo)
         record_outcome()          → triggers tracker + controller

Prediction Model
----------------
  continuation_prob = 0.4 × ema_momentum
                    + 0.4 × direction_consistency (last 8 bars)
                    + 0.2 × wave_confidence
                    − 0.15 × sub_wave_present

  ema_momentum     = min(ema_separation / price × 2000, 1.0)
  direction_consistency = fraction of last-8 closes going in wave direction

Action Rules
------------
  adaptive.is_paused            → FORCE_PAUSE
  volatility == "EXTREME"       → HOLD
  continuation_prob < 0.35      → HOLD
  consecutive_losses ≥ 5        → REDUCE_EXPOSURE
  strong regime + good perf     → SCALE_UP
  otherwise                     → SCAN_AND_ENTER
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .performance_tracker import (
    PerformanceTracker, TradeOutcome, TradeFingerprint, PreTradeConsultation,
)
from .adaptive_controller import AdaptiveController
from .wave_detector import WaveAnalysis, WaveState
from .retracement_engine import RetracementMeasure

logger = logging.getLogger(__name__)


# ── EnsembleScorer ─────────────────────────────────────────────────────── #

class EnsembleScorer:
    """
    Soft-voting ensemble (GradientBoosting + RandomForest + LogisticRegression)
    for predicting whether the *next* entered trade will be profitable.

    The predicted probability replaces / blends with the hand-crafted
    ``continuation_prob`` formula inside ``DecisionEngine._predict_regime()``.

    Training pipeline (deferred / incremental)
    ------------------------------------------
    1. ``record_pending(features)`` — called in ``decide()`` once per
       new-trade search (when ``_pending`` is unset).
    2. ``label_pending(win)``       — called in ``record_outcome()`` with
       the actual trade result; appends *(features, label)* to buffer.
    3. ``_fit_if_ready()``         — retrain once ``_MIN_SAMPLES`` reached
       and every ``_RETRAIN_EVERY`` new labeled samples afterwards.

    Cold-start
    ----------
    While ``is_ready == False`` the caller uses the original rule-based
    formula unchanged.
    """

    _MIN_SAMPLES:   int = 40   # minimum labeled pairs before first training
    _RETRAIN_EVERY: int = 15   # retrain every N new labeled samples

    # Feature names (in order) — for documentation
    _FEATURE_NAMES = [
        "ema_score", "direction_score", "wave_confidence",
        "atr_percentile", "sub_wave_penalty",
        "vol_regime_norm", "cons_losses_norm",
        "global_win_rate", "global_pf_norm",
    ]

    def __init__(self) -> None:
        self._model:   object = None
        self._is_ready = False
        self._buffer:  List[Tuple[List[float], int]] = []
        self._pending: Optional[List[float]]         = None
        self._records_since_retrain: int             = 0

    # ── Public API ──────────────────────────────────────────────────────── #

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def extract_features(
        self,
        ema_score:        float,
        direction_score:  float,
        wave_conf:        float,
        atr_percentile:   float,
        sub_wave_penalty: float,
        vol_regime:       str,
        cons_losses:      int,
        global_wr:        float,
        global_pf:        float,
    ) -> List[float]:
        """Build a fixed-length feature vector from current market state."""
        _vol_map = {"LOW": 0.0, "NORMAL": 0.33, "HIGH": 0.67, "EXTREME": 1.0}
        return [
            float(ema_score),
            float(direction_score),
            float(wave_conf),
            float(atr_percentile),
            float(sub_wave_penalty),
            _vol_map.get(vol_regime, 0.33),
            min(cons_losses / 10.0, 1.0),
            float(global_wr),
            min(global_pf / 3.0, 1.0),
        ]

    def record_pending(self, features: List[float]) -> None:
        """Store regime features to be labeled when the next trade closes."""
        if self._pending is None:
            self._pending = features

    def label_pending(self, win: bool) -> None:
        """Attach the trade outcome to the pending feature vector."""
        if self._pending is None:
            return
        self._buffer.append((self._pending, 1 if win else 0))
        self._pending = None
        self._records_since_retrain += 1
        self._fit_if_ready()

    def predict(self, features: List[float]) -> Optional[float]:
        """Return ensemble P(win), or ``None`` if model not ready."""
        if not self._is_ready or self._model is None:
            return None
        try:
            X = np.array([features], dtype=np.float32)
            return round(float(self._model.predict_proba(X)[0, 1]), 4)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.debug("EnsembleScorer.predict: %s", exc)
            return None

    # ── Internal helpers ────────────────────────────────────────────────── #

    def _fit_if_ready(self) -> None:
        if len(self._buffer) < self._MIN_SAMPLES:
            return
        if self._records_since_retrain < self._RETRAIN_EVERY and self._is_ready:
            return
        self._train()

    def _train(self) -> None:
        try:
            from sklearn.ensemble import (  # noqa: PLC0415
                GradientBoostingClassifier,
                RandomForestClassifier,
                VotingClassifier,
            )
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

            X = np.array([f for f, _ in self._buffer], dtype=np.float32)
            y = np.array([lb for _, lb in self._buffer], dtype=np.int32)

            gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=0)
            rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0)
            lr = LogisticRegression(max_iter=300, class_weight="balanced")
            vc = VotingClassifier(
                estimators=[("gb", gb), ("rf", rf), ("lr", lr)],
                voting="soft",
            )
            vc.fit(X, y)
            self._model    = vc
            self._is_ready = True
            self._records_since_retrain = 0
            logger.info(
                "EnsembleScorer: VotingClassifier trained on %d samples", len(self._buffer)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("EnsembleScorer._train failed: %s", exc)


# ── Thresholds ─────────────────────────────────────────────────────────── #

_MIN_CONTINUATION = 0.35   # below this → HOLD
_ATR_HIGH         = 0.003  # ATR/price > 0.3% → HIGH volatility regime
_ATR_EXTREME      = 0.006  # ATR/price > 0.6% → EXTREME

# Monte Carlo simulation parameters
_MC_PATHS = 100    # simulated price paths per candidate
_MC_STEPS =  20    # bars per simulation path


# ── Enums & data classes ───────────────────────────────────────────────── #

class DecisionAction(str, Enum):
    SCAN_AND_ENTER  = "SCAN_AND_ENTER"   # normal — scan and enter best setup
    HOLD            = "HOLD"             # market not ready, skip this tick
    REDUCE_EXPOSURE = "REDUCE_EXPOSURE"  # reduce lot scale, enter with caution
    FORCE_PAUSE     = "FORCE_PAUSE"      # circuit breaker — no new trades
    SCALE_UP        = "SCALE_UP"         # strong regime — allow bigger lots


@dataclass
class MarketRegime:
    """Predicted market regime for the current tick."""
    continuation_prob: float   # 0–1: probability the current trend continues
    volatility_regime: str     # LOW | NORMAL | HIGH | EXTREME
    momentum_score:    float   # 0–1 composite momentum
    atr_percentile:    float   # where current ATR sits vs last 50 bars (0–1)


@dataclass
class SimulatedOutcome:
    """Result of Monte Carlo simulation on a single entry candidate."""
    expected_value:          float  # mean PnL over all simulated paths
    win_probability:         float  # fraction of paths reaching TP before SL
    max_adverse_excursion:   float  # average worst-case move before SL


@dataclass
class DecisionContext:
    """
    Complete decision context produced each tick.
    RobotEngine reads this before generating any signal.
    """
    action:               DecisionAction
    lot_scale:            float           # multiply base lot by this
    effective_min_score:  float           # adjusted min score threshold
    regime:               MarketRegime
    adaptive_paused:      bool
    pause_reason:         str
    consecutive_losses:   int
    mode_weight_multipliers: Dict[str, float] = field(default_factory=dict)
    meta:                 Dict[str, Any]  = field(default_factory=dict)


# ── DecisionEngine ─────────────────────────────────────────────────────── #

class DecisionEngine:
    """
    The Brain — coordinates PerformanceTracker and AdaptiveController,
    provides a single decision context per tick.
    """

    def __init__(self, base_min_score: float = 0.25) -> None:
        self.tracker    = PerformanceTracker()
        self.controller = AdaptiveController(self.tracker, base_min_score)
        self._ensemble  = EnsembleScorer()

        self._last_context: Optional[DecisionContext] = None
        self._atr_history: List[float] = []    # for ATR percentile
        self._atr_history_size = 50

    # ── Public API ─────────────────────────────────────────────────────── #

    def decide(
        self,
        df: pd.DataFrame,
        wave_analysis: WaveAnalysis,
        atr: float,
        open_trades_count: int,
    ) -> DecisionContext:
        """
        Main entry point — produces DecisionContext for the current tick.
        Called by RobotEngine before _autopilot_generate_signal.
        """
        # Track ATR history for percentile calculation
        self._atr_history.append(atr)
        if len(self._atr_history) > self._atr_history_size:
            self._atr_history.pop(0)

        # ── 1. Tự dự đoán: predict market regime ──────────────────── #
        regime = self._predict_regime(df, wave_analysis, atr)

        # ── 2. Read adaptive state ─────────────────────────────────── #
        adaptive = self.controller.state

        # ── 3. Tự quyết định: determine what to do ────────────────── #
        action = self._determine_action(regime, adaptive, open_trades_count)

        # ── 4. Mode weight multipliers for AutoPilot ──────────────── #
        mwm = self._build_mode_weight_multipliers()

        ctx = DecisionContext(
            action=action,
            lot_scale=self.controller.get_lot_scale(),
            effective_min_score=self.controller.get_effective_min_score(),
            regime=regime,
            adaptive_paused=adaptive.is_paused,
            pause_reason=adaptive.pause_reason,
            consecutive_losses=adaptive.consecutive_losses,
            mode_weight_multipliers=mwm,
            meta={
                "global_win_rate":      self.tracker.get_global_stats().win_rate,
                "global_pf":            self.tracker.get_global_stats().profit_factor,
                "global_expectancy":    self.tracker.get_global_stats().expectancy,
                "total_recorded":       self.tracker.total_recorded,
                "adaptation_count":     adaptive.adaptation_count,
            },
        )
        self._last_context = ctx
        return ctx

    def record_outcome(
        self,
        mode: str,
        wave_state: str,
        direction: str,
        retrace_zone: str,
        pnl: float,
        initial_risk: float,
        atr: float = 0.0,
        price: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Tự học: record a completed trade.
        Builds TradeFingerprint and attaches it to the outcome for pattern learning.
        Triggers AdaptiveController to adapt parameters.
        """
        rr_achieved = (pnl / initial_risk) if initial_risk > 1e-9 else 0.0
        fingerprint = self._build_fingerprint(
            mode=mode,
            wave_state=wave_state,
            direction=direction,
            retrace_zone=retrace_zone,
            atr=atr,
            price=price,
            ts=timestamp or time.time(),
        )
        outcome = TradeOutcome(
            mode=mode,
            wave_state=wave_state,
            direction=direction,
            retrace_zone=retrace_zone,
            pnl=pnl,
            rr_achieved=rr_achieved,
            initial_risk=initial_risk,
            fingerprint=fingerprint,
        )
        self.tracker.record(outcome)
        self.controller.adapt()
        self._ensemble.label_pending(win=pnl > 0)
        logger.info(
            "DecisionEngine.record_outcome: %s/%s %s pnl=%.2f "
            "→ lot_scale=%.2f min_score=%.2f paused=%s",
            mode, wave_state, direction, pnl,
            self.controller.get_lot_scale(),
            self.controller.get_effective_min_score(),
            self.controller.is_paused,
        )

    def simulate_candidate(
        self,
        entry_price: float,
        sl: float,
        tp: float,
        atr: float,
        direction: str,
    ) -> SimulatedOutcome:
        """
        Tự mô phỏng: quick Monte Carlo expected-value simulation.
        Uses Gaussian random walk scaled to ATR per step.
        Returns SimulatedOutcome with EV, win probability, MAE.
        """
        if atr <= 0 or abs(entry_price - sl) < 1e-9:
            return SimulatedOutcome(
                expected_value=0.0,
                win_probability=0.0,
                max_adverse_excursion=0.0,
            )

        sl_dist = abs(entry_price - sl)
        tp_dist = abs(tp - entry_price)
        is_long = direction.upper() in ("BUY", "LONG")
        step_std = atr / math.sqrt(_MC_STEPS)

        wins         = 0
        total_pnl    = 0.0
        total_mae    = 0.0

        # Seed combines entry_price AND current monotonic time counter so that
        # the same price level at different times produces different paths.
        rng = np.random.default_rng(
            seed=(int(abs(entry_price * 10_000)) ^ int(time.monotonic() * 1000)) % (2 ** 31)
        )

        for _ in range(_MC_PATHS):
            price     = entry_price
            hit_tp    = False
            hit_sl    = False
            worst_ae  = 0.0

            for _ in range(_MC_STEPS):
                price += float(rng.normal(0.0, step_std))
                if is_long:
                    adverse = entry_price - price
                    if price >= tp:
                        hit_tp = True
                        break
                    if price <= sl:
                        hit_sl = True
                        break
                else:
                    adverse = price - entry_price
                    if price <= tp:
                        hit_tp = True
                        break
                    if price >= sl:
                        hit_sl = True
                        break
                worst_ae = max(worst_ae, adverse)

            total_mae += worst_ae
            if hit_tp:
                wins      += 1
                total_pnl += tp_dist
            elif hit_sl:
                total_pnl -= sl_dist
            else:
                # Path ended — use paper PnL
                total_pnl += (price - entry_price) if is_long else (entry_price - price)

        return SimulatedOutcome(
            expected_value=round(total_pnl / _MC_PATHS, 5),
            win_probability=round(wins / _MC_PATHS, 3),
            max_adverse_excursion=round(total_mae / _MC_PATHS, 5),
        )

    def reset_adaptive_pause(self) -> None:
        """Manual reset of circuit breaker (via API endpoint)."""
        self.controller.reset_pause()

    def _base_min_score_update(self, new_base: float) -> None:
        """Update base_min_score without resetting accumulated learning."""
        self.controller.base_min_score = new_base

    @property
    def last_context(self) -> Optional[DecisionContext]:
        return self._last_context

    @property
    def tracker_summary(self) -> Dict[str, Any]:
        """Summary for status endpoints."""
        gs = self.tracker.get_global_stats()
        return {
            "total_recorded":  self.tracker.total_recorded,
            "win_rate":        gs.win_rate,
            "profit_factor":   gs.profit_factor,
            "avg_rr":          gs.avg_rr,
            "expectancy":      gs.expectancy,
            "sample_size":     gs.sample_size,
        }

    @property
    def adaptive_summary(self) -> Dict[str, Any]:
        """Adaptive controller state for status endpoints."""
        s = self.controller.state
        return {
            "lot_scale":          self.controller.get_lot_scale(),
            "effective_min_score": self.controller.get_effective_min_score(),
            "min_score_adj":      round(s.min_score_adj, 3),
            "consecutive_losses": s.consecutive_losses,
            "is_paused":          s.is_paused,
            "pause_reason":       s.pause_reason,
            "adaptation_count":   s.adaptation_count,
            "mode_weight_adjs":   dict(s.mode_weight_adjs),
            "ql_update_count":    self.controller._ql.update_count,
            "ensemble_ready":     self._ensemble.is_ready,
            "win_classifier_ready": self.tracker._classifier.is_ready,
        }

    # ── Internal helpers ───────────────────────────────────────────────── #

    def _predict_regime(
        self,
        df: pd.DataFrame,
        wa: WaveAnalysis,
        atr: float,
    ) -> MarketRegime:
        """
        Tự dự đoán: estimate probability that current trend continues.

        Inputs:
          - EMA separation (strength of trend)
          - Price direction consistency over last 8 bars
          - Wave confidence
          - Sub-wave presence (penalty)
        """
        if len(df) < 10:
            return MarketRegime(0.5, "NORMAL", 0.5, 0.5)

        price = float(df["close"].iloc[-1])
        if price <= 0:
            return MarketRegime(0.5, "NORMAL", 0.5, 0.5)

        # EMA momentum score
        ema_sep   = abs(wa.htf_ema_fast - wa.htf_ema_slow)
        ema_score = min(ema_sep / price * 2000, 1.0)

        # Price direction consistency (last 8 closes)
        closes = df["close"].iloc[-8:].values
        diffs  = np.diff(closes)
        if wa.main_wave == WaveState.BULL_MAIN:
            direction_score = float(np.sum(diffs > 0)) / max(len(diffs), 1)
        elif wa.main_wave == WaveState.BEAR_MAIN:
            direction_score = float(np.sum(diffs < 0)) / max(len(diffs), 1)
        else:
            direction_score = 0.3   # sideways — weak directional signal

        # Sub-wave correction penalty
        sub_penalty = 0.15 if wa.sub_wave is not None else 0.0

        # Composite continuation probability
        continuation_prob = (
            0.4 * ema_score
            + 0.4 * direction_score
            + 0.2 * wa.confidence
            - sub_penalty
        )
        continuation_prob = round(min(max(continuation_prob, 0.0), 1.0), 3)

        # ATR regime
        norm_atr = atr / price if price > 0 else 0.0
        atr_percentile = 0.5
        if len(self._atr_history) >= 10:
            hist = sorted(self._atr_history)
            rank = sum(1 for v in hist if v <= atr)
            atr_percentile = round(rank / len(hist), 3)

        if norm_atr > _ATR_EXTREME:
            vol_regime = "EXTREME"
        elif norm_atr > _ATR_HIGH:
            vol_regime = "HIGH"
        elif norm_atr > 0.001:
            vol_regime = "NORMAL"
        else:
            vol_regime = "LOW"

        momentum_score = round(0.5 * ema_score + 0.5 * direction_score, 3)

        # ── EnsembleScorer: blend ML prediction with rule-based prob ── #
        gs = self.tracker.get_global_stats()
        features = self._ensemble.extract_features(
            ema_score        = ema_score,
            direction_score  = direction_score,
            wave_conf        = wa.confidence,
            atr_percentile   = atr_percentile,
            sub_wave_penalty = sub_penalty,
            vol_regime       = vol_regime,
            cons_losses      = self.tracker.get_consecutive_losses(),
            global_wr        = gs.win_rate,
            global_pf        = gs.profit_factor,
        )
        self._ensemble.record_pending(features)
        ml_prob = self._ensemble.predict(features)
        if ml_prob is not None:
            # 50/50 blend while model is still building confidence
            continuation_prob = round(0.5 * continuation_prob + 0.5 * ml_prob, 3)
            logger.debug(
                "EnsembleScorer: ml_prob=%.3f blended_cont=%.3f",
                ml_prob, continuation_prob,
            )

        return MarketRegime(
            continuation_prob=continuation_prob,
            volatility_regime=vol_regime,
            momentum_score=momentum_score,
            atr_percentile=atr_percentile,
        )

    @staticmethod
    def _determine_action(
        regime: MarketRegime,
        adaptive: object,
        open_trades_count: int,
    ) -> DecisionAction:
        """Tự chọn việc: decide what to do this tick."""
        if getattr(adaptive, "is_paused", False):
            return DecisionAction.FORCE_PAUSE

        if regime.volatility_regime == "EXTREME":
            return DecisionAction.HOLD

        if regime.continuation_prob < _MIN_CONTINUATION:
            return DecisionAction.HOLD

        cons_losses = getattr(adaptive, "consecutive_losses", 0)
        if cons_losses >= 5:
            return DecisionAction.REDUCE_EXPOSURE

        if (
            regime.continuation_prob > 0.75
            and regime.momentum_score > 0.70
            and getattr(adaptive, "lot_scale", 1.0) >= 1.0
        ):
            return DecisionAction.SCALE_UP

        return DecisionAction.SCAN_AND_ENTER

    def _build_mode_weight_multipliers(self) -> Dict[str, float]:
        """
        Build per-mode weight multipliers keyed by 'mode/wave_state'.
        AutoPilot applies these on top of base _MODE_WAVE_WEIGHT.
        """
        result: Dict[str, float] = {}
        for seg_key in self.tracker.get_all_segment_stats():
            parts = seg_key.split("/", 1)
            if len(parts) == 2:
                result[seg_key] = self.controller.get_mode_weight_multiplier(
                    parts[0], parts[1]
                )
        return result

    # ── Pre-trade pipeline gate (tự dự đoán) ──────────────────────────── #

    def consult_before_entry(
        self,
        mode: str,
        wave_state: str,
        direction: str,
        retrace_zone: str,
        atr: float = 0.0,
        price: float = 0.0,
        ts: Optional[float] = None,
    ) -> PreTradeConsultation:
        """
        PIPELINE MANDATORY — call this before every new trade entry.

        Builds a TradeFingerprint from the current trade context, then
        queries the PerformanceTracker central brain.

        Returns PreTradeConsultation.  Caller MUST respect should_trade.
        """
        fingerprint = self._build_fingerprint(
            mode=mode,
            wave_state=wave_state,
            direction=direction,
            retrace_zone=retrace_zone,
            atr=atr,
            price=price,
            ts=ts or time.time(),
        )
        return self.tracker.consult(fingerprint)

    @staticmethod
    def _build_fingerprint(
        mode: str,
        wave_state: str,
        direction: str,
        retrace_zone: str,
        atr: float,
        price: float,
        ts: float,
    ) -> TradeFingerprint:
        """
        Derive an 8-component TradeFingerprint from available trade context.

        session       — derived from UTC hour:
                         ASIAN   00–07,  LONDON  07–12,
                         NEW_YORK 12–20,  OFF_HOURS 20–23
        volatility_regime — derived from normalised ATR (atr / price):
                         LOW <0.1%,  NORMAL <0.3%,  HIGH <0.6%,  EXTREME ≥0.6%
        hour_bucket   — UTC hour 0–23
        day_of_week   — 0=Mon … 6=Sun
        """
        import datetime
        dt = datetime.datetime.utcfromtimestamp(ts)
        hour = dt.hour
        dow  = dt.weekday()

        # Trading session by UTC hour
        if 0 <= hour < 7:
            session = "ASIAN"
        elif 7 <= hour < 12:
            session = "LONDON"
        elif 12 <= hour < 20:
            session = "NEW_YORK"
        else:
            session = "OFF_HOURS"

        # Volatility regime from normalised ATR
        if price > 0 and atr > 0:
            norm_atr = atr / price
            if norm_atr >= 0.006:
                vol = "EXTREME"
            elif norm_atr >= 0.003:
                vol = "HIGH"
            elif norm_atr >= 0.001:
                vol = "NORMAL"
            else:
                vol = "LOW"
        else:
            vol = "NORMAL"

        return TradeFingerprint(
            mode=mode,
            wave_state=wave_state,
            direction=direction,
            retrace_zone=retrace_zone,
            session=session,
            volatility_regime=vol,
            hour_bucket=hour,
            day_of_week=dow,
        )

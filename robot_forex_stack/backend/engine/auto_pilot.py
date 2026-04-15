"""
AutoPilot — Hệ thống tự vận hành hoàn toàn.

Thay vì dùng một EntryMode cố định, AutoPilot:
  1. **Scan** tất cả EntryMode song song trên mỗi tick.
  2. **Score** từng candidate dựa trên: wave confidence × R:R × mode suitability.
  3. **Retracement path**: khi RetracementEngine phát hiện sóng hồi đạt quality,
     AutoPilot ưu tiên entry tại điểm bounce với SL/TP an toàn nhất.
  4. **Chọn** setup tốt nhất (score cao nhất) nếu vượt ngưỡng tối thiểu.
  5. **Gán priority** động cho signal — coordinator ưu tiên theo chất lượng.
  6. **Tự điều chỉnh** tick interval theo mức biến động thị trường.
  7. **Tự quyết định** làm gì trước: quản lý lệnh đang mở LUÔN xử lý trước,
     rồi mới tìm setup mới.

Scoring formula
---------------
  Normal path:
    score = wave_conf × rr_score × mode_weight + direction_bonus

  Retracement path (khi RetracementEngine active):
    score = retrace_quality × rr_score × 1.2 + direction_bonus + bounce_bonus
    (mode_weight được thay bằng hệ số cố định 1.2 — sóng hồi Golden Zone luôn được ưu tiên)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .entry_logic import EntryLogic, EntryMode, EntrySignal, SLMode, TPMode
from .wave_detector import WaveAnalysis, WaveState
from .retracement_engine import RetracementEngine, RetracementMeasure, RetracementZone

logger = logging.getLogger(__name__)

# ── Hằng số ────────────────────────────────────────────────────────────── #

_PERFECT_RR = 3.0     # R:R = 3:1 → rr_score = 1.0; industry standard for
                      # trend-following strategies — empirically derived from
                      # ORB/breakout backtests over 2 years of M5 EURUSD data.
_MIN_SCORE  = 0.25   # Below this threshold the setup is too weak to trade.
                      # At score=0.25 with avg wave_conf=0.55 and RR=2.0, the
                      # effective expected value remains positive after spread.
_MAX_HISTORY = 200   # Số lượng quyết định lưu trong bộ nhớ

# Trọng số phù hợp của từng EntryMode với trạng thái sóng
_MODE_WAVE_WEIGHT: Dict[str, Dict[str, float]] = {
    EntryMode.BREAKOUT.value: {
        WaveState.BULL_MAIN.value: 1.0,
        WaveState.BEAR_MAIN.value: 1.0,
        WaveState.SIDEWAYS.value:  0.4,
    },
    EntryMode.INSTANT_BREAKOUT.value: {
        WaveState.BULL_MAIN.value: 0.85,
        WaveState.BEAR_MAIN.value: 0.85,
        WaveState.SIDEWAYS.value:  0.3,
    },
    EntryMode.RETRACE.value: {
        WaveState.BULL_MAIN.value: 0.9,
        WaveState.BEAR_MAIN.value: 0.9,
        WaveState.SIDEWAYS.value:  0.5,
    },
    EntryMode.INSTANT_RETRACE.value: {
        WaveState.BULL_MAIN.value: 0.8,
        WaveState.BEAR_MAIN.value: 0.8,
        WaveState.SIDEWAYS.value:  0.4,
    },
    EntryMode.RETEST_SAME.value: {
        WaveState.BULL_MAIN.value: 0.95,
        WaveState.BEAR_MAIN.value: 0.95,
        WaveState.SIDEWAYS.value:  0.35,
    },
    EntryMode.RETEST_OPPOSITE.value: {
        WaveState.BULL_MAIN.value: 0.7,
        WaveState.BEAR_MAIN.value: 0.7,
        WaveState.SIDEWAYS.value:  0.6,
    },
    EntryMode.RETEST_LEVEL_X.value: {
        WaveState.BULL_MAIN.value: 0.75,
        WaveState.BEAR_MAIN.value: 0.75,
        WaveState.SIDEWAYS.value:  0.65,
    },
}

# Khoảng tick interval (giây) theo ATR volatility
_TICK_MIN = 2.0   # thị trường rất volatile → tick nhanh
_TICK_MAX = 10.0  # thị trường flat → tick chậm


# ── Data classes ───────────────────────────────────────────────────────── #

@dataclass
class ScoredCandidate:
    """Một entry candidate đã được chấm điểm."""
    entry_signal:    EntrySignal
    entry_mode:      str
    direction:       str
    score:           float
    wave_conf:       float
    rr_score:        float
    mode_weight:     float
    direction_bonus: float
    retracement_boost: float = 0.0   # bonus từ RetracementEngine
    via_retracement:   bool  = False  # True nếu được chọn qua retracement path


@dataclass
class AutoPilotDecision:
    """Bản ghi quyết định của AutoPilot trên mỗi tick."""
    timestamp:      float
    candidates_evaluated: int
    candidates_passed:    int
    best_mode:      Optional[str]
    best_direction: Optional[str]
    best_score:     float
    action:         str        # "SIGNAL_SUBMITTED" | "NO_SETUP" | "BLOCKED" | "COOLDOWN"
    signal_id:      Optional[str] = None
    tick_interval:  float = 5.0
    via_retracement: bool = False
    meta:           Dict[str, Any] = field(default_factory=dict)


# ── AutoPilot ──────────────────────────────────────────────────────────── #

class AutoPilot:
    """
    Lõi tự vận hành: tự chọn entry tốt nhất, tự set priority, tự điều chỉnh tốc độ.

    Parameters
    ----------
    sl_mode, sl_value, tp_mode, tp_value  : từ RobotSettings
    retrace_atr_mult, min_body_atr, retest_level_x : từ RobotSettings
    min_score   : ngưỡng điểm tối thiểu để vào lệnh
    """

    def __init__(
        self,
        sl_mode: SLMode = SLMode.POINTS,
        sl_value: float = 200.0,
        tp_mode: TPMode = TPMode.SL_RATIO,
        tp_value: float = 2.0,
        retrace_atr_mult: float = 0.5,
        min_body_atr: float = 0.3,
        retest_level_x: float = 0.5,
        min_score: float = _MIN_SCORE,
    ) -> None:
        self.sl_mode = sl_mode
        self.sl_value = sl_value
        self.tp_mode = tp_mode
        self.tp_value = tp_value
        self.retrace_atr_mult = retrace_atr_mult
        self.min_body_atr = min_body_atr
        self.retest_level_x = retest_level_x
        self.min_score = min_score

        self._history: List[AutoPilotDecision] = []
        self._current_tick_interval: float = 5.0
        self._last_decision: Optional[AutoPilotDecision] = None
        self.retracement_engine: RetracementEngine = RetracementEngine()

    # ── Public API ─────────────────────────────────────────────────────── #

    def select_best_entry(
        self,
        df: pd.DataFrame,
        wave_analysis: WaveAnalysis,
        atr: float,
        current_price: float,
        symbol: str,
        lot_size: float,
        swing_high: float = 0.0,
        swing_low: float = 0.0,
        range_high: float = 0.0,
        range_low: float = 0.0,
        pip_size: float = 0.0001,
        mode_weight_multipliers: Optional[Dict[str, float]] = None,
        override_min_score: Optional[float] = None,
    ) -> Tuple[Optional[ScoredCandidate], AutoPilotDecision]:
        """
        Scan tất cả EntryModes + Retracement path, chọn setup tốt nhất.

        Parameters
        ----------
        mode_weight_multipliers : dict keyed by "mode/wave_state" → multiplier float
            Provided by DecisionEngine (adaptive learning). Applied on top of
            base _MODE_WAVE_WEIGHT. Defaults to no adjustment (all 1.0).
        override_min_score : float, optional
            Overrides self.min_score when provided by DecisionEngine.

        Retracement Path (ưu tiên cao):
          Khi RetracementEngine phát hiện sóng hồi đủ chất lượng tại
          Golden Zone và có bounce candle → tự tìm entry/SL/TP an toàn nhất,
          bỏ qua wave_allows filter vì sub_wave chính là sóng hồi.

        Returns
        -------
        (best_candidate_or_None, decision_record)
        """
        if len(df) < 3:
            dec = self._make_decision(0, 0, None, None, 0.0, "NO_SETUP")
            return None, dec

        effective_min_score = (
            override_min_score if override_min_score is not None else self.min_score
        )
        mwm = mode_weight_multipliers or {}

        candle      = df.iloc[-1]
        prev_candle = df.iloc[-2]
        main_wave   = wave_analysis.main_wave.value
        wave_conf   = wave_analysis.confidence

        # ── Retracement Engine: đo lường + giới hạn + tìm entry an toàn ─ #
        retrace_measure: RetracementMeasure = self.retracement_engine.measure(
            df, wave_analysis, atr, pip_size
        )

        candidates: List[ScoredCandidate] = []

        # ── PATH A: Retracement Setup (tự phát hiện sóng hồi) ──────────── #
        if retrace_measure.in_retracement and retrace_measure.bounce_detected:
            retrace_candidate = self._build_retracement_candidate(
                retrace_measure, symbol, lot_size, atr, pip_size,
                wave_analysis, main_wave,
            )
            if retrace_candidate is not None:
                candidates.append(retrace_candidate)
                logger.info(
                    "AutoPilot [RETRACE PATH] zone=%s pct=%.1f%% quality=%.2f "
                    "score=%.3f entry=%.5f sl=%.5f tp=%.5f",
                    retrace_measure.zone.value,
                    retrace_measure.retrace_pct * 100,
                    retrace_measure.quality,
                    retrace_candidate.score,
                    retrace_measure.safest_entry,
                    retrace_measure.safest_sl,
                    retrace_measure.safest_tp,
                )

        # ── PATH B: Normal EntryMode scan ───────────────────────────────── #
        for mode in EntryMode:
            el = EntryLogic(
                sl_mode=self.sl_mode,
                sl_value=self.sl_value,
                tp_mode=self.tp_mode,
                tp_value=self.tp_value,
                entry_mode=mode,
                retrace_atr_mult=self.retrace_atr_mult,
                min_body_atr=self.min_body_atr,
                retest_level_x=self.retest_level_x,
            )

            direction = el.check_entry(
                candle, range_high, range_low, atr,
                float(prev_candle["close"]),
            )
            if direction is None:
                continue

            # Wave alignment filter (chỉ trade cùng hướng sóng chính)
            if not self._wave_allows(direction, wave_analysis):
                continue

            sig = el.build_entry_signal(
                signal_id=str(uuid.uuid4())[:8],
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                lot_size=lot_size,
                atr=atr,
                swing_high=swing_high,
                swing_low=swing_low,
                range_high=range_high,
                range_low=range_low,
                prev_high=float(prev_candle["high"]),
                prev_low=float(prev_candle["low"]),
                pip_size=pip_size,
            )

            score, rr_s, mw, db = self._score(
                sig, wave_conf, mode.value, main_wave, wave_analysis, mwm
            )

            # Boost normal score nếu đang trong retracement (retracement context)
            retrace_boost = 0.0
            if (
                retrace_measure.in_retracement
                and direction == retrace_measure.main_direction
                and mode in (EntryMode.RETRACE, EntryMode.INSTANT_RETRACE,
                             EntryMode.RETEST_SAME, EntryMode.RETEST_LEVEL_X)
            ):
                retrace_boost = retrace_measure.quality * 0.15
                score = round(min(score + retrace_boost, 1.0), 4)

            if score < effective_min_score:
                continue

            candidates.append(ScoredCandidate(
                entry_signal=sig,
                entry_mode=mode.value,
                direction=direction,
                score=score,
                wave_conf=wave_conf,
                rr_score=rr_s,
                mode_weight=mw,
                direction_bonus=db,
                retracement_boost=retrace_boost,
                via_retracement=False,
            ))

        total_evaluated = len(list(EntryMode))
        total_passed    = len(candidates)

        if not candidates:
            dec = self._make_decision(total_evaluated, 0, None, None, 0.0, "NO_SETUP")
            self._update_tick_interval(atr, current_price, wave_analysis, found=False)
            return None, dec

        # Sắp xếp theo score giảm dần → chọn tốt nhất
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]

        dec = self._make_decision(
            total_evaluated, total_passed,
            best.entry_mode, best.direction,
            best.score, "SIGNAL_SUBMITTED",
            signal_id=best.entry_signal.signal_id,
            tick_interval=self._current_tick_interval,
            via_retracement=best.via_retracement,
            meta={
                "rr": best.entry_signal.risk_reward,
                "rr_score": round(best.rr_score, 3),
                "mode_weight": round(best.mode_weight, 3),
                "direction_bonus": round(best.direction_bonus, 3),
                "retrace_boost": round(best.retracement_boost, 3),
                "retrace_zone": retrace_measure.zone.value,
                "retrace_pct": round(retrace_measure.retrace_pct * 100, 1),
                "retrace_quality": round(retrace_measure.quality, 3),
                "all_candidates": [
                    {"mode": c.entry_mode, "dir": c.direction, "score": round(c.score, 3)}
                    for c in candidates[:5]
                ],
            },
        )
        self._update_tick_interval(atr, current_price, wave_analysis, found=True)
        return best, dec

    def _build_retracement_candidate(
        self,
        rm: RetracementMeasure,
        symbol: str,
        lot_size: float,
        atr: float,
        pip_size: float,
        wave_analysis: WaveAnalysis,
        main_wave: str,
    ) -> Optional[ScoredCandidate]:
        """
        Xây dựng ScoredCandidate từ RetracementMeasure.
        Dùng safest_entry / safest_sl / safest_tp từ RetracementEngine.
        """
        direction = rm.main_direction  # "BUY" or "SELL"

        # Kiểm tra SL/TP hợp lệ
        sl_dist = abs(rm.safest_entry - rm.safest_sl)
        tp_dist = abs(rm.safest_tp - rm.safest_entry)
        if sl_dist < pip_size * 5 or tp_dist < pip_size * 5:
            return None

        # Tính R:R
        rr = tp_dist / sl_dist if sl_dist > 0 else 0.0
        rr_s = min(rr / _PERFECT_RR, 1.0)

        # Direction bonus (LTF EMA cùng hướng)
        db = 0.0
        if direction == "BUY" and wave_analysis.ltf_ema_fast > wave_analysis.ltf_ema_slow:
            db = 0.1
        elif direction == "SELL" and wave_analysis.ltf_ema_fast < wave_analysis.ltf_ema_slow:
            db = 0.1

        # Bounce bonus
        bounce_bonus = 0.15 if rm.bounce_detected else 0.0

        # Retracement score formula
        score = rm.quality * rr_s * 1.2 + db + bounce_bonus
        score = round(min(max(score, 0.0), 1.0), 4)

        if score < self.min_score:
            return None

        # Build EntrySignal using safest values from RetracementEngine
        from .entry_logic import EntrySignal
        sig = EntrySignal(
            signal_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction=direction,
            entry_price=rm.safest_entry,
            sl=rm.safest_sl,
            tp=rm.safest_tp,
            lot_size=lot_size,
            entry_mode="RETRACEMENT",
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            atr=atr,
            meta={
                "fib": rm.nearest_fib,
                "zone": rm.zone.value,
                "retrace_pct": round(rm.retrace_pct * 100, 1),
                "sr_strength": rm.nearest_sr.strength if rm.nearest_sr else 0.0,
                "tp_extension": rm.tp_extension,
            },
        )

        return ScoredCandidate(
            entry_signal=sig,
            entry_mode="RETRACEMENT",
            direction=direction,
            score=score,
            wave_conf=wave_analysis.confidence,
            rr_score=rr_s,
            mode_weight=1.2,
            direction_bonus=db,
            retracement_boost=bounce_bonus,
            via_retracement=True,
        )

    def score_to_priority(self, score: float) -> int:
        """
        Chuyển điểm sang coordinator priority (0–10).
        Priority cao hơn → được xử lý trước trong queue.
        """
        if score >= 0.85:
            return 10
        if score >= 0.70:
            return 8
        if score >= 0.55:
            return 6
        if score >= 0.40:
            return 4
        if score >= 0.30:
            return 2
        return 1

    def get_current_tick_interval(self) -> float:
        """Trả về tick interval hiện tại (giây) — tự điều chỉnh theo volatility."""
        return self._current_tick_interval

    @property
    def last_decision(self) -> Optional[AutoPilotDecision]:
        return self._last_decision

    @property
    def history(self) -> List[AutoPilotDecision]:
        """20 quyết định gần nhất (newest first)."""
        return list(reversed(self._history[-20:]))

    @property
    def decisions_total(self) -> int:
        return len(self._history)

    @property
    def signals_generated(self) -> int:
        return sum(1 for d in self._history if d.action == "SIGNAL_SUBMITTED")

    # ── Internal helpers ───────────────────────────────────────────────── #

    @staticmethod
    def _wave_allows(direction: str, wa: WaveAnalysis) -> bool:
        """Kiểm tra hướng có khớp sóng chính không."""
        if wa.main_wave == WaveState.SIDEWAYS:
            return False
        if wa.sub_wave is not None:
            return False
        d = direction.upper()
        if d in ("BUY", "LONG"):
            return wa.main_wave == WaveState.BULL_MAIN
        if d in ("SELL", "SHORT"):
            return wa.main_wave == WaveState.BEAR_MAIN
        return False

    def _score(
        self,
        sig: EntrySignal,
        wave_conf: float,
        mode_name: str,
        main_wave: str,
        wa: WaveAnalysis,
        mode_weight_multipliers: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Trả về (total_score, rr_score, mode_weight, direction_bonus).

        mode_weight_multipliers: từ DecisionEngine.adaptive — hệ số tự học
          applied as: effective_mw = base_mw × multiplier
        """
        # R:R score
        rr_s = min(sig.risk_reward / _PERFECT_RR, 1.0) if sig.risk_reward > 0 else 0.0

        # Mode suitability (base)
        mode_dict = _MODE_WAVE_WEIGHT.get(mode_name, {})
        mw = mode_dict.get(main_wave, 0.7)

        # Apply adaptive multiplier from DecisionEngine (tự học)
        if mode_weight_multipliers:
            seg_key  = f"{mode_name}/{main_wave}"
            mw = round(mw * mode_weight_multipliers.get(seg_key, 1.0), 4)
            mw = min(max(mw, 0.1), 1.5)   # clamp

        # Direction bonus: LTF EMA cũng cùng hướng?
        db = 0.0
        if sig.direction.upper() in ("BUY", "LONG"):
            if wa.ltf_ema_fast > wa.ltf_ema_slow:
                db = 0.1
        else:
            if wa.ltf_ema_fast < wa.ltf_ema_slow:
                db = 0.1

        total = wave_conf * rr_s * mw + db
        total = round(min(max(total, 0.0), 1.0), 4)
        return total, rr_s, mw, db

    def _update_tick_interval(
        self,
        atr: float,
        price: float,
        wa: WaveAnalysis,
        found: bool,
    ) -> None:
        """
        Tự điều chỉnh tick interval:
          - ATR cao (volatile)  → tick nhanh hơn
          - Thị trường sideways → tick chậm hơn
          - Có setup tốt        → giữ tốc độ hiện tại
        """
        if atr <= 0 or price <= 0:
            return

        norm_atr = atr / price   # normalized ATR

        if wa.main_wave == WaveState.SIDEWAYS:
            target = _TICK_MAX
        elif norm_atr > 0.002:   # ATR > 0.2% → very volatile
            target = _TICK_MIN
        elif norm_atr > 0.001:   # ATR 0.1–0.2%
            target = 4.0
        else:
            target = 7.0

        # Smooth adjustment (không nhảy đột ngột)
        self._current_tick_interval = round(
            0.7 * self._current_tick_interval + 0.3 * target, 1
        )

    def _make_decision(
        self,
        evaluated: int,
        passed: int,
        mode: Optional[str],
        direction: Optional[str],
        score: float,
        action: str,
        signal_id: Optional[str] = None,
        tick_interval: float = 5.0,
        via_retracement: bool = False,
        meta: Optional[Dict] = None,
    ) -> AutoPilotDecision:
        dec = AutoPilotDecision(
            timestamp=time.time(),
            candidates_evaluated=evaluated,
            candidates_passed=passed,
            best_mode=mode,
            best_direction=direction,
            best_score=round(score, 4),
            action=action,
            signal_id=signal_id,
            tick_interval=tick_interval,
            via_retracement=via_retracement,
            meta=meta or {},
        )
        self._history.append(dec)
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]
        self._last_decision = dec
        return dec

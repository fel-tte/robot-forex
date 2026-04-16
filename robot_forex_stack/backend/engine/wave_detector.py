"""
Wave Detector — Multi-timeframe EMA trend analysis with fractal swing detection.

Logic:
  1. HTF EMA cross determines MAIN WAVE direction (BULL / BEAR / SIDEWAYS).
  2. LTF structure (HH/HL vs LL/LH) confirms or flags a SUB_WAVE correction.
  3. Sideways: price oscillating within N * ATR for X consecutive candles.
  4. can_trade(direction) → True only when direction matches main wave
     AND we are NOT currently in a sub-wave correction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class WaveState(str, Enum):
    BULL_MAIN = "BULL_MAIN"
    BEAR_MAIN = "BEAR_MAIN"
    SIDEWAYS = "SIDEWAYS"
    SUB_WAVE_UP = "SUB_WAVE_UP"
    SUB_WAVE_DOWN = "SUB_WAVE_DOWN"


@dataclass
class SwingPoint:
    index: int
    price: float
    is_high: bool


@dataclass
class WaveAnalysis:
    main_wave: WaveState
    sub_wave: Optional[WaveState]
    confidence: float          # 0.0 – 1.0
    htf_ema_fast: float
    htf_ema_slow: float
    ltf_ema_fast: float
    ltf_ema_slow: float
    atr: float
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    sideways_detected: bool = False
    description: str = ""


# ── LSTMWaveClassifier ────────────────────────────────────────────────────── #

class LSTMWaveClassifier:
    """
    LSTM-based wave direction classifier (PyTorch).

    Self-labelling training pipeline
    ---------------------------------
    1. On each ``analyse()`` call, ``record_sample(df, rule_wave_state)``
       appends the last *_SEQ_LEN* normalised close prices together with
       the rule-derived label to an internal buffer.
    2. Once *_MIN_SAMPLES* labeled windows are collected (and every
       *_RETRAIN_EVERY* new samples afterwards), ``fit_if_ready()`` trains
       a small single-layer LSTM for 30 epochs.
    3. ``predict_proba(df)`` returns P(BULL | BEAR | SIDEWAYS); the caller
       blends this with the rule-based confidence score.

    Graceful degradation
    --------------------
    If PyTorch is not installed the classifier silently stays inactive
    (``is_ready == False``) and the system falls back to the purely
    rule-based confidence.
    """

    _SEQ_LEN:      int = 30   # lookback window (close price bars)
    _HIDDEN:       int = 32   # LSTM hidden units
    _MIN_SAMPLES:  int = 50   # minimum labeled windows before first training
    _RETRAIN_EVERY: int = 25  # retrain every N new samples
    _EPOCHS:       int = 30   # training epochs per fit

    # Label mapping: WaveState value → class index {0=BULL, 1=BEAR, 2=SIDEWAYS}
    _LABEL_MAP: Dict[str, int] = {
        "BULL_MAIN":    0,
        "SUB_WAVE_UP":  0,
        "BEAR_MAIN":    1,
        "SUB_WAVE_DOWN": 1,
        "SIDEWAYS":     2,
    }

    def __init__(self) -> None:
        self._model: object = None
        self._is_ready = False
        self._buffer: List[Tuple[np.ndarray, int]] = []  # (seq, label)
        self._records_since_retrain: int = 0

        try:
            import torch as _torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            logger.info(
                "LSTMWaveClassifier: torch not installed — LSTM disabled"
            )

    # ── Public API ──────────────────────────────────────────────────────── #

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def record_sample(self, df: pd.DataFrame, wave_state: WaveState) -> None:
        """Collect one labeled training window from a rule-based analysis."""
        if not self._torch_available:
            return
        seq = self._extract_sequence(df)
        if seq is None:
            return
        label = self._LABEL_MAP.get(wave_state.value, 2)
        self._buffer.append((seq, label))
        self._records_since_retrain += 1

    def fit_if_ready(self) -> None:
        """Train LSTM when buffer has enough samples and enough new arrivals."""
        if not self._torch_available:
            return
        if len(self._buffer) < self._MIN_SAMPLES:
            return
        if self._records_since_retrain < self._RETRAIN_EVERY and self._is_ready:
            return
        self._train()

    def predict_proba(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Return ``{'BULL': p, 'BEAR': p, 'SIDEWAYS': p}`` or ``None``.
        The three probabilities sum to 1.0.
        """
        if not self._is_ready or self._model is None:
            return None
        seq = self._extract_sequence(df)
        if seq is None:
            return None
        try:
            import torch  # noqa: PLC0415
            import torch.nn.functional as F  # noqa: PLC0415

            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            self._model.eval()  # type: ignore[union-attr]
            with torch.no_grad():
                logits = self._model(x)  # type: ignore[operator]
                probs  = F.softmax(logits, dim=-1).squeeze(0).numpy()
            return {
                "BULL":     round(float(probs[0]), 4),
                "BEAR":     round(float(probs[1]), 4),
                "SIDEWAYS": round(float(probs[2]), 4),
            }
        except Exception as exc:  # noqa: BLE001
            logger.debug("LSTMWaveClassifier.predict_proba: %s", exc)
            return None

    # ── Internal helpers ─────────────────────────────────────────────────── #

    def _extract_sequence(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Return the last *_SEQ_LEN* normalised close prices, or None."""
        if len(df) < self._SEQ_LEN:
            return None
        closes = df["close"].iloc[-self._SEQ_LEN:].values.astype(np.float32)
        mu, sigma = float(closes.mean()), float(closes.std())
        if sigma < 1e-9:
            return None
        return (closes - mu) / sigma

    def _train(self) -> None:
        """Train a small single-layer LSTM on the current buffer."""
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415

        class _LSTMNet(nn.Module):
            def __init__(self, hidden: int = 32) -> None:
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, batch_first=True)
                self.fc   = nn.Linear(hidden, 3)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                _, (hn, _) = self.lstm(x)
                return self.fc(hn.squeeze(0))

        seqs   = torch.tensor(
            np.array([s for s, _ in self._buffer]), dtype=torch.float32
        ).unsqueeze(-1)  # (N, SEQ_LEN, 1)
        labels = torch.tensor(
            [lb for _, lb in self._buffer], dtype=torch.long
        )

        model   = _LSTMNet(hidden=self._HIDDEN)
        opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        last_loss = 0.0
        for _ in range(self._EPOCHS):
            opt.zero_grad()
            out      = model(seqs)
            loss     = loss_fn(out, labels)
            loss.backward()
            opt.step()
            last_loss = float(loss)

        self._model    = model
        self._is_ready = True
        self._records_since_retrain = 0
        logger.info(
            "LSTMWaveClassifier: trained on %d samples (loss=%.4f)",
            len(self._buffer), last_loss,
        )


class WaveDetector:
    """
    Parameters
    ----------
    htf_ema_fast : int   Higher-TF fast EMA period (default 21)
    htf_ema_slow : int   Higher-TF slow EMA period (default 50)
    ltf_ema_fast : int   Lower-TF fast EMA period  (default 8)
    ltf_ema_slow : int   Lower-TF slow EMA period  (default 21)
    fractal_period : int Fractal lookback left + right bars (default 2)
    sideways_atr_mult : float  Price range / ATR threshold for sideways (default 1.5)
    sideways_candles : int     Min candles within range to call SIDEWAYS (default 10)
    atr_period : int    ATR smoothing period (default 14)
    """

    def __init__(
        self,
        htf_ema_fast: int = 21,
        htf_ema_slow: int = 50,
        ltf_ema_fast: int = 8,
        ltf_ema_slow: int = 21,
        fractal_period: int = 2,
        sideways_atr_mult: float = 1.5,
        sideways_candles: int = 10,
        atr_period: int = 14,
    ) -> None:
        self.htf_ema_fast = htf_ema_fast
        self.htf_ema_slow = htf_ema_slow
        self.ltf_ema_fast = ltf_ema_fast
        self.ltf_ema_slow = ltf_ema_slow
        self.fractal_period = fractal_period
        self.sideways_atr_mult = sideways_atr_mult
        self.sideways_candles = sideways_candles
        self.atr_period = atr_period

        self._last_analysis: Optional[WaveAnalysis] = None
        self._lstm = LSTMWaveClassifier()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyse(self, df: pd.DataFrame) -> WaveAnalysis:
        """
        df must have columns: open, high, low, close, volume
        Returns a WaveAnalysis with current wave state.
        """
        df = df.copy().reset_index(drop=True)
        if len(df) < self.htf_ema_slow + 5:
            return self._flat_analysis(df)

        atr = self._calc_atr(df)

        # EMA calculations
        htf_fast = self._ema(df["close"], self.htf_ema_fast)
        htf_slow = self._ema(df["close"], self.htf_ema_slow)
        ltf_fast = self._ema(df["close"], self.ltf_ema_fast)
        ltf_slow = self._ema(df["close"], self.ltf_ema_slow)

        # Fractals
        swing_highs, swing_lows = self._calc_fractals(df)

        # ---- MAIN WAVE ----
        main_wave, htf_conf = self._determine_main_wave(
            df, htf_fast, htf_slow, swing_highs, swing_lows, atr
        )

        # ---- SIDEWAYS ----
        sideways = self._detect_sideways(df, atr)
        if sideways:
            main_wave = WaveState.SIDEWAYS

        # ---- SUB WAVE ----
        sub_wave = None
        if main_wave in (WaveState.BULL_MAIN, WaveState.BEAR_MAIN):
            sub_wave = self._detect_sub_wave(
                df, ltf_fast, ltf_slow, main_wave, swing_highs, swing_lows
            )

        # ---- Confidence ----
        confidence = self._calc_confidence(
            df, htf_fast, htf_slow, ltf_fast, ltf_slow, main_wave, sub_wave, sideways
        )

        description = self._build_description(main_wave, sub_wave, sideways, confidence)

        analysis = WaveAnalysis(
            main_wave=main_wave,
            sub_wave=sub_wave,
            confidence=confidence,
            htf_ema_fast=float(htf_fast.iloc[-1]),
            htf_ema_slow=float(htf_slow.iloc[-1]),
            ltf_ema_fast=float(ltf_fast.iloc[-1]),
            ltf_ema_slow=float(ltf_slow.iloc[-1]),
            atr=float(atr),
            swing_highs=swing_highs[-5:],
            swing_lows=swing_lows[-5:],
            sideways_detected=sideways,
            description=description,
        )

        # ── LSTM: collect sample + optionally boost confidence ───────── #
        self._lstm.record_sample(df, main_wave)
        self._lstm.fit_if_ready()
        lstm_probs = self._lstm.predict_proba(df)
        if lstm_probs is not None:
            # Map current main_wave to the LSTM class
            wave_key = "SIDEWAYS" if sideways else (
                "BULL" if main_wave == WaveState.BULL_MAIN else
                "BEAR" if main_wave == WaveState.BEAR_MAIN else "SIDEWAYS"
            )
            lstm_conf = lstm_probs.get(wave_key, 0.5)
            # Blend: 60% rule-based, 40% LSTM
            blended_confidence = round(0.6 * confidence + 0.4 * lstm_conf, 3)
            analysis = WaveAnalysis(
                main_wave=analysis.main_wave,
                sub_wave=analysis.sub_wave,
                confidence=blended_confidence,
                htf_ema_fast=analysis.htf_ema_fast,
                htf_ema_slow=analysis.htf_ema_slow,
                ltf_ema_fast=analysis.ltf_ema_fast,
                ltf_ema_slow=analysis.ltf_ema_slow,
                atr=analysis.atr,
                swing_highs=analysis.swing_highs,
                swing_lows=analysis.swing_lows,
                sideways_detected=analysis.sideways_detected,
                description=self._build_description(
                    main_wave, sub_wave, sideways, blended_confidence
                ),
            )

        self._last_analysis = analysis
        return analysis

    def can_trade(self, direction: str, analysis: Optional[WaveAnalysis] = None) -> bool:
        """
        Returns True only when:
          - direction matches the main wave
          - we are NOT in a sub-wave correction
          - market is not SIDEWAYS
        """
        wa = analysis or self._last_analysis
        if wa is None:
            return False

        if wa.main_wave == WaveState.SIDEWAYS:
            return False

        if wa.sub_wave in (WaveState.SUB_WAVE_UP, WaveState.SUB_WAVE_DOWN):
            return False

        direction_upper = direction.upper()
        if direction_upper in ("BUY", "LONG", "BULL"):
            return wa.main_wave == WaveState.BULL_MAIN
        if direction_upper in ("SELL", "SHORT", "BEAR"):
            return wa.main_wave == WaveState.BEAR_MAIN
        return False

    @property
    def last_analysis(self) -> Optional[WaveAnalysis]:
        return self._last_analysis

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _calc_atr(self, df: pd.DataFrame) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return float(tr.rolling(self.atr_period).mean().iloc[-1])

    def _calc_fractals(
        self, df: pd.DataFrame
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        fp = self.fractal_period
        highs: List[SwingPoint] = []
        lows: List[SwingPoint] = []
        n = len(df)
        for i in range(fp, n - fp):
            window_h = df["high"].iloc[i - fp : i + fp + 1]
            if df["high"].iloc[i] == window_h.max():
                highs.append(SwingPoint(i, float(df["high"].iloc[i]), True))
            window_l = df["low"].iloc[i - fp : i + fp + 1]
            if df["low"].iloc[i] == window_l.min():
                lows.append(SwingPoint(i, float(df["low"].iloc[i]), False))
        return highs, lows

    def _determine_main_wave(
        self,
        df: pd.DataFrame,
        htf_fast: pd.Series,
        htf_slow: pd.Series,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        atr: float,
    ) -> Tuple[WaveState, float]:
        fast_last = float(htf_fast.iloc[-1])
        slow_last = float(htf_slow.iloc[-1])
        fast_prev = float(htf_fast.iloc[-2])
        slow_prev = float(htf_slow.iloc[-2])

        ema_bull = fast_last > slow_last
        ema_bear = fast_last < slow_last
        # Check if EMA cross happened recently (within last 3 bars)
        cross_up = fast_last > slow_last and fast_prev <= slow_prev
        cross_dn = fast_last < slow_last and fast_prev >= slow_prev

        # Structure analysis (HH/HL vs LH/LL)
        struct_bull = self._is_bullish_structure(swing_highs, swing_lows)
        struct_bear = self._is_bearish_structure(swing_highs, swing_lows)

        if ema_bull and struct_bull:
            conf = 0.9 if cross_up else 0.75
            return WaveState.BULL_MAIN, conf
        if ema_bear and struct_bear:
            conf = 0.9 if cross_dn else 0.75
            return WaveState.BEAR_MAIN, conf
        if ema_bull:
            return WaveState.BULL_MAIN, 0.55
        if ema_bear:
            return WaveState.BEAR_MAIN, 0.55
        return WaveState.SIDEWAYS, 0.4

    @staticmethod
    def _is_bullish_structure(
        highs: List[SwingPoint], lows: List[SwingPoint]
    ) -> bool:
        if len(highs) < 2 or len(lows) < 2:
            return False
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        return hh and hl

    @staticmethod
    def _is_bearish_structure(
        highs: List[SwingPoint], lows: List[SwingPoint]
    ) -> bool:
        if len(highs) < 2 or len(lows) < 2:
            return False
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price
        return lh and ll

    def _detect_sideways(self, df: pd.DataFrame, atr: float) -> bool:
        if atr <= 0:
            return False
        window = df["close"].iloc[-self.sideways_candles :]
        price_range = float(window.max() - window.min())
        return price_range < self.sideways_atr_mult * atr

    def _detect_sub_wave(
        self,
        df: pd.DataFrame,
        ltf_fast: pd.Series,
        ltf_slow: pd.Series,
        main_wave: WaveState,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> Optional[WaveState]:
        """Detect counter-trend sub-wave using LTF EMA + structure."""
        lf = float(ltf_fast.iloc[-1])
        ls = float(ltf_slow.iloc[-1])

        if main_wave == WaveState.BULL_MAIN:
            # In a bull trend, a sub-wave is a bearish correction
            if lf < ls:
                # Confirm with structure: lower high
                if len(swing_highs) >= 2 and swing_highs[-1].price < swing_highs[-2].price:
                    return WaveState.SUB_WAVE_DOWN
        elif main_wave == WaveState.BEAR_MAIN:
            # In a bear trend, a sub-wave is a bullish correction
            if lf > ls:
                if len(swing_lows) >= 2 and swing_lows[-1].price > swing_lows[-2].price:
                    return WaveState.SUB_WAVE_UP
        return None

    def _calc_confidence(
        self,
        df: pd.DataFrame,
        htf_fast: pd.Series,
        htf_slow: pd.Series,
        ltf_fast: pd.Series,
        ltf_slow: pd.Series,
        main_wave: WaveState,
        sub_wave: Optional[WaveState],
        sideways: bool,
    ) -> float:
        if sideways:
            return 0.3
        score = 0.0
        # EMA separation
        ema_sep = abs(float(htf_fast.iloc[-1]) - float(htf_slow.iloc[-1]))
        price = float(df["close"].iloc[-1])
        if price > 0:
            score += min(ema_sep / price * 1000, 0.4)  # max 0.4 from separation

        # Trend momentum (last 5 bars all same direction)
        closes = df["close"].iloc[-5:].values
        diffs = np.diff(closes)
        if main_wave == WaveState.BULL_MAIN and np.all(diffs > 0):
            score += 0.3
        elif main_wave == WaveState.BEAR_MAIN and np.all(diffs < 0):
            score += 0.3
        else:
            score += 0.15

        # LTF alignment
        lf = float(ltf_fast.iloc[-1])
        ls = float(ltf_slow.iloc[-1])
        if main_wave == WaveState.BULL_MAIN and lf > ls:
            score += 0.3
        elif main_wave == WaveState.BEAR_MAIN and lf < ls:
            score += 0.3
        else:
            score += 0.1

        # Sub-wave penalty
        if sub_wave is not None:
            score *= 0.6

        return round(min(max(score, 0.0), 1.0), 3)

    @staticmethod
    def _build_description(
        main_wave: WaveState,
        sub_wave: Optional[WaveState],
        sideways: bool,
        confidence: float,
    ) -> str:
        parts = [f"Main Wave: {main_wave.value}"]
        if sideways:
            parts.append("(Sideways detected)")
        if sub_wave:
            parts.append(f"| Sub-Wave: {sub_wave.value} — trading paused")
        parts.append(f"| Confidence: {confidence:.0%}")
        return " ".join(parts)

    def _flat_analysis(self, df: pd.DataFrame) -> WaveAnalysis:
        close = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
        return WaveAnalysis(
            main_wave=WaveState.SIDEWAYS,
            sub_wave=None,
            confidence=0.0,
            htf_ema_fast=close,
            htf_ema_slow=close,
            ltf_ema_fast=close,
            ltf_ema_slow=close,
            atr=0.0,
            description="Insufficient data",
        )

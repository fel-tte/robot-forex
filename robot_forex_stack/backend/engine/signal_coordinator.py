"""
Signal Coordinator — Queue-based order authority system.

Flow: submit_signal → authority check → load limit check → queue → validate → execute

States
------
IDLE        : Robot is on but no setup yet
MONITORING  : Watching for setups
COOLDOWN    : After a loss, locked for N minutes
RESTRICTED  : Max trades / daily limit reached
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class SignalAuthority(str, Enum):
    BLOCKED = "BLOCKED"
    RESTRICTED = "RESTRICTED"
    NORMAL = "NORMAL"
    PRIORITY = "PRIORITY"


class CoordinatorState(str, Enum):
    IDLE = "IDLE"
    MONITORING = "MONITORING"
    COOLDOWN = "COOLDOWN"
    RESTRICTED = "RESTRICTED"


@dataclass
class TradeSignal:
    signal_id: str
    symbol: str
    direction: str           # BUY / SELL
    entry_price: float
    sl: float
    tp: float
    lot_size: float
    entry_mode: str
    priority: int = 0        # higher = more important
    timestamp: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalRecord:
    signal: TradeSignal
    status: str              # QUEUED / EXECUTED / REJECTED / EXPIRED
    reject_reason: str = ""
    processed_at: Optional[float] = None


@dataclass
class QueueMetrics:
    signals_queued: int = 0
    signals_executed: int = 0
    signals_rejected: int = 0
    signals_expired: int = 0
    queue_depth: int = 0
    cooldown_until: float = 0.0
    state: CoordinatorState = CoordinatorState.IDLE
    authority: SignalAuthority = SignalAuthority.NORMAL


class SignalCoordinator:
    """
    Thread-safe (asyncio) coordinator.

    Parameters
    ----------
    max_queue_size : int  Hard limit on pending signals (load limit)
    max_concurrent_trades : int
    cooldown_minutes : float  After a losing trade, pause for this many minutes
    signal_expiry_seconds : float  Signals older than this are discarded
    """

    def __init__(
        self,
        max_queue_size: int = 10,
        max_concurrent_trades: int = 3,
        cooldown_minutes: float = 5.0,
        signal_expiry_seconds: float = 300.0,
    ) -> None:
        self.max_queue_size = max_queue_size
        self.max_concurrent_trades = max_concurrent_trades
        self.cooldown_minutes = cooldown_minutes
        self.signal_expiry_seconds = signal_expiry_seconds

        self._queue: asyncio.Queue[TradeSignal] = asyncio.Queue(maxsize=max_queue_size)
        self._state: CoordinatorState = CoordinatorState.IDLE
        self._authority: SignalAuthority = SignalAuthority.NORMAL
        self._metrics = QueueMetrics()
        self._history: List[SignalRecord] = []
        self._open_trade_count: int = 0
        self._on_execute: Optional[Callable[[TradeSignal], Coroutine]] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_execute_callback(self, cb: Callable[[TradeSignal], Coroutine]) -> None:
        """Register async callback invoked when a signal is approved for execution."""
        self._on_execute = cb

    def start(self) -> None:
        self._state = CoordinatorState.MONITORING
        self._metrics.state = self._state
        logger.info("SignalCoordinator started — state: MONITORING")

    def stop(self) -> None:
        self._state = CoordinatorState.IDLE
        self._metrics.state = self._state
        logger.info("SignalCoordinator stopped")

    def submit_signal(self, signal: TradeSignal) -> str:
        """
        Attempt to enqueue a signal.
        Returns: 'QUEUED' | 'REJECTED:<reason>'
        """
        reject = self._pre_check(signal)
        if reject:
            self._record(signal, "REJECTED", reject)
            self._metrics.signals_rejected += 1
            logger.debug("Signal %s rejected: %s", signal.signal_id, reject)
            return f"REJECTED:{reject}"

        try:
            self._queue.put_nowait(signal)
            self._record(signal, "QUEUED")
            self._metrics.signals_queued += 1
            self._metrics.queue_depth = self._queue.qsize()
            logger.info(
                "Signal %s queued [%s %s @%.5f]",
                signal.signal_id,
                signal.direction,
                signal.symbol,
                signal.entry_price,
            )
            return "QUEUED"
        except asyncio.QueueFull:
            self._record(signal, "REJECTED", "QUEUE_FULL")
            self._metrics.signals_rejected += 1
            return "REJECTED:QUEUE_FULL"

    async def process_next(
        self,
        wave_can_trade: Callable[[str], bool],
    ) -> Optional[str]:
        """
        Dequeue one signal, validate against current wave state, execute.
        Returns signal_id if executed, None otherwise.
        """
        if self._queue.empty():
            return None

        signal: TradeSignal = await self._queue.get()
        self._metrics.queue_depth = self._queue.qsize()

        # Expiry check
        age = time.time() - signal.timestamp
        if age > self.signal_expiry_seconds:
            self._update_record(signal.signal_id, "EXPIRED")
            self._metrics.signals_expired += 1
            logger.debug("Signal %s expired (age=%.1fs)", signal.signal_id, age)
            return None

        # Re-check cooldown (could have changed since submit)
        if self._in_cooldown():
            self._update_record(signal.signal_id, "REJECTED", "COOLDOWN_ACTIVE")
            self._metrics.signals_rejected += 1
            return None

        # Wave alignment check
        if not wave_can_trade(signal.direction):
            self._update_record(signal.signal_id, "REJECTED", "WAVE_MISALIGNED")
            self._metrics.signals_rejected += 1
            logger.info(
                "Signal %s rejected — wave not aligned for %s",
                signal.signal_id,
                signal.direction,
            )
            return None

        # Concurrent trades check
        if self._open_trade_count >= self.max_concurrent_trades:
            self._update_record(signal.signal_id, "REJECTED", "MAX_CONCURRENT")
            self._metrics.signals_rejected += 1
            return None

        # Execute
        self._update_record(signal.signal_id, "EXECUTED")
        self._metrics.signals_executed += 1
        self._open_trade_count += 1
        logger.info(
            "Executing signal %s: %s %s @%.5f  lot=%.2f",
            signal.signal_id,
            signal.direction,
            signal.symbol,
            signal.entry_price,
            signal.lot_size,
        )
        if self._on_execute:
            await self._on_execute(signal)
        return signal.signal_id

    async def process_all(self, wave_can_trade: Callable[[str], bool]) -> List[str]:
        """Drain the entire queue, returns list of executed signal IDs."""
        executed = []
        while not self._queue.empty():
            sid = await self.process_next(wave_can_trade)
            if sid:
                executed.append(sid)
        return executed

    def on_trade_closed(self, profit: float) -> None:
        """Called when a trade closes. Triggers cooldown on loss."""
        self._open_trade_count = max(0, self._open_trade_count - 1)
        if profit < 0:
            self.trigger_cooldown()

    def trigger_cooldown(self) -> None:
        self._state = CoordinatorState.COOLDOWN
        self._metrics.cooldown_until = time.time() + self.cooldown_minutes * 60
        self._metrics.state = self._state
        logger.info(
            "Cooldown triggered for %.1f minutes", self.cooldown_minutes
        )

    def update_authority(self, authority: SignalAuthority) -> None:
        self._authority = authority
        self._metrics.authority = authority
        logger.debug("Authority updated to %s", authority.value)

    def set_state(self, state: CoordinatorState) -> None:
        self._state = state
        self._metrics.state = state

    @property
    def state(self) -> CoordinatorState:
        self._refresh_cooldown()
        return self._state

    @property
    def metrics(self) -> QueueMetrics:
        self._refresh_cooldown()
        m = QueueMetrics(
            signals_queued=self._metrics.signals_queued,
            signals_executed=self._metrics.signals_executed,
            signals_rejected=self._metrics.signals_rejected,
            signals_expired=self._metrics.signals_expired,
            queue_depth=self._queue.qsize(),
            cooldown_until=self._metrics.cooldown_until,
            state=self._state,
            authority=self._authority,
        )
        return m

    @property
    def history(self) -> List[SignalRecord]:
        return list(reversed(self._history[-100:]))  # last 100, newest first

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pre_check(self, signal: TradeSignal) -> Optional[str]:
        """Returns rejection reason string, or None if OK."""
        if self._authority == SignalAuthority.BLOCKED:
            return "AUTHORITY_BLOCKED"
        if self._in_cooldown():
            return "COOLDOWN_ACTIVE"
        if self._state == CoordinatorState.IDLE:
            return "COORDINATOR_IDLE"
        if self._state == CoordinatorState.RESTRICTED:
            if signal.priority < 5:
                return "RESTRICTED_LOW_PRIORITY"
        if self._authority == SignalAuthority.RESTRICTED and signal.priority < 3:
            return "AUTHORITY_RESTRICTED"
        return None

    def _in_cooldown(self) -> bool:
        if self._state == CoordinatorState.COOLDOWN:
            if time.time() >= self._metrics.cooldown_until:
                self._refresh_cooldown()
                return False
            return True
        return False

    def _refresh_cooldown(self) -> None:
        if (
            self._state == CoordinatorState.COOLDOWN
            and time.time() >= self._metrics.cooldown_until
        ):
            self._state = CoordinatorState.MONITORING
            self._metrics.state = self._state
            logger.info("Cooldown expired, resuming MONITORING")

    def _record(self, signal: TradeSignal, status: str, reason: str = "") -> None:
        rec = SignalRecord(signal=signal, status=status, reject_reason=reason)
        self._history.append(rec)
        if len(self._history) > 500:
            self._history = self._history[-500:]

    def _update_record(self, signal_id: str, status: str, reason: str = "") -> None:
        for rec in reversed(self._history):
            if rec.signal.signal_id == signal_id:
                rec.status = status
                rec.reject_reason = reason
                rec.processed_at = time.time()
                return

"""
Trade Manager — Partial close, trailing stop, grid system, trade lifecycle.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    direction: str          # BUY / SELL
    lot_size: float
    entry_price: float
    sl: float
    tp: float
    entry_mode: str
    open_time: float = field(default_factory=time.time)
    close_time: Optional[float] = None
    close_price: Optional[float] = None
    pnl: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    remaining_lots: float = 0.0
    be_moved: bool = False          # SL moved to break-even
    grid_level: int = 0
    comment: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.remaining_lots == 0.0:
            self.remaining_lots = self.lot_size

    @property
    def is_open(self) -> bool:
        return self.status == TradeStatus.OPEN

    def calculate_pnl(self, current_price: float, pip_value: float = 10.0) -> float:
        pip_size = 0.0001
        if self.direction.upper() == "BUY":
            pips = (current_price - self.entry_price) / pip_size
        else:
            pips = (self.entry_price - current_price) / pip_size
        return round(pips * pip_value * self.remaining_lots, 2)


@dataclass
class PartialCloseConfig:
    enabled: bool = False
    trigger_pct: float = 50.0     # close when TP% reached
    close_pct: float = 50.0       # % of lots to close
    move_sl_to_be: bool = True    # move SL to BE after partial


@dataclass
class TrailingConfig:
    enabled: bool = False
    mode: str = "PCT_TP"          # PCT_TP or HILO
    trigger_pct: float = 50.0     # start trailing after X% TP hit
    trail_pct: float = 30.0       # trail at X% of TP distance behind


@dataclass
class GridConfig:
    enabled: bool = False
    levels: int = 3
    distance_pips: float = 200.0
    distance_multiplier: float = 1.5
    volume_multiplier: float = 1.5
    max_grid_lot: float = 1.0


class PartialCloseManager:
    def __init__(self, config: PartialCloseConfig) -> None:
        self.config = config
        self._triggered: Dict[str, bool] = {}

    def check_and_close(self, trade: TradeRecord, current_price: float) -> Optional[float]:
        """
        Returns lots_to_close if partial close should trigger, else None.
        Modifies trade in place (remaining_lots, sl adjustment).
        """
        if not self.config.enabled:
            return None
        if trade.trade_id in self._triggered:
            return None

        tp_dist = abs(trade.tp - trade.entry_price)
        if tp_dist <= 0:
            return None

        price_dist = abs(current_price - trade.entry_price)
        pct_reached = price_dist / tp_dist * 100

        if pct_reached >= self.config.trigger_pct:
            lots_to_close = round(trade.remaining_lots * (self.config.close_pct / 100), 8)
            lots_to_close = min(lots_to_close, trade.remaining_lots)
            trade.remaining_lots = round(trade.remaining_lots - lots_to_close, 8)

            if self.config.move_sl_to_be:
                trade.sl = trade.entry_price
                trade.be_moved = True

            self._triggered[trade.trade_id] = True
            logger.info(
                "Partial close: trade %s — closed %.2f lots at %.5f (%.0f%% of TP)",
                trade.trade_id,
                lots_to_close,
                current_price,
                pct_reached,
            )
            return lots_to_close
        return None


class TrailingStopManager:
    def __init__(self, config: TrailingConfig) -> None:
        self.config = config
        self._best_price: Dict[str, float] = {}
        self._trailing_active: Dict[str, bool] = {}

    def update(self, trade: TradeRecord, current_price: float, atr: float = 0.0) -> Optional[float]:
        """
        Returns updated SL price if trailing stop should move, else None.
        """
        if not self.config.enabled:
            return None

        tid = trade.trade_id
        is_buy = trade.direction.upper() == "BUY"

        # Track best price
        if tid not in self._best_price:
            self._best_price[tid] = current_price
        else:
            if is_buy:
                self._best_price[tid] = max(self._best_price[tid], current_price)
            else:
                self._best_price[tid] = min(self._best_price[tid], current_price)

        tp_dist = abs(trade.tp - trade.entry_price)
        if tp_dist <= 0:
            return None

        # Check if trailing should activate
        price_dist = abs(current_price - trade.entry_price)
        pct_reached = price_dist / tp_dist * 100

        if pct_reached < self.config.trigger_pct:
            return None

        self._trailing_active[tid] = True

        if self.config.mode == "PCT_TP":
            trail_dist = tp_dist * (self.config.trail_pct / 100)
        else:  # HILO — trail by ATR
            trail_dist = atr if atr > 0 else tp_dist * 0.3

        best = self._best_price[tid]
        if is_buy:
            new_sl = best - trail_dist
            if new_sl > trade.sl:
                return round(new_sl, 5)
        else:
            new_sl = best + trail_dist
            if new_sl < trade.sl:
                return round(new_sl, 5)

        return None


class GridManager:
    """Calculates grid entry levels based on multipliers."""

    def __init__(self, config: GridConfig) -> None:
        self.config = config

    def get_grid_levels(
        self, base_price: float, direction: str, pip_size: float = 0.0001
    ) -> List[Dict]:
        """
        Returns list of {'price': float, 'lot': float, 'level': int}
        """
        levels = []
        is_buy = direction.upper() == "BUY"
        dist = self.config.distance_pips * pip_size
        base_lot = 0.01  # will be overridden by lot manager

        for i in range(1, self.config.levels + 1):
            level_dist = dist * (self.config.distance_multiplier ** (i - 1))
            if is_buy:
                price = base_price - level_dist
            else:
                price = base_price + level_dist
            lot = min(
                base_lot * (self.config.volume_multiplier ** (i - 1)),
                self.config.max_grid_lot,
            )
            levels.append({"price": round(price, 5), "lot": round(lot, 8), "level": i})
        return levels


class TradeManager:
    """
    Orchestrates the full trade lifecycle:
    open → partial close → trailing stop → grid → close.
    """

    def __init__(
        self,
        partial_config: Optional[PartialCloseConfig] = None,
        trailing_config: Optional[TrailingConfig] = None,
        grid_config: Optional[GridConfig] = None,
        pip_value: float = 10.0,
    ) -> None:
        self.pip_value = pip_value
        self._partial = PartialCloseManager(partial_config or PartialCloseConfig())
        self._trailing = TrailingStopManager(trailing_config or TrailingConfig())
        self._grid = GridManager(grid_config or GridConfig())
        self._trades: Dict[str, TradeRecord] = {}
        self._closed_trades: List[TradeRecord] = []

    def open_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        lot_size: float,
        entry_mode: str = "BREAKOUT",
        grid_level: int = 0,
        comment: str = "",
    ) -> TradeRecord:
        trade = TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction=direction,
            lot_size=lot_size,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            entry_mode=entry_mode,
            grid_level=grid_level,
            comment=comment,
        )
        self._trades[trade.trade_id] = trade
        logger.info(
            "Trade opened %s: %s %s lot=%.2f @%.5f SL=%.5f TP=%.5f",
            trade.trade_id, direction, symbol, lot_size, entry_price, sl, tp
        )
        return trade

    def update_trade(
        self, trade_id: str, current_price: float, atr: float = 0.0
    ) -> Dict[str, Any]:
        """Update a trade with current price, apply partial / trailing."""
        trade = self._trades.get(trade_id)
        if not trade or not trade.is_open:
            return {}

        actions = {}

        # Partial close
        closed_lots = self._partial.check_and_close(trade, current_price)
        if closed_lots is not None:
            actions["partial_close"] = closed_lots

        # Trailing stop
        new_sl = self._trailing.update(trade, current_price, atr)
        if new_sl is not None:
            trade.sl = new_sl
            actions["trailing_sl"] = new_sl

        # Check SL hit
        if self._sl_hit(trade, current_price):
            self._close_trade(trade, trade.sl, "SL")
            actions["closed"] = "SL"

        # Check TP hit
        elif self._tp_hit(trade, current_price):
            self._close_trade(trade, trade.tp, "TP")
            actions["closed"] = "TP"

        return actions

    def close_trade(self, trade_id: str, price: float, reason: str = "MANUAL") -> Optional[TradeRecord]:
        trade = self._trades.get(trade_id)
        if trade and trade.is_open:
            self._close_trade(trade, price, reason)
            return trade
        return None

    def get_open_trades(self) -> List[TradeRecord]:
        return [t for t in self._trades.values() if t.is_open]

    def get_closed_trades(self) -> List[TradeRecord]:
        return list(reversed(self._closed_trades))

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        return self._trades.get(trade_id)

    def total_pnl(self) -> float:
        return round(sum(t.pnl for t in self._closed_trades), 2)

    def win_rate(self) -> float:
        if not self._closed_trades:
            return 0.0
        wins = sum(1 for t in self._closed_trades if t.pnl > 0)
        return round(wins / len(self._closed_trades) * 100, 1)

    def profit_factor(self) -> float:
        gross_win = sum(t.pnl for t in self._closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self._closed_trades if t.pnl < 0))
        if gross_loss == 0:
            return 0.0 if gross_win == 0 else 999.9
        return round(gross_win / gross_loss, 2)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sl_hit(trade: TradeRecord, price: float) -> bool:
        if trade.direction.upper() == "BUY":
            return price <= trade.sl
        return price >= trade.sl

    @staticmethod
    def _tp_hit(trade: TradeRecord, price: float) -> bool:
        if trade.direction.upper() == "BUY":
            return price >= trade.tp
        return price <= trade.tp

    def _close_trade(self, trade: TradeRecord, price: float, reason: str) -> None:
        trade.close_price = price
        trade.close_time = time.time()
        trade.status = TradeStatus.CLOSED
        trade.pnl = trade.calculate_pnl(price, self.pip_value)
        trade.comment = f"Closed by {reason}"
        self._closed_trades.append(trade)
        if trade.trade_id in self._trades:
            del self._trades[trade.trade_id]
        logger.info(
            "Trade closed %s @%.5f | PnL=%.2f | Reason=%s",
            trade.trade_id, price, trade.pnl, reason
        )

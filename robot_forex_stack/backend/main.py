"""
Robot Forex — FastAPI Backend
Endpoints + WebSocket streaming + background RobotEngine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import (
    SessionLocal,
    create_tables,
    get_db,
    get_all_trades,
    get_trade_count,
    load_settings,
    save_settings,
    save_trade,
)
from engine import (
    WaveDetector, WaveState,
    SignalCoordinator, CoordinatorState, SignalAuthority,
    RiskManager, LotMode,
    EntryLogic, EntryMode, SLMode, TPMode,
    TradeManager,
    SessionManager, TradingSession,
    MockDataProvider,
)
from engine.signal_coordinator import TradeSignal as CoordSignal
from engine.risk_manager import RiskConfig, MartingaleConfig
from engine.trade_manager import PartialCloseConfig, TrailingConfig, GridConfig
from engine.session_manager import DSTMode
from models.schemas import (
    CandleSchema,
    PaginatedTrades,
    QueueStatusSchema,
    RiskMetricsSchema,
    RobotSettings,
    RobotStatusSchema,
    TradeRecordSchema,
    WaveAnalysisSchema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Shared application state ───────────────────────────────────────────── #

class AppState:
    def __init__(self) -> None:
        self.settings: RobotSettings = RobotSettings()
        self.robot_running: bool = False
        self.start_time: float = 0.0
        self.balance: float = 10_000.0
        self.equity: float = 10_000.0

        self.data_provider: MockDataProvider = MockDataProvider()
        self.wave_detector: WaveDetector = WaveDetector()
        self.coordinator: SignalCoordinator = SignalCoordinator()
        self.risk_manager: RiskManager = RiskManager()
        self.entry_logic: EntryLogic = EntryLogic()
        self.trade_manager: TradeManager = TradeManager()
        self.session_manager: SessionManager = SessionManager()

        self._ws_clients: Set[WebSocket] = set()
        self._engine_task: Optional[asyncio.Task] = None

    def rebuild_components(self) -> None:
        s = self.settings
        self.data_provider = MockDataProvider(symbol=s.symbol)
        self.wave_detector = WaveDetector(
            htf_ema_fast=s.htf_ema_fast,
            htf_ema_slow=s.htf_ema_slow,
            ltf_ema_fast=s.ltf_ema_fast,
            ltf_ema_slow=s.ltf_ema_slow,
            sideways_atr_mult=s.sideways_atr_mult,
            sideways_candles=s.sideways_candles,
        )
        self.coordinator = SignalCoordinator(
            max_queue_size=s.max_queue_size,
            max_concurrent_trades=s.max_trades_at_time,
            cooldown_minutes=s.cooldown_minutes,
            signal_expiry_seconds=s.signal_expiry_seconds,
        )
        self.risk_manager = RiskManager(
            config=RiskConfig(
                lot_mode=LotMode(s.lot_mode),
                lot_value=s.lot_value,
                min_lot=s.min_lot,
                max_lot=s.max_lot,
                max_account_equity=s.max_account_equity,
                max_daily_dd_pct=s.max_daily_dd_pct,
                max_overall_dd_pct=s.max_overall_dd_pct,
                pip_value_per_lot=s.pip_value_per_lot,
            ),
            martingale=MartingaleConfig(
                enabled=s.martingale.enabled,
                multiplier=s.martingale.multiplier,
                max_steps=s.martingale.max_steps,
            ),
        )
        self.entry_logic = EntryLogic(
            sl_mode=SLMode(s.sl_mode),
            sl_value=s.sl_value,
            tp_mode=TPMode(s.tp_mode),
            tp_value=s.tp_value,
            entry_mode=EntryMode(s.entry_mode),
            retrace_atr_mult=s.retrace_atr_mult,
            min_body_atr=s.min_body_atr,
            retest_level_x=s.retest_level_x,
        )
        self.trade_manager = TradeManager(
            partial_config=PartialCloseConfig(
                enabled=s.partial_close.enabled,
                trigger_pct=s.partial_close.trigger_pct,
                close_pct=s.partial_close.close_pct,
                move_sl_to_be=s.partial_close.move_sl_to_be,
            ),
            trailing_config=TrailingConfig(
                enabled=s.trailing.enabled,
                mode=s.trailing.mode,
                trigger_pct=s.trailing.trigger_pct,
                trail_pct=s.trailing.trail_pct,
            ),
            grid_config=GridConfig(
                enabled=s.grid.enabled,
                levels=s.grid.levels,
                distance_pips=s.grid.distance_pips,
                distance_multiplier=s.grid.distance_multiplier,
                volume_multiplier=s.grid.volume_multiplier,
                max_grid_lot=s.grid.max_grid_lot,
            ),
            pip_value=s.pip_value_per_lot,
        )
        self.session_manager = SessionManager(
            session=TradingSession(s.session),
            dst_mode=DSTMode(s.dst_mode),
            gmt_offset=s.gmt_offset,
        )
        self.coordinator.set_execute_callback(self._on_signal_execute)

    async def _on_signal_execute(self, signal: CoordSignal) -> None:
        """Called by coordinator when a signal is approved for execution."""
        trade = self.trade_manager.open_trade(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            sl=signal.sl,
            tp=signal.tp,
            lot_size=signal.lot_size,
            entry_mode=signal.entry_mode,
        )
        # Persist to DB
        db = SessionLocal()
        try:
            save_trade(db, {
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "direction": trade.direction,
                "lot_size": trade.lot_size,
                "entry_price": trade.entry_price,
                "sl": trade.sl,
                "tp": trade.tp,
                "entry_mode": trade.entry_mode,
                "open_time": trade.open_time,
                "close_time": None,
                "close_price": None,
                "pnl": 0.0,
                "status": "OPEN",
                "remaining_lots": trade.remaining_lots,
                "be_moved": False,
                "grid_level": 0,
                "comment": "",
                "meta": {},
            })
        finally:
            db.close()

        await self.broadcast({"event": "trade_opened", "trade": {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "lot_size": trade.lot_size,
        }})

    async def broadcast(self, data: Dict[str, Any]) -> None:
        dead = set()
        for ws in self._ws_clients:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead


app_state = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────── #

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    # Load persisted settings
    db = SessionLocal()
    try:
        stored = load_settings(db)
        if stored:
            app_state.settings = RobotSettings(**stored)
            logger.info("Settings loaded from DB")
    finally:
        db.close()
    app_state.rebuild_components()
    yield
    # Shutdown
    if app_state._engine_task:
        app_state._engine_task.cancel()


app = FastAPI(title="Robot Forex API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Robot Engine (background task) ────────────────────────────────────────#

class RobotEngine:
    """
    Async background engine loop.
    Every tick:
      1. Get latest candles from MockDataProvider
      2. Run WaveDetector
      3. If session + wave + risk allow: generate signal, submit to coordinator
      4. Coordinator processes queue
      5. Update open trades (SL/TP/trailing/partial)
      6. Persist closed trades, update equity
    """

    def __init__(self, state: AppState) -> None:
        self.state = state
        self._tick_interval = 5.0   # seconds between ticks
        self._daily_trades = 0
        self._last_day: Optional[int] = None

    async def run(self) -> None:
        logger.info("RobotEngine started")
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                logger.info("RobotEngine stopped")
                break
            except Exception as exc:
                logger.error("Engine tick error: %s", exc, exc_info=True)
            await asyncio.sleep(self._tick_interval)

    async def _tick(self) -> None:
        if not self.state.robot_running:
            return

        s = self.state.settings

        # Advance mock data
        self.state.data_provider.advance()
        df = self.state.data_provider.get_candles(limit=200)
        if len(df) < 60:
            return

        # Daily reset check
        today = int(time.time() // 86400)
        if today != self._last_day:
            self._last_day = today
            self._daily_trades = 0
            self.state.risk_manager.reset_daily(self.state.balance)

        # Wave analysis
        wave_analysis = self.state.wave_detector.analyse(df)

        # ATR
        atr = self.state.data_provider.calculate_atr(df, s.atr_period)

        # Risk update
        open_pnl = sum(
            t.calculate_pnl(df["close"].iloc[-1])
            for t in self.state.trade_manager.get_open_trades()
        )
        self.state.equity = self.state.balance + open_pnl
        self.state.risk_manager.update_equity(self.state.balance, self.state.equity)

        # Update open trades (check SL/TP/trailing/partial)
        current_price = float(df["close"].iloc[-1])
        closed_this_tick = []
        for trade in list(self.state.trade_manager.get_open_trades()):
            actions = self.state.trade_manager.update_trade(
                trade.trade_id, current_price, atr
            )
            if "closed" in actions:
                trade_obj = self.state.trade_manager.get_trade(trade.trade_id)
                # Trade was moved to closed list already
                # Find it in closed list
                for ct in self.state.trade_manager.get_closed_trades():
                    if ct.trade_id == trade.trade_id and ct.close_time:
                        closed_this_tick.append(ct)
                        break

        # Persist closed trades and update balance
        if closed_this_tick:
            db = SessionLocal()
            try:
                for ct in closed_this_tick:
                    save_trade(db, {
                        "trade_id": ct.trade_id,
                        "symbol": ct.symbol,
                        "direction": ct.direction,
                        "lot_size": ct.lot_size,
                        "entry_price": ct.entry_price,
                        "sl": ct.sl,
                        "tp": ct.tp,
                        "entry_mode": ct.entry_mode,
                        "open_time": ct.open_time,
                        "close_time": ct.close_time,
                        "close_price": ct.close_price,
                        "pnl": ct.pnl,
                        "status": "CLOSED",
                        "remaining_lots": ct.remaining_lots,
                        "be_moved": ct.be_moved,
                        "grid_level": ct.grid_level,
                        "comment": ct.comment,
                        "meta": {},
                    })
                    self.state.balance += ct.pnl
                    self.state.risk_manager.on_trade_closed(ct.pnl)
                    self.state.coordinator.on_trade_closed(ct.pnl)
            finally:
                db.close()

        # Check if we can open new trade
        daily_limit = s.max_trades_daily
        open_count = len(self.state.trade_manager.get_open_trades())
        coordinator_state = self.state.coordinator.state

        can_enter = (
            self._daily_trades < daily_limit
            and open_count < s.max_trades_at_time
            and self.state.risk_manager.is_trading_allowed(self.state.equity)
            and coordinator_state
            not in (CoordinatorState.IDLE, CoordinatorState.COOLDOWN, CoordinatorState.RESTRICTED)
            and self.state.session_manager.is_trading_time()
        )

        if can_enter:
            # Check spread filter
            spread = self.state.data_provider.get_spread_points()
            self.state.risk_manager.update_spread(s.symbol, spread)
            spread_ok = self.state.risk_manager.check_spread(s.symbol, s.max_spread)

            if spread_ok:
                await self._try_generate_signal(df, wave_analysis, atr, current_price)

        # Broadcast live update
        await self.state.broadcast({
            "event": "tick",
            "wave": wave_analysis.main_wave,
            "sub_wave": wave_analysis.sub_wave,
            "confidence": wave_analysis.confidence,
            "price": current_price,
            "equity": self.state.equity,
            "balance": self.state.balance,
            "open_trades": open_count,
            "coordinator_state": self.state.coordinator.state.value,
            "timestamp": time.time(),
        })

    async def _try_generate_signal(
        self, df, wave_analysis, atr: float, current_price: float
    ) -> None:
        s = self.state.settings
        candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # Get range high/low (last N candles as proxy for ORB)
        n_range = max(4, s.monitoring_minutes // 5)
        range_df = df.iloc[-n_range - 1 : -1]
        range_high = float(range_df["high"].max())
        range_low = float(range_df["low"].min())

        # Swing points from wave analysis
        swing_high = (
            self.state.wave_detector.last_analysis.swing_highs[-1].price
            if self.state.wave_detector.last_analysis and self.state.wave_detector.last_analysis.swing_highs
            else 0.0
        )
        swing_low = (
            self.state.wave_detector.last_analysis.swing_lows[-1].price
            if self.state.wave_detector.last_analysis and self.state.wave_detector.last_analysis.swing_lows
            else 0.0
        )

        direction = self.state.entry_logic.check_entry(
            candle, range_high, range_low, atr, float(prev_candle["close"])
        )
        if direction is None:
            return

        # EMA filter
        if s.ema_filter_enabled:
            if not self.state.wave_detector.can_trade(direction, wave_analysis):
                return

        lot_size = self.state.risk_manager.calculate_lot_size(
            self.state.balance, self.state.equity
        )

        entry_signal = self.state.entry_logic.build_entry_signal(
            signal_id=str(uuid.uuid4())[:8],
            symbol=s.symbol,
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
        )

        coord_signal = CoordSignal(
            signal_id=entry_signal.signal_id,
            symbol=entry_signal.symbol,
            direction=entry_signal.direction,
            entry_price=entry_signal.entry_price,
            sl=entry_signal.sl,
            tp=entry_signal.tp,
            lot_size=entry_signal.lot_size,
            entry_mode=entry_signal.entry_mode,
        )
        result = self.state.coordinator.submit_signal(coord_signal)
        logger.debug("Signal %s: %s", entry_signal.signal_id, result)

        if "QUEUED" in result:
            await self.state.coordinator.process_next(
                lambda d: self.state.wave_detector.can_trade(d, wave_analysis)
            )
            self._daily_trades += 1


_engine = RobotEngine(app_state)


# ── REST Endpoints ─────────────────────────────────────────────────────── #

@app.get("/api/status", response_model=RobotStatusSchema)
async def get_status():
    wa = app_state.wave_detector.last_analysis
    cm = app_state.coordinator.metrics
    tm = app_state.trade_manager
    uptime = time.time() - app_state.start_time if app_state.robot_running else 0.0
    return RobotStatusSchema(
        running=app_state.robot_running,
        state="RUNNING" if app_state.robot_running else "STOPPED",
        wave_state=wa.main_wave.value if wa else "SIDEWAYS",
        sub_wave=wa.sub_wave.value if wa and wa.sub_wave else None,
        confidence=wa.confidence if wa else 0.0,
        coordinator_state=cm.state.value,
        balance=app_state.balance,
        equity=app_state.equity,
        total_pnl=tm.total_pnl(),
        win_rate=tm.win_rate(),
        profit_factor=tm.profit_factor(),
        total_trades=len(tm.get_closed_trades()),
        open_trades=len(tm.get_open_trades()),
        daily_pnl=app_state.risk_manager.daily_pnl,
        uptime_seconds=uptime,
    )


@app.get("/api/settings", response_model=RobotSettings)
async def get_settings():
    return app_state.settings


@app.post("/api/settings", response_model=RobotSettings)
async def update_settings(settings: RobotSettings, db: Session = Depends(get_db)):
    app_state.settings = settings
    save_settings(db, settings.model_dump())
    app_state.rebuild_components()
    if app_state.robot_running:
        app_state.coordinator.start()
    return app_state.settings


@app.get("/api/trades", response_model=PaginatedTrades)
async def get_trades(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    rows = get_all_trades(db, page, page_size)
    total = get_trade_count(db)
    # Also include in-memory closed trades not yet persisted
    trades = [TradeRecordSchema(**r) for r in rows]
    return PaginatedTrades(trades=trades, total=total, page=page, page_size=page_size)


@app.get("/api/trades/open", response_model=List[TradeRecordSchema])
async def get_open_trades():
    open_trades = app_state.trade_manager.get_open_trades()
    return [TradeRecordSchema(**{
        "trade_id": t.trade_id,
        "symbol": t.symbol,
        "direction": t.direction,
        "lot_size": t.lot_size,
        "entry_price": t.entry_price,
        "sl": t.sl,
        "tp": t.tp,
        "entry_mode": t.entry_mode,
        "open_time": t.open_time,
        "close_time": None,
        "close_price": None,
        "pnl": t.calculate_pnl(
            app_state.data_provider.get_candles(limit=1)["close"].iloc[-1]
            if app_state.data_provider else t.entry_price
        ),
        "status": t.status.value,
        "remaining_lots": t.remaining_lots,
        "be_moved": t.be_moved,
        "grid_level": t.grid_level,
        "comment": t.comment,
    }) for t in open_trades]


@app.post("/api/robot/start")
async def start_robot():
    if app_state.robot_running:
        return {"status": "already_running"}
    app_state.robot_running = True
    app_state.start_time = time.time()
    app_state.coordinator.start()
    app_state._engine_task = asyncio.create_task(_engine.run())
    logger.info("Robot started")
    await app_state.broadcast({"event": "robot_started", "timestamp": time.time()})
    return {"status": "started"}


@app.post("/api/robot/stop")
async def stop_robot():
    if not app_state.robot_running:
        return {"status": "not_running"}
    app_state.robot_running = False
    app_state.coordinator.stop()
    if app_state._engine_task:
        app_state._engine_task.cancel()
        app_state._engine_task = None
    logger.info("Robot stopped")
    await app_state.broadcast({"event": "robot_stopped", "timestamp": time.time()})
    return {"status": "stopped"}


@app.get("/api/wave/analysis", response_model=WaveAnalysisSchema)
async def get_wave_analysis():
    df = app_state.data_provider.get_candles(limit=200)
    wa = app_state.wave_detector.analyse(df)
    can_buy = app_state.wave_detector.can_trade("BUY", wa)
    can_sell = app_state.wave_detector.can_trade("SELL", wa)
    return WaveAnalysisSchema(
        main_wave=wa.main_wave.value,
        sub_wave=wa.sub_wave.value if wa.sub_wave else None,
        confidence=wa.confidence,
        htf_ema_fast=wa.htf_ema_fast,
        htf_ema_slow=wa.htf_ema_slow,
        ltf_ema_fast=wa.ltf_ema_fast,
        ltf_ema_slow=wa.ltf_ema_slow,
        atr=wa.atr,
        swing_highs=[{"index": p.index, "price": p.price, "is_high": p.is_high} for p in wa.swing_highs],
        swing_lows=[{"index": p.index, "price": p.price, "is_high": p.is_high} for p in wa.swing_lows],
        sideways_detected=wa.sideways_detected,
        description=wa.description,
        can_trade_buy=can_buy,
        can_trade_sell=can_sell,
    )


@app.get("/api/queue/status", response_model=QueueStatusSchema)
async def get_queue_status():
    m = app_state.coordinator.metrics
    history = app_state.coordinator.history[:10]
    recent = [
        {
            "signal_id": r.signal.signal_id,
            "symbol": r.signal.symbol,
            "direction": r.signal.direction,
            "status": r.status,
            "reason": r.reject_reason,
            "timestamp": r.signal.timestamp,
        }
        for r in history
    ]
    return QueueStatusSchema(
        signals_queued=m.signals_queued,
        signals_executed=m.signals_executed,
        signals_rejected=m.signals_rejected,
        signals_expired=m.signals_expired,
        queue_depth=m.queue_depth,
        cooldown_until=m.cooldown_until,
        state=m.state.value,
        authority=m.authority.value,
        recent_signals=recent,
    )


@app.get("/api/risk/metrics", response_model=RiskMetricsSchema)
async def get_risk_metrics():
    spread = app_state.data_provider.get_spread_points()
    return RiskMetricsSchema(
        balance=app_state.balance,
        equity=app_state.equity,
        daily_pnl=app_state.risk_manager.daily_pnl,
        peak_equity=app_state.risk_manager.peak_equity,
        martingale_step=app_state.risk_manager.martingale_step,
        consecutive_losses=app_state.risk_manager.consecutive_losses,
        dd_triggered=app_state.risk_manager.dd_triggered,
        open_trades=len(app_state.trade_manager.get_open_trades()),
        spread=spread,
    )


@app.get("/api/candles", response_model=List[CandleSchema])
async def get_candles(
    symbol: str = Query("EURUSD"),
    tf: str = Query("M5"),
    limit: int = Query(100, ge=10, le=500),
):
    df = app_state.data_provider.get_candles(limit=limit, timeframe=tf)
    result = []
    for _, row in df.iterrows():
        result.append(CandleSchema(
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            datetime=str(row["datetime"]),
        ))
    return result


# ── WebSocket ──────────────────────────────────────────────────────────── #

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    app_state._ws_clients.add(websocket)
    try:
        # Send initial state
        status = await get_status()
        await websocket.send_json({"event": "init", "status": status.model_dump()})
        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({"event": "heartbeat", "ts": time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        app_state._ws_clients.discard(websocket)


# ── Health check ───────────────────────────────────────────────────────── #

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

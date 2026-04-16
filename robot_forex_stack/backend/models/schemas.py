"""
Pydantic schemas for all API entities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Nested settings ────────────────────────────────────────────────────── #

class MartingaleSettings(BaseModel):
    enabled: bool = False
    multiplier: float = Field(2.0, ge=1.0, le=10.0)
    max_steps: int = Field(4, ge=1, le=10)


class PartialCloseSettings(BaseModel):
    enabled: bool = False
    trigger_pct: float = Field(50.0, ge=1.0, le=99.0)
    close_pct: float = Field(50.0, ge=1.0, le=100.0)
    move_sl_to_be: bool = True


class TrailingSettings(BaseModel):
    enabled: bool = False
    mode: str = "PCT_TP"           # PCT_TP | HILO
    trigger_pct: float = Field(50.0, ge=1.0, le=100.0)
    trail_pct: float = Field(30.0, ge=1.0, le=100.0)


class GridSettings(BaseModel):
    enabled: bool = False
    levels: int = Field(3, ge=1, le=10)
    distance_pips: float = Field(200.0, ge=1.0)
    distance_multiplier: float = Field(1.5, ge=1.0, le=5.0)
    volume_multiplier: float = Field(1.5, ge=1.0, le=5.0)
    max_grid_lot: float = Field(1.0, ge=0.01, le=100.0)


# ── Main settings ──────────────────────────────────────────────────────── #

class RobotSettings(BaseModel):
    # Basic
    username: str = "Trader"
    magic_number: int = 100001
    symbol: str = "EURUSD"
    timeframe: str = "M5"
    htf_timeframe: str = "H1"
    comment: str = "RobotForex"

    # Risk & Position sizing
    lot_mode: str = "STATIC"            # STATIC | DYNAMIC_PERCENT | LOT_PER_X_BALANCE
    lot_value: float = Field(0.01, ge=0.001, le=100.0)
    min_lot: float = 0.01
    max_lot: float = 10.0
    pip_value_per_lot: float = 10.0
    martingale: MartingaleSettings = Field(default_factory=MartingaleSettings)

    # SL/TP
    sl_mode: str = "POINTS"
    sl_value: float = Field(200.0, ge=1.0)
    tp_mode: str = "SL_RATIO"
    tp_value: float = Field(2.0, ge=0.1)

    # Entry
    entry_mode: str = "BREAKOUT"
    retrace_atr_mult: float = 0.5
    min_body_atr: float = 0.3
    retest_level_x: float = 0.5

    # Session
    session: str = "LONDON"
    dst_mode: str = "NO_DST"
    gmt_offset: float = 0.0
    monitoring_minutes: int = 60

    # Filters
    ema_filter_enabled: bool = True
    ema_fast: int = 21
    ema_slow: int = 50
    sr_filter_enabled: bool = True
    max_spread: float = 30.0
    news_filter_enabled: bool = False
    max_trades_at_time: int = 3
    max_trades_daily: int = 10

    # ATR
    atr_period: int = 14
    atr_timeframe: str = "M5"

    # Wave detector
    htf_ema_fast: int = 21
    htf_ema_slow: int = 50
    ltf_ema_fast: int = 8
    ltf_ema_slow: int = 21
    sideways_atr_mult: float = 1.5
    sideways_candles: int = 10

    # Trade management
    partial_close: PartialCloseSettings = Field(default_factory=PartialCloseSettings)
    trailing: TrailingSettings = Field(default_factory=TrailingSettings)
    grid: GridSettings = Field(default_factory=GridSettings)

    # Risk management
    max_account_equity: float = 0.0
    max_daily_dd_pct: float = 5.0
    max_overall_dd_pct: float = 20.0

    # Coordinator
    max_queue_size: int = 10
    cooldown_minutes: float = 5.0
    signal_expiry_seconds: float = 300.0


# ── API response schemas ────────────────────────────────────────────────── #

class TradeSignalSchema(BaseModel):
    signal_id: str
    symbol: str
    direction: str
    entry_price: float
    sl: float
    tp: float
    lot_size: float
    entry_mode: str
    priority: int = 0
    timestamp: float
    meta: Dict[str, Any] = {}


class TradeRecordSchema(BaseModel):
    trade_id: str
    symbol: str
    direction: str
    lot_size: float
    entry_price: float
    sl: float
    tp: float
    entry_mode: str
    open_time: float
    close_time: Optional[float] = None
    close_price: Optional[float] = None
    pnl: float = 0.0
    status: str = "OPEN"
    remaining_lots: float = 0.0
    be_moved: bool = False
    grid_level: int = 0
    comment: str = ""


class SwingPointSchema(BaseModel):
    index: int
    price: float
    is_high: bool


class WaveAnalysisSchema(BaseModel):
    main_wave: str
    sub_wave: Optional[str] = None
    confidence: float
    htf_ema_fast: float
    htf_ema_slow: float
    ltf_ema_fast: float
    ltf_ema_slow: float
    atr: float
    swing_highs: List[SwingPointSchema] = []
    swing_lows: List[SwingPointSchema] = []
    sideways_detected: bool = False
    description: str = ""
    can_trade_buy: bool = False
    can_trade_sell: bool = False


class QueueStatusSchema(BaseModel):
    signals_queued: int
    signals_executed: int
    signals_rejected: int
    signals_expired: int
    queue_depth: int
    cooldown_until: float
    state: str
    authority: str
    recent_signals: List[Dict[str, Any]] = []


class RiskMetricsSchema(BaseModel):
    balance: float
    equity: float
    daily_pnl: float
    peak_equity: float
    martingale_step: int
    consecutive_losses: int
    dd_triggered: bool
    open_trades: int
    spread: float = 0.0


class RobotStatusSchema(BaseModel):
    running: bool
    state: str                   # IDLE | MONITORING | RUNNING | STOPPED
    wave_state: str
    sub_wave: Optional[str]
    confidence: float
    coordinator_state: str
    balance: float
    equity: float
    total_pnl: float
    win_rate: float
    profit_factor: float
    total_trades: int
    open_trades: int
    daily_pnl: float
    uptime_seconds: float


class CandleSchema(BaseModel):
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    datetime: str = ""


class AutoPilotCandidateSchema(BaseModel):
    mode: str
    direction: str
    score: float


class AutoPilotLastDecisionSchema(BaseModel):
    timestamp: float
    candidates_evaluated: int
    candidates_passed: int
    best_mode: Optional[str] = None
    best_direction: Optional[str] = None
    best_score: float
    action: str
    signal_id: Optional[str] = None
    tick_interval: float
    via_retracement: bool = False
    top_candidates: List[AutoPilotCandidateSchema] = []


class AutoPilotStatusSchema(BaseModel):
    enabled: bool
    current_tick_interval: float
    decisions_total: int
    signals_generated: int
    min_score_threshold: float
    last_decision: Optional[AutoPilotLastDecisionSchema] = None
    recent_decisions: List[AutoPilotLastDecisionSchema] = []


class SupportResistanceLevelSchema(BaseModel):
    price: float
    strength: float
    sr_type: str
    touch_count: int


class RetracementStatusSchema(BaseModel):
    """Trạng thái real-time của Retracement Engine — dành cho operator giám sát."""
    in_retracement:  bool
    main_direction:  str
    zone:            str       # NOT_RETRACING | SHALLOW | GOLDEN_ZONE | DEEP | STRUCTURE_BROKEN
    retrace_pct:     float     # phần trăm đã hồi (0.0–1.0)
    nearest_fib:     str       # "0.382", "0.500", "0.618", etc.
    quality:         float     # 0.0–1.0
    bounce_detected: bool
    impulse_start:   float
    impulse_end:     float
    current_price:   float
    safest_entry:    float
    safest_sl:       float
    safest_tp:       float
    tp_extension:    float
    risk_reward:     float
    fib_levels:      Dict[str, float] = {}
    sr_levels:       List[SupportResistanceLevelSchema] = []
    description:     str = ""


class BrokerStatusSchema(BaseModel):
    provider_type: str                    # MOCK | CTRADER
    connected: bool
    app_authenticated: bool
    account_authenticated: bool
    history_loaded: bool
    symbol: str
    symbol_id: int
    timeframe: str
    live: bool
    last_error: str
    last_tick_ts: float
    bars_loaded: int
    account_id: int


class PaginatedTrades(BaseModel):
    trades: List[TradeRecordSchema]
    total: int
    page: int
    page_size: int


# ── Decision Engine schemas ────────────────────────────────────────────── #

class MarketRegimeSchema(BaseModel):
    continuation_prob: float   # 0–1: probability trend continues
    volatility_regime: str     # LOW | NORMAL | HIGH | EXTREME
    momentum_score:    float   # 0–1
    atr_percentile:    float   # 0–1


class SimulatedOutcomeSchema(BaseModel):
    expected_value:        float
    win_probability:       float
    max_adverse_excursion: float


class SegmentStatsSchema(BaseModel):
    win_rate:      float
    profit_factor: float
    avg_rr:        float
    expectancy:    float
    sample_size:   int


class DecisionContextSchema(BaseModel):
    """Current tick decision context from DecisionEngine."""
    action:               str              # SCAN_AND_ENTER | HOLD | etc.
    lot_scale:            float
    effective_min_score:  float
    regime:               MarketRegimeSchema
    adaptive_paused:      bool
    pause_reason:         str
    consecutive_losses:   int
    mode_weight_multipliers: Dict[str, float] = {}
    meta:                 Dict[str, Any] = {}


class DecisionEngineStatusSchema(BaseModel):
    """Full status of the Decision Engine for operator monitoring."""
    last_action:          str
    lot_scale:            float
    effective_min_score:  float
    adaptive_paused:      bool
    pause_reason:         str
    consecutive_losses:   int
    adaptation_count:     int
    regime:               Optional[MarketRegimeSchema] = None
    global_stats:         SegmentStatsSchema
    segment_stats:        Dict[str, SegmentStatsSchema] = {}
    mode_weight_adjs:     Dict[str, float] = {}
    recent_outcomes:      List[Dict[str, Any]] = []


# ── Performance Tracker / Central Brain schemas ────────────────────────── #

class TradeFingerprintSchema(BaseModel):
    """8-component pattern fingerprint."""
    mode:              str
    wave_state:        str
    direction:         str
    retrace_zone:      str
    session:           str
    volatility:        str
    hour:              int
    dow:               int


class PreTradeConsultationSchema(BaseModel):
    """Result of mandatory pre-trade pipeline gate."""
    should_trade:     bool
    win_probability:  float
    loss_risk:        float
    authority:        str    # CLEAR | RESTRICTED | BLOCKED
    block_reason:     str
    pattern_known:    bool
    pattern_win_rate: float
    global_win_rate:  float
    priority_boost:   float
    consultation_id:  str
    timestamp:        float


class PatternSummarySchema(BaseModel):
    """Summary of a single win or loss pattern."""
    fingerprint:  TradeFingerprintSchema
    win_rate:     Optional[float] = None
    loss_rate:    Optional[float] = None
    total:        int
    avg_pnl:      float


class PerformanceDashboardSchema(BaseModel):
    """Full performance dashboard — central brain status."""
    total_recorded:       int
    pattern_count:        int
    global_win_rate:      float
    global_profit_factor: float
    global_avg_rr:        float
    global_expectancy:    float
    global_sample_size:   int
    consecutive_losses:   int
    win_patterns_count:   int
    loss_patterns_count:  int
    top_win_patterns:     List[PatternSummarySchema] = []
    top_loss_patterns:    List[PatternSummarySchema] = []
    last_consultation:    Optional[Dict[str, Any]] = None

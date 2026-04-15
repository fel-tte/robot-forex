"""
Robot Forex — Streamlit Dashboard
5-page trading robot dashboard polling the FastAPI backend.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ── Config ─────────────────────────────────────────────────────────────── #

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")
POLL_INTERVAL = 5   # seconds between auto-refresh

st.set_page_config(
    page_title="Robot Forex Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API helpers ────────────────────────────────────────────────────────── #

def api_get(path: str, default: Any = None) -> Any:
    try:
        r = requests.get(f"{BACKEND}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


def api_post(path: str, data: Optional[dict] = None) -> Any:
    try:
        r = requests.post(f"{BACKEND}{path}", json=data or {}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Custom CSS ─────────────────────────────────────────────────────────── #

st.markdown("""
<style>
.metric-card {
    background: #1e2130;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    border: 1px solid #2d3250;
}
.metric-card .label { font-size: 0.78rem; color: #8b95b0; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
.bull { color: #00e676; }
.bear { color: #ff5252; }
.sideways { color: #ffd740; }
.subwave { color: #ff9100; }
.running-badge { background: #00c853; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
.stopped-badge { background: #c62828; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
.cooldown-badge { background: #e65100; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────── #

with st.sidebar:
    st.markdown("## 🤖 Robot Forex")
    st.markdown("---")

    status = api_get("/api/status", {})
    running = status.get("running", False)

    if running:
        st.markdown('<span class="running-badge">● RUNNING</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="stopped-badge">● STOPPED</span>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True, type="primary", disabled=running):
            result = api_post("/api/robot/start")
            if result:
                st.success("Started!")
                st.rerun()
    with col2:
        if st.button("■ Stop", use_container_width=True, type="secondary", disabled=not running):
            result = api_post("/api/robot/stop")
            if result:
                st.warning("Stopped")
                st.rerun()

    st.markdown("---")
    wave = status.get("wave_state", "SIDEWAYS")
    sub = status.get("sub_wave")
    conf = status.get("confidence", 0.0)

    wave_cls = "bull" if "BULL" in wave else ("bear" if "BEAR" in wave else "sideways")
    st.markdown(f'<div style="text-align:center"><span class="{wave_cls}" style="font-size:1.2rem;font-weight:700">{wave}</span></div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div style="text-align:center"><span class="subwave">⚠ Sub-wave: {sub}</span></div>', unsafe_allow_html=True)
    st.progress(conf, text=f"Confidence: {conf:.0%}")

    st.markdown("---")
    bal = status.get("balance", 10000)
    eq = status.get("equity", 10000)
    pnl = status.get("total_pnl", 0)
    st.metric("Balance", f"${bal:,.2f}")
    st.metric("Equity", f"${eq:,.2f}", delta=f"{eq - bal:+.2f}")
    st.metric("Total P&L", f"${pnl:,.2f}", delta=f"{pnl:+.2f}")

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    if st.button("🔄 Refresh Now"):
        st.rerun()

    if auto_refresh:
        time.sleep(POLL_INTERVAL)
        st.rerun()

# ── Navigation ─────────────────────────────────────────────────────────── #

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🌊 Wave Analysis",
    "📋 Signal Queue",
    "⚙️ Settings",
    "📜 Trade History",
])


# ══════════════════════════════════════════════════════════════════════════ #
#  PAGE 1 — DASHBOARD                                                       #
# ══════════════════════════════════════════════════════════════════════════ #

with tab1:
    st.markdown("## 📊 Dashboard")

    status = api_get("/api/status", {})
    risk = api_get("/api/risk/metrics", {})
    candles = api_get("/api/candles?limit=100", [])

    # Top metrics row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        wave = status.get("wave_state", "—")
        wcolor = "🟢" if "BULL" in wave else ("🔴" if "BEAR" in wave else "🟡")
        st.metric("Main Wave", f"{wcolor} {wave}")
    with c2:
        st.metric("Confidence", f"{status.get('confidence', 0):.0%}")
    with c3:
        st.metric("Win Rate", f"{status.get('win_rate', 0):.1f}%")
    with c4:
        st.metric("Profit Factor", f"{status.get('profit_factor', 0):.2f}")
    with c5:
        st.metric("Open Trades", status.get("open_trades", 0))
    with c6:
        st.metric("Total Trades", status.get("total_trades", 0))

    st.markdown("---")

    # Sub-wave warning
    sub_wave = status.get("sub_wave")
    if sub_wave:
        st.warning(f"⚠️ **Sub-wave detected: {sub_wave}** — Trading paused until main wave resumes")

    # Drawdown alert
    if risk.get("dd_triggered"):
        st.error("🚨 **Drawdown protection triggered** — All trading halted")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Equity curve from closed trades
        trades_data = api_get("/api/trades?page_size=200", {})
        trades = trades_data.get("trades", []) if trades_data else []
        closed = [t for t in trades if t.get("status") == "CLOSED"]

        if closed:
            df_trades = pd.DataFrame(closed)
            df_trades["open_time"] = pd.to_datetime(df_trades["open_time"], unit="s")
            df_trades = df_trades.sort_values("open_time")
            df_trades["cumulative_pnl"] = df_trades["pnl"].cumsum()

            fig_equity = go.Figure()
            colors = ["#00e676" if p >= 0 else "#ff5252" for p in df_trades["cumulative_pnl"]]
            fig_equity.add_trace(go.Scatter(
                x=df_trades["open_time"],
                y=df_trades["cumulative_pnl"],
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color="#00e676", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,230,118,0.1)",
            ))
            fig_equity.update_layout(
                title="Cumulative P&L Curve",
                xaxis_title="Time",
                yaxis_title="P&L ($)",
                height=350,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="#1e2130"),
                yaxis=dict(gridcolor="#1e2130"),
            )
            st.plotly_chart(fig_equity, use_container_width=True)
        else:
            st.info("📈 Equity curve will appear after trades are closed.")

        # Price mini-chart
        if candles:
            df_c = pd.DataFrame(candles)
            df_c["dt"] = pd.to_datetime(df_c["timestamp"], unit="s")
            df_c = df_c.tail(50)
            fig_price = go.Figure(data=[go.Candlestick(
                x=df_c["dt"],
                open=df_c["open"],
                high=df_c["high"],
                low=df_c["low"],
                close=df_c["close"],
                name="Price",
                increasing_line_color="#00e676",
                decreasing_line_color="#ff5252",
            )])
            fig_price.update_layout(
                title="Recent Price Action (last 50 candles)",
                height=280,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="#1e2130", rangeslider=dict(visible=False)),
                yaxis=dict(gridcolor="#1e2130"),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_price, use_container_width=True)

    with col_right:
        # Risk panel
        st.markdown("### Risk Metrics")
        balance = risk.get("balance", 0)
        equity_r = risk.get("equity", balance)
        peak = risk.get("peak_equity", balance)
        dd_pct = (peak - equity_r) / peak * 100 if peak > 0 else 0.0

        st.metric("Balance", f"${balance:,.2f}")
        st.metric("Equity", f"${equity_r:,.2f}")
        st.metric("Daily P&L", f"${risk.get('daily_pnl', 0):+,.2f}")
        st.metric("Peak Equity", f"${peak:,.2f}")

        dd_color = "normal" if dd_pct < 5 else ("off" if dd_pct < 15 else "inverse")
        st.metric("Current DD", f"{dd_pct:.2f}%", delta=f"-{dd_pct:.2f}%", delta_color="inverse")
        st.metric("Martingale Step", risk.get("martingale_step", 0))
        st.metric("Consec. Losses", risk.get("consecutive_losses", 0))
        st.metric("Spread", f"{risk.get('spread', 0):.1f} pips")

        # Open trades
        open_trades = api_get("/api/trades/open", [])
        if open_trades:
            st.markdown("### Open Trades")
            df_open = pd.DataFrame(open_trades)
            st.dataframe(
                df_open[["trade_id", "symbol", "direction", "lot_size", "entry_price", "pnl"]],
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════ #
#  PAGE 2 — WAVE ANALYSIS                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

with tab2:
    st.markdown("## 🌊 Wave Analysis")

    wave_data = api_get("/api/wave/analysis", {})
    candles = api_get("/api/candles?limit=200", [])

    if not wave_data:
        st.error("Could not fetch wave analysis. Is the backend running?")
    else:
        # State banner
        main_wave = wave_data.get("main_wave", "SIDEWAYS")
        sub_wave = wave_data.get("sub_wave")
        conf = wave_data.get("confidence", 0.0)
        sideways = wave_data.get("sideways_detected", False)

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            color = "🟢" if "BULL" in main_wave else ("🔴" if "BEAR" in main_wave else "🟡")
            st.metric("Main Wave", f"{color} {main_wave}")
        with bcol2:
            sub_label = sub_wave if sub_wave else "None"
            st.metric("Sub Wave", f"{'⚠️ ' if sub_wave else ''}{sub_label}")
        with bcol3:
            st.metric("Confidence", f"{conf:.0%}")
        with bcol4:
            st.metric("Sideways", "Yes 🟡" if sideways else "No ✅")

        can_buy = wave_data.get("can_trade_buy", False)
        can_sell = wave_data.get("can_trade_sell", False)
        trade_status = []
        if can_buy:
            trade_status.append("✅ BUY signals allowed")
        if can_sell:
            trade_status.append("✅ SELL signals allowed")
        if not can_buy and not can_sell:
            trade_status.append("🚫 Trading paused (sub-wave or sideways)")

        for ts in trade_status:
            if "✅" in ts:
                st.success(ts)
            else:
                st.warning(ts)

        st.markdown(f"**Analysis:** {wave_data.get('description', '')}")

        if candles:
            df_c = pd.DataFrame(candles).tail(150)
            df_c["dt"] = pd.to_datetime(df_c["timestamp"], unit="s")

            htf_fast = wave_data.get("htf_ema_fast", 0)
            htf_slow = wave_data.get("htf_ema_slow", 0)
            ltf_fast = wave_data.get("ltf_ema_fast", 0)
            ltf_slow = wave_data.get("ltf_ema_slow", 0)
            atr = wave_data.get("atr", 0)

            # Compute EMA series from candle data
            close_series = df_c["close"]
            ema_htf_fast = close_series.ewm(span=21, adjust=False).mean()
            ema_htf_slow = close_series.ewm(span=50, adjust=False).mean()
            ema_ltf_fast = close_series.ewm(span=8, adjust=False).mean()
            ema_ltf_slow = close_series.ewm(span=21, adjust=False).mean()

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.75, 0.25],
                vertical_spacing=0.03,
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_c["dt"],
                open=df_c["open"],
                high=df_c["high"],
                low=df_c["low"],
                close=df_c["close"],
                name="Price",
                increasing_line_color="#00e676",
                decreasing_line_color="#ff5252",
            ), row=1, col=1)

            # EMAs
            fig.add_trace(go.Scatter(x=df_c["dt"], y=ema_htf_fast, name="HTF Fast EMA(21)",
                                      line=dict(color="#40c4ff", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c["dt"], y=ema_htf_slow, name="HTF Slow EMA(50)",
                                      line=dict(color="#ff6d00", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c["dt"], y=ema_ltf_fast, name="LTF Fast EMA(8)",
                                      line=dict(color="#b39ddb", width=1, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_c["dt"], y=ema_ltf_slow, name="LTF Slow EMA(21)",
                                      line=dict(color="#f48fb1", width=1, dash="dot")), row=1, col=1)

            # Fractal swing points
            swing_highs = wave_data.get("swing_highs", [])
            swing_lows = wave_data.get("swing_lows", [])

            if swing_highs:
                sh_prices = [p["price"] for p in swing_highs]
                n = len(df_c)
                sh_xs = [df_c["dt"].iloc[max(0, min(int(p["index"]) % n, n - 1))] for p in swing_highs]
                fig.add_trace(go.Scatter(
                    x=sh_xs, y=sh_prices,
                    mode="markers",
                    name="Fractal High",
                    marker=dict(symbol="triangle-up", size=10, color="#ff5252"),
                ), row=1, col=1)

            if swing_lows:
                sl_prices = [p["price"] for p in swing_lows]
                n = len(df_c)
                sl_xs = [df_c["dt"].iloc[max(0, min(int(p["index"]) % n, n - 1))] for p in swing_lows]
                fig.add_trace(go.Scatter(
                    x=sl_xs, y=sl_prices,
                    mode="markers",
                    name="Fractal Low",
                    marker=dict(symbol="triangle-down", size=10, color="#00e676"),
                ), row=1, col=1)

            # Volume bar
            fig.add_trace(go.Bar(
                x=df_c["dt"], y=df_c["volume"],
                name="Volume",
                marker_color="rgba(128,128,200,0.5)",
            ), row=2, col=1)

            fig.update_layout(
                title=f"Wave Analysis Chart — {main_wave}",
                height=650,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="#1e2130", rangeslider=dict(visible=False)),
                yaxis=dict(gridcolor="#1e2130"),
                xaxis2=dict(gridcolor="#1e2130"),
                yaxis2=dict(gridcolor="#1e2130"),
                legend=dict(bgcolor="rgba(30,33,48,0.8)"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # EMA values
        st.markdown("### Current EMA Values")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("HTF Fast EMA(21)", f"{wave_data.get('htf_ema_fast', 0):.5f}")
        col_b.metric("HTF Slow EMA(50)", f"{wave_data.get('htf_ema_slow', 0):.5f}")
        col_c.metric("LTF Fast EMA(8)", f"{wave_data.get('ltf_ema_fast', 0):.5f}")
        col_d.metric("LTF Slow EMA(21)", f"{wave_data.get('ltf_ema_slow', 0):.5f}")
        st.metric("ATR(14)", f"{wave_data.get('atr', 0):.5f}")


# ══════════════════════════════════════════════════════════════════════════ #
#  PAGE 3 — SIGNAL QUEUE                                                    #
# ══════════════════════════════════════════════════════════════════════════ #

with tab3:
    st.markdown("## 📋 Signal Queue")

    queue = api_get("/api/queue/status", {})
    if not queue:
        st.error("Could not fetch queue status.")
    else:
        state = queue.get("state", "IDLE")
        authority = queue.get("authority", "NORMAL")
        cooldown_until = queue.get("cooldown_until", 0)

        # State indicators
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        with s_col1:
            state_emoji = {"IDLE": "⚫", "MONITORING": "🔵", "COOLDOWN": "🟠", "RESTRICTED": "🔴"}.get(state, "⚪")
            st.metric("Coordinator State", f"{state_emoji} {state}")
        with s_col2:
            auth_emoji = {"BLOCKED": "🚫", "RESTRICTED": "⚠️", "NORMAL": "✅", "PRIORITY": "⭐"}.get(authority, "❓")
            st.metric("Authority", f"{auth_emoji} {authority}")
        with s_col3:
            st.metric("Queue Depth", queue.get("queue_depth", 0))
        with s_col4:
            if cooldown_until > time.time():
                remaining = max(0, int(cooldown_until - time.time()))
                st.metric("Cooldown Remaining", f"{remaining}s")
            else:
                st.metric("Cooldown", "None ✅")

        if state == "COOLDOWN":
            st.warning(f"⏱ Cooldown active after loss. Resumes in {max(0, int(cooldown_until - time.time()))} seconds.")

        st.markdown("---")

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signals Queued (total)", queue.get("signals_queued", 0))
        m2.metric("Signals Executed", queue.get("signals_executed", 0))
        m3.metric("Signals Rejected", queue.get("signals_rejected", 0))
        m4.metric("Signals Expired", queue.get("signals_expired", 0))

        # Execution rate
        total = queue.get("signals_queued", 1) or 1
        exec_rate = queue.get("signals_executed", 0) / total * 100
        st.progress(exec_rate / 100, text=f"Execution Rate: {exec_rate:.1f}%")

        # Recent signal history
        recent = queue.get("recent_signals", [])
        if recent:
            st.markdown("### Recent Signal History")
            df_sig = pd.DataFrame(recent)
            if "timestamp" in df_sig.columns:
                df_sig["time"] = pd.to_datetime(df_sig["timestamp"], unit="s").dt.strftime("%H:%M:%S")
            status_colors = {
                "EXECUTED": "background-color: rgba(0,200,83,0.2)",
                "REJECTED": "background-color: rgba(255,82,82,0.2)",
                "QUEUED": "background-color: rgba(64,196,255,0.2)",
                "EXPIRED": "background-color: rgba(255,145,0,0.2)",
            }

            def color_status(val):
                return status_colors.get(val, "")

            display_cols = ["time", "signal_id", "symbol", "direction", "status", "reason"] if "time" in df_sig.columns else df_sig.columns.tolist()
            available_cols = [c for c in display_cols if c in df_sig.columns]

            if available_cols:
                styled = df_sig[available_cols].style.applymap(color_status, subset=["status"] if "status" in available_cols else [])
                st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("No signal history yet. Start the robot to generate signals.")

        # Visual queue depth gauge
        st.markdown("### Load Monitor")
        max_q = api_get("/api/settings", {}).get("max_queue_size", 10)
        depth = queue.get("queue_depth", 0)
        load_pct = depth / max_q if max_q > 0 else 0

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=depth,
            title={"text": "Queue Depth", "font": {"color": "#fafafa"}},
            gauge={
                "axis": {"range": [0, max_q], "tickcolor": "#fafafa"},
                "bar": {"color": "#40c4ff"},
                "bgcolor": "#1e2130",
                "steps": [
                    {"range": [0, max_q * 0.5], "color": "rgba(0,230,118,0.2)"},
                    {"range": [max_q * 0.5, max_q * 0.8], "color": "rgba(255,215,0,0.2)"},
                    {"range": [max_q * 0.8, max_q], "color": "rgba(255,82,82,0.2)"},
                ],
                "threshold": {"line": {"color": "#ff5252", "width": 3}, "thickness": 0.75, "value": max_q * 0.9},
            },
        ))
        fig_gauge.update_layout(
            height=250,
            paper_bgcolor="#0e1117",
            font=dict(color="#fafafa"),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════ #
#  PAGE 4 — SETTINGS                                                        #
# ══════════════════════════════════════════════════════════════════════════ #

with tab4:
    st.markdown("## ⚙️ Robot Settings")

    current = api_get("/api/settings", {})
    if not current:
        st.error("Could not load settings.")
    else:
        with st.form("settings_form"):
            # ── Basic Setup ──────────────────────────────────────────── #
            with st.expander("🔧 Basic Setup", expanded=True):
                bc1, bc2, bc3 = st.columns(3)
                username = bc1.text_input("Username", value=current.get("username", "Trader"))
                magic = bc2.number_input("Magic Number", value=current.get("magic_number", 100001), min_value=1)
                symbol = bc3.text_input("Symbol", value=current.get("symbol", "EURUSD"))
                tf1, tf2 = st.columns(2)
                timeframe = tf1.selectbox("Timeframe", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                                           index=["M1", "M5", "M15", "M30", "H1", "H4", "D1"].index(current.get("timeframe", "M5")))
                htf_tf = tf2.selectbox("HTF Timeframe", ["M15", "M30", "H1", "H4", "D1"],
                                        index=["M15", "M30", "H1", "H4", "D1"].index(current.get("htf_timeframe", "H1")))

            # ── Risk & Position Sizing ───────────────────────────────── #
            with st.expander("💰 Risk & Position Sizing"):
                r1, r2, r3 = st.columns(3)
                lot_mode = r1.selectbox("Lot Mode", ["STATIC", "DYNAMIC_PERCENT", "LOT_PER_X_BALANCE"],
                                         index=["STATIC", "DYNAMIC_PERCENT", "LOT_PER_X_BALANCE"].index(current.get("lot_mode", "STATIC")))
                lot_value = r2.number_input("Lot Value", value=float(current.get("lot_value", 0.01)),
                                             min_value=0.001, max_value=100.0, format="%.3f")
                pip_value = r3.number_input("Pip Value/Lot ($)", value=float(current.get("pip_value_per_lot", 10.0)))
                rr1, rr2 = st.columns(2)
                min_lot = rr1.number_input("Min Lot", value=float(current.get("min_lot", 0.01)), format="%.2f")
                max_lot = rr2.number_input("Max Lot", value=float(current.get("max_lot", 10.0)), format="%.2f")

                st.markdown("**Martingale**")
                mg = current.get("martingale", {})
                m1c, m2c, m3c = st.columns(3)
                mg_enabled = m1c.checkbox("Enable Martingale", value=mg.get("enabled", False))
                mg_mult = m2c.number_input("Multiplier", value=float(mg.get("multiplier", 2.0)), min_value=1.0, max_value=10.0)
                mg_steps = m3c.number_input("Max Steps", value=int(mg.get("max_steps", 4)), min_value=1, max_value=10)

            # ── SL / TP ──────────────────────────────────────────────── #
            with st.expander("🎯 Stop Loss & Take Profit"):
                sl_modes = ["POINTS", "ATR", "RANGE_SIZE", "PREV_CANDLE_POINTS", "PREV_CANDLE_ATR",
                            "LAST_SWING_POINTS", "LAST_SWING_ATR", "RANGE_OPPOSITE_POINTS", "RANGE_OPPOSITE_ATR"]
                tp_modes = ["SL_RATIO", "ATR", "POINTS"]
                s1, s2 = st.columns(2)
                sl_mode = s1.selectbox("SL Mode", sl_modes,
                                        index=sl_modes.index(current.get("sl_mode", "POINTS")))
                sl_value = s1.number_input("SL Value", value=float(current.get("sl_value", 200.0)), min_value=1.0)
                tp_mode = s2.selectbox("TP Mode", tp_modes,
                                        index=tp_modes.index(current.get("tp_mode", "SL_RATIO")))
                tp_value = s2.number_input("TP Value", value=float(current.get("tp_value", 2.0)), min_value=0.1)

            # ── Sessions & Time ──────────────────────────────────────── #
            with st.expander("🕐 Sessions & Time"):
                ss1, ss2, ss3 = st.columns(3)
                sessions = ["AMERICAN", "NYSE", "EUROPEAN", "LONDON", "ASIAN", "CUSTOM", "ALL_DAY"]
                session = ss1.selectbox("Session", sessions,
                                         index=sessions.index(current.get("session", "LONDON")))
                dst_modes = ["NO_DST", "NORTH_AMERICA", "EUROPE"]
                dst_mode = ss2.selectbox("DST Mode", dst_modes,
                                          index=dst_modes.index(current.get("dst_mode", "NO_DST")))
                gmt_offset = ss3.number_input("GMT Offset", value=float(current.get("gmt_offset", 0.0)),
                                               min_value=-12.0, max_value=14.0, step=0.5)
                monitoring_minutes = st.number_input("Monitoring Minutes (Range period)",
                                                      value=int(current.get("monitoring_minutes", 60)),
                                                      min_value=5, max_value=240)

            # ── Entry Logic ──────────────────────────────────────────── #
            with st.expander("📐 Entry Logic"):
                entry_modes = ["BREAKOUT", "INSTANT_BREAKOUT", "RETRACE", "INSTANT_RETRACE",
                               "RETEST_SAME", "RETEST_OPPOSITE", "RETEST_LEVEL_X"]
                e1, e2 = st.columns(2)
                entry_mode = e1.selectbox("Entry Mode", entry_modes,
                                           index=entry_modes.index(current.get("entry_mode", "BREAKOUT")))
                retrace_mult = e1.number_input("Retrace ATR Multiplier", value=float(current.get("retrace_atr_mult", 0.5)))
                min_body_atr = e2.number_input("Min Body ATR", value=float(current.get("min_body_atr", 0.3)))
                retest_lvl = e2.number_input("Retest Level X (0-1)", value=float(current.get("retest_level_x", 0.5)),
                                              min_value=0.0, max_value=1.0)

            # ── Filters ──────────────────────────────────────────────── #
            with st.expander("🔍 Filters"):
                f1, f2 = st.columns(2)
                ema_filter = f1.checkbox("EMA Filter", value=current.get("ema_filter_enabled", True))
                ema_fast = f1.number_input("EMA Fast Period", value=int(current.get("ema_fast", 21)))
                ema_slow = f1.number_input("EMA Slow Period", value=int(current.get("ema_slow", 50)))
                sr_filter = f2.checkbox("S/R Filter (Fractals)", value=current.get("sr_filter_enabled", True))
                max_spread = f2.number_input("Max Spread (pips)", value=float(current.get("max_spread", 30.0)))
                news_filter = f2.checkbox("News Filter", value=current.get("news_filter_enabled", False))
                mt1, mt2 = st.columns(2)
                max_trades_time = mt1.number_input("Max Trades at a Time", value=int(current.get("max_trades_at_time", 3)),
                                                    min_value=1, max_value=50)
                max_trades_daily = mt2.number_input("Max Trades Daily", value=int(current.get("max_trades_daily", 10)),
                                                     min_value=1, max_value=200)

            # ── ATR ──────────────────────────────────────────────────── #
            with st.expander("📏 ATR Settings"):
                at1, at2 = st.columns(2)
                atr_period = at1.number_input("ATR Period", value=int(current.get("atr_period", 14)), min_value=1)
                atr_tf = at2.selectbox("ATR Timeframe", ["M1", "M5", "M15", "M30", "H1", "H4"],
                                        index=["M1", "M5", "M15", "M30", "H1", "H4"].index(current.get("atr_timeframe", "M5")))

            # ── Trade Management ─────────────────────────────────────── #
            with st.expander("🔄 Trade Management"):
                st.markdown("**Partial Close**")
                pc = current.get("partial_close", {})
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc_enabled = pc1.checkbox("Enable Partial Close", value=pc.get("enabled", False))
                pc_trigger = pc2.number_input("Trigger % of TP", value=float(pc.get("trigger_pct", 50.0)), min_value=1.0, max_value=99.0)
                pc_close = pc3.number_input("Close % Lots", value=float(pc.get("close_pct", 50.0)), min_value=1.0, max_value=100.0)
                pc_be = pc4.checkbox("Move SL to BE", value=pc.get("move_sl_to_be", True))

                st.markdown("**Trailing Stop**")
                tr = current.get("trailing", {})
                tr1, tr2, tr3, tr4 = st.columns(4)
                tr_enabled = tr1.checkbox("Enable Trailing", value=tr.get("enabled", False))
                tr_mode = tr2.selectbox("Trail Mode", ["PCT_TP", "HILO"],
                                         index=["PCT_TP", "HILO"].index(tr.get("mode", "PCT_TP")))
                tr_trigger = tr3.number_input("Trail Trigger %", value=float(tr.get("trigger_pct", 50.0)))
                tr_pct = tr4.number_input("Trail Distance %", value=float(tr.get("trail_pct", 30.0)))

                st.markdown("**Grid System**")
                gr = current.get("grid", {})
                gr1, gr2, gr3 = st.columns(3)
                gr_enabled = gr1.checkbox("Enable Grid", value=gr.get("enabled", False))
                gr_levels = gr1.number_input("Grid Levels", value=int(gr.get("levels", 3)), min_value=1, max_value=10)
                gr_dist = gr2.number_input("Distance (pips)", value=float(gr.get("distance_pips", 200.0)))
                gr_dist_mult = gr2.number_input("Distance Multiplier", value=float(gr.get("distance_multiplier", 1.5)))
                gr_vol_mult = gr3.number_input("Volume Multiplier", value=float(gr.get("volume_multiplier", 1.5)))
                gr_max_lot = gr3.number_input("Max Grid Lot", value=float(gr.get("max_grid_lot", 1.0)))

            # ── Wave Detector ────────────────────────────────────────── #
            with st.expander("🌊 Wave Detector Parameters"):
                wd1, wd2 = st.columns(2)
                htf_ef = wd1.number_input("HTF EMA Fast", value=int(current.get("htf_ema_fast", 21)))
                htf_es = wd1.number_input("HTF EMA Slow", value=int(current.get("htf_ema_slow", 50)))
                ltf_ef = wd2.number_input("LTF EMA Fast", value=int(current.get("ltf_ema_fast", 8)))
                ltf_es = wd2.number_input("LTF EMA Slow", value=int(current.get("ltf_ema_slow", 21)))
                sw1, sw2 = st.columns(2)
                sw_mult = sw1.number_input("Sideways ATR Mult", value=float(current.get("sideways_atr_mult", 1.5)))
                sw_candles = sw2.number_input("Sideways Candles", value=int(current.get("sideways_candles", 10)))

            # ── Advanced Risk ────────────────────────────────────────── #
            with st.expander("🛡️ Advanced Risk Management"):
                ar1, ar2, ar3 = st.columns(3)
                max_eq = ar1.number_input("Max Account Equity ($, 0=off)", value=float(current.get("max_account_equity", 0.0)), min_value=0.0)
                max_daily_dd = ar2.number_input("Max Daily DD (%)", value=float(current.get("max_daily_dd_pct", 5.0)), min_value=0.1, max_value=100.0)
                max_overall_dd = ar3.number_input("Max Overall DD (%)", value=float(current.get("max_overall_dd_pct", 20.0)), min_value=0.1, max_value=100.0)

                cq1, cq2, cq3 = st.columns(3)
                max_q = cq1.number_input("Max Queue Size", value=int(current.get("max_queue_size", 10)))
                cooldown = cq2.number_input("Cooldown (minutes)", value=float(current.get("cooldown_minutes", 5.0)))
                expiry = cq3.number_input("Signal Expiry (seconds)", value=float(current.get("signal_expiry_seconds", 300.0)))

            submitted = st.form_submit_button("💾 Save Settings", use_container_width=True, type="primary")

        if submitted:
            new_settings = {
                "username": username,
                "magic_number": int(magic),
                "symbol": symbol,
                "timeframe": timeframe,
                "htf_timeframe": htf_tf,
                "lot_mode": lot_mode,
                "lot_value": float(lot_value),
                "min_lot": float(min_lot),
                "max_lot": float(max_lot),
                "pip_value_per_lot": float(pip_value),
                "martingale": {
                    "enabled": mg_enabled,
                    "multiplier": float(mg_mult),
                    "max_steps": int(mg_steps),
                },
                "sl_mode": sl_mode,
                "sl_value": float(sl_value),
                "tp_mode": tp_mode,
                "tp_value": float(tp_value),
                "entry_mode": entry_mode,
                "retrace_atr_mult": float(retrace_mult),
                "min_body_atr": float(min_body_atr),
                "retest_level_x": float(retest_lvl),
                "session": session,
                "dst_mode": dst_mode,
                "gmt_offset": float(gmt_offset),
                "monitoring_minutes": int(monitoring_minutes),
                "ema_filter_enabled": ema_filter,
                "ema_fast": int(ema_fast),
                "ema_slow": int(ema_slow),
                "sr_filter_enabled": sr_filter,
                "max_spread": float(max_spread),
                "news_filter_enabled": news_filter,
                "max_trades_at_time": int(max_trades_time),
                "max_trades_daily": int(max_trades_daily),
                "atr_period": int(atr_period),
                "atr_timeframe": atr_tf,
                "partial_close": {
                    "enabled": pc_enabled,
                    "trigger_pct": float(pc_trigger),
                    "close_pct": float(pc_close),
                    "move_sl_to_be": pc_be,
                },
                "trailing": {
                    "enabled": tr_enabled,
                    "mode": tr_mode,
                    "trigger_pct": float(tr_trigger),
                    "trail_pct": float(tr_pct),
                },
                "grid": {
                    "enabled": gr_enabled,
                    "levels": int(gr_levels),
                    "distance_pips": float(gr_dist),
                    "distance_multiplier": float(gr_dist_mult),
                    "volume_multiplier": float(gr_vol_mult),
                    "max_grid_lot": float(gr_max_lot),
                },
                "htf_ema_fast": int(htf_ef),
                "htf_ema_slow": int(htf_es),
                "ltf_ema_fast": int(ltf_ef),
                "ltf_ema_slow": int(ltf_es),
                "sideways_atr_mult": float(sw_mult),
                "sideways_candles": int(sw_candles),
                "max_account_equity": float(max_eq),
                "max_daily_dd_pct": float(max_daily_dd),
                "max_overall_dd_pct": float(max_overall_dd),
                "max_queue_size": int(max_q),
                "cooldown_minutes": float(cooldown),
                "signal_expiry_seconds": float(expiry),
            }
            result = api_post("/api/settings", new_settings)
            if result:
                st.success("✅ Settings saved and applied!")
            else:
                st.error("Failed to save settings.")


# ══════════════════════════════════════════════════════════════════════════ #
#  PAGE 5 — TRADE HISTORY                                                   #
# ══════════════════════════════════════════════════════════════════════════ #

with tab5:
    st.markdown("## 📜 Trade History")

    trades_data = api_get("/api/trades?page_size=200", {})
    trades = trades_data.get("trades", []) if trades_data else []
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    open_t = [t for t in trades if t.get("status") == "OPEN"]

    if not trades:
        st.info("No trade history yet. Start the robot to begin trading.")
    else:
        # Summary metrics
        h1, h2, h3, h4, h5 = st.columns(5)
        total_pnl_hist = sum(t.get("pnl", 0) for t in closed)
        wins = [t for t in closed if t.get("pnl", 0) > 0]
        losses = [t for t in closed if t.get("pnl", 0) <= 0]
        gross_win = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        win_rate_hist = len(wins) / len(closed) * 100 if closed else 0.0
        avg_win = gross_win / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0

        h1.metric("Total P&L", f"${total_pnl_hist:,.2f}")
        h2.metric("Win Rate", f"{win_rate_hist:.1f}%")
        h3.metric("Profit Factor", f"{pf:.2f}")
        h4.metric("Avg Win", f"${avg_win:.2f}")
        h5.metric("Avg Loss", f"-${avg_loss:.2f}")

        st.markdown("---")

        col_chart, col_daily = st.columns([3, 2])

        with col_chart:
            # Equity curve
            if closed:
                df_hist = pd.DataFrame(closed)
                df_hist["open_time"] = pd.to_datetime(df_hist["open_time"], unit="s")
                df_hist = df_hist.sort_values("open_time")
                df_hist["cum_pnl"] = df_hist["pnl"].cumsum()

                fig_eq = go.Figure()
                bar_colors = ["#00e676" if p > 0 else "#ff5252" for p in df_hist["pnl"]]
                fig_eq.add_trace(go.Bar(
                    x=df_hist["open_time"],
                    y=df_hist["pnl"],
                    name="Trade P&L",
                    marker_color=bar_colors,
                ))
                fig_eq.add_trace(go.Scatter(
                    x=df_hist["open_time"],
                    y=df_hist["cum_pnl"],
                    name="Cumulative P&L",
                    line=dict(color="#40c4ff", width=2),
                    yaxis="y2",
                ))
                fig_eq.update_layout(
                    title="Trade P&L + Equity Curve",
                    height=350,
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="#fafafa"),
                    xaxis=dict(gridcolor="#1e2130"),
                    yaxis=dict(gridcolor="#1e2130", title="Per-Trade P&L"),
                    yaxis2=dict(overlaying="y", side="right", gridcolor="#1e2130", title="Cumulative"),
                    legend=dict(bgcolor="rgba(30,33,48,0.8)"),
                )
                st.plotly_chart(fig_eq, use_container_width=True)

                # Drawdown chart
                peak = df_hist["cum_pnl"].cummax()
                drawdown = df_hist["cum_pnl"] - peak
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=df_hist["open_time"],
                    y=drawdown,
                    fill="tozeroy",
                    name="Drawdown",
                    line=dict(color="#ff5252"),
                    fillcolor="rgba(255,82,82,0.2)",
                ))
                fig_dd.update_layout(
                    title="Drawdown Chart",
                    height=200,
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font=dict(color="#fafafa"),
                    xaxis=dict(gridcolor="#1e2130"),
                    yaxis=dict(gridcolor="#1e2130", title="DD ($)"),
                )
                st.plotly_chart(fig_dd, use_container_width=True)

        with col_daily:
            # Daily P&L breakdown
            if closed:
                df_daily = pd.DataFrame(closed)
                df_daily["date"] = pd.to_datetime(df_daily["open_time"], unit="s").dt.date
                daily_pnl = df_daily.groupby("date")["pnl"].sum().reset_index()
                daily_pnl.columns = ["Date", "P&L"]
                daily_pnl["Color"] = daily_pnl["P&L"].apply(lambda x: "🟢" if x > 0 else "🔴")

                st.markdown("### Daily P&L")
                st.dataframe(
                    daily_pnl.assign(**{"P&L": daily_pnl["P&L"].map(lambda x: f"${x:+.2f}")}),
                    use_container_width=True,
                    hide_index=True,
                )

                # Direction breakdown
                dir_counts = df_hist["direction"].value_counts().reset_index()
                dir_counts.columns = ["Direction", "Count"]
                fig_pie = go.Figure(data=[go.Pie(
                    labels=dir_counts["Direction"],
                    values=dir_counts["Count"],
                    marker_colors=["#00e676", "#ff5252"],
                    hole=0.4,
                )])
                fig_pie.update_layout(
                    title="BUY vs SELL",
                    height=220,
                    paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # Full trade table
        st.markdown("### All Closed Trades")
        if closed:
            df_table = pd.DataFrame(closed)
            display_cols = ["trade_id", "symbol", "direction", "lot_size",
                            "entry_price", "close_price", "sl", "tp", "pnl",
                            "entry_mode", "status"]
            available = [c for c in display_cols if c in df_table.columns]
            if "open_time" in df_table.columns:
                df_table["open_time"] = pd.to_datetime(df_table["open_time"], unit="s").dt.strftime("%Y-%m-%d %H:%M")

            def pnl_color(val):
                if isinstance(val, (int, float)):
                    return "color: #00e676" if val > 0 else "color: #ff5252"
                return ""

            styled_table = df_table[available].style.applymap(pnl_color, subset=["pnl"] if "pnl" in available else [])
            st.dataframe(styled_table, use_container_width=True, hide_index=True)

        if open_t:
            st.markdown("### Open Trades")
            df_open_t = pd.DataFrame(open_t)
            st.dataframe(df_open_t, use_container_width=True, hide_index=True)

    # Total stats footer
    st.markdown("---")
    st.markdown(f"*Total records: {trades_data.get('total', 0) if trades_data else 0} | "
                f"Displayed: {len(trades)} | Last updated: {datetime.now().strftime('%H:%M:%S')}*")

"""Streamlit application for the FX Option Backtesting Engine."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

from fx_backtester.engine import Backtester
from fx_backtester.strategies.straddle import Straddle
from fx_backtester.strategies.ratio_spread import CallRatioSpread
from fx_backtester.strategies.calendar_spread import CalendarSpread
from fx_backtester.pricing.garman_kohlhagen import garman_kohlhagen
from fx_backtester.analytics.performance import compute_metrics

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FX Option Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

# ── Cached data loading ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading market data…")
def get_backtester() -> Backtester:
    bt = Backtester(DATA_DIR)
    bt.load_data()
    return bt


@st.cache_data(show_spinner=False)
def get_date_range(_bt: Backtester) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = _bt._spot.index
    return dates.min(), dates.max()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

st.sidebar.header("Strategy")
strategy_name = st.sidebar.selectbox(
    "Select strategy",
    ["Straddle", "Call Ratio Spread", "Calendar Spread"],
)

tenor_days = st.sidebar.slider("Primary tenor (days)", 7, 180, 30, step=7)

if strategy_name == "Call Ratio Spread":
    otm_delta = st.sidebar.slider("OTM delta (short leg)", 0.10, 0.40, 0.25, step=0.05)
elif strategy_name == "Calendar Spread":
    far_tenor = st.sidebar.slider("Far tenor (days)", 30, 360, 90, step=30)
    far_tenor = max(far_tenor, tenor_days + 7)

direction = st.sidebar.radio("Direction", ["Long", "Short"], horizontal=True)
direction_int = 1 if direction == "Long" else -1

st.sidebar.header("Backtest Parameters")
notional = st.sidebar.number_input("Notional (domestic ccy)", value=1_000_000, step=100_000)
roll_freq = st.sidebar.selectbox(
    "Roll frequency",
    {"Weekly (Fri)": "W-FRI", "Monthly": "MS", "Quarterly": "QS"},
)

st.sidebar.header("Date Range")
bt = get_backtester()
d_min, d_max = get_date_range(bt)

col_s, col_e = st.sidebar.columns(2)
start_date = col_s.date_input("Start", value=d_min.date(), min_value=d_min.date(), max_value=d_max.date())
end_date = col_e.date_input("End", value=d_max.date(), min_value=d_min.date(), max_value=d_max.date())

run_btn = st.sidebar.button("▶ Run Backtest", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📊 FX Option Backtesting Engine")
st.caption("EUR/NOK · Garman-Kohlhagen pricing · Cubic spline vol surface")

tabs = st.tabs(["🏠 Overview", "📈 Results", "🔬 Greeks P&L", "🛠 Pricer"])

# ── Tab 0: Overview ───────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Architecture")
    col1, col2, col3 = st.columns(3)
    col1.info("**Data Layer**\n\nParquet ingestion → pivot → curve interpolators (linear) + vol surface (cubic spline)")
    col2.info("**Pricing Engine**\n\nGarman-Kohlhagen model with full Greeks: Δ, Γ, ν, Θ, ρ")
    col3.info("**Portfolio**\n\nMulti-leg strategies · Daily delta hedging · Roll scheduling")

    st.subheader("Market Data Preview")
    preview_tab = st.selectbox("Dataset", ["Spot", "Forward Curve", "Interest Rates", "Vol Surface"])

    if preview_tab == "Spot":
        df = bt._spot.reset_index().tail(60)
        fig = px.line(df, x="DATE", y="MID_PRICE", title="EUR/NOK Spot")
        st.plotly_chart(fig, use_container_width=True)
    elif preview_tab == "Forward Curve":
        sample_date = bt._fwd["DATE"].max()
        df = bt._fwd[bt._fwd["DATE"] == sample_date].sort_values("MATURITY")
        fig = px.line(df, x="MATURITY", y="MID_PRICE", title=f"Forward Curve @ {sample_date.date()}")
        st.plotly_chart(fig, use_container_width=True)
    elif preview_tab == "Interest Rates":
        sample_date = bt._rates["DATE"].max()
        df = bt._rates[bt._rates["DATE"] == sample_date].sort_values(["curve_id", "MATURITY"])
        fig = px.line(df, x="MATURITY", y="MID_PRICE", color="curve_id",
                      title=f"Deposit Curves @ {sample_date.date()} (1=domestic, 2=foreign)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        sample_date = bt._vol["DATE"].max()
        df = bt._vol[bt._vol["DATE"] == sample_date]
        sample_mat = df["MATURITY"].iloc[0]
        df2 = df[df["MATURITY"] == sample_mat].sort_values("MID_STRIKE")
        fig = px.line(df2, x="MID_STRIKE", y="MID_PRICE",
                      title=f"Vol Smile @ {sample_date.date()} | Maturity {sample_mat.date()}")
        fig.update_yaxes(title="Implied Vol")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 1: Results ────────────────────────────────────────────────────────────
with tabs[1]:
    if not run_btn:
        st.info("Configure parameters in the sidebar and click **Run Backtest**.")
    else:
        # Build strategy
        if strategy_name == "Straddle":
            strategy = Straddle(tenor_days=tenor_days, direction=direction_int)
        elif strategy_name == "Call Ratio Spread":
            strategy = CallRatioSpread(tenor_days=tenor_days, otm_delta=otm_delta)
        else:
            strategy = CalendarSpread(
                near_tenor_days=tenor_days,
                far_tenor_days=far_tenor,
            )

        freq_map = {"Weekly (Fri)": "W-FRI", "Monthly": "MS", "Quarterly": "QS"}

        with st.spinner("Running backtest…"):
            results = bt.run(
                strategy,
                start=str(start_date),
                end=str(end_date),
                notional=notional,
                roll_freq=freq_map.get(roll_freq, roll_freq),
            )

        pnl = results["daily_pnl"]
        metrics = results["metrics"]
        cumulative = pnl.cumsum()

        # KPI cards
        st.subheader(f"Strategy: {strategy.name} | {direction} | {tenor_days}d tenor")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total P&L", f"{metrics['total_pnl']:,.0f}")
        k2.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
        k3.metric("Sortino Ratio", f"{metrics['sortino']:.2f}")
        k4.metric("Max Drawdown", f"{metrics['max_drawdown']:,.0f}")
        k5.metric("Win Rate", f"{metrics['win_rate']:.1%}")

        k6, k7, k8 = st.columns(3)
        k6.metric("VaR 95%", f"{metrics['var_95']:,.0f}")
        k7.metric("CVaR 95%", f"{metrics['cvar_95']:,.0f}")
        k8.metric("Ann. Vol", f"{metrics['volatility_ann']:,.0f}")

        # Cumulative P&L chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            subplot_titles=["Cumulative P&L", "Daily P&L"])
        fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative.values,
                                 name="Cumulative P&L", fill="tozeroy",
                                 line=dict(color="#00b4d8")), row=1, col=1)
        colors = ["#06d6a0" if v >= 0 else "#ef476f" for v in pnl.values]
        fig.add_trace(go.Bar(x=pnl.index, y=pnl.values, name="Daily P&L",
                             marker_color=colors), row=2, col=1)
        fig.update_layout(height=550, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Return distribution
        fig2 = px.histogram(pnl, nbins=60, title="Daily P&L Distribution",
                            color_discrete_sequence=["#00b4d8"])
        fig2.add_vline(x=metrics["var_95"], line_dash="dash", line_color="red",
                       annotation_text="VaR 95%")
        fig2.add_vline(x=metrics["cvar_95"], line_dash="dot", line_color="orange",
                       annotation_text="CVaR 95%")
        st.plotly_chart(fig2, use_container_width=True)

        # Store results in session state for Greeks tab
        st.session_state["results"] = results

# ── Tab 2: Greeks P&L ─────────────────────────────────────────────────────────
with tabs[2]:
    if "results" not in st.session_state:
        st.info("Run the backtest first.")
    else:
        greeks = st.session_state["results"]["greeks"]

        st.subheader("Greeks Exposure Over Time")
        greek_sel = st.multiselect("Select Greeks", ["delta", "gamma", "vega", "theta"],
                                   default=["delta", "vega"])
        fig = px.line(greeks[greek_sel], title="Greeks Exposure")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Greeks Distribution")
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=["Delta", "Gamma", "Vega", "Theta"])
        for idx, g in enumerate(["delta", "gamma", "vega", "theta"]):
            r, c = divmod(idx, 2)
            fig2.add_trace(go.Histogram(x=greeks[g], name=g, nbinsx=40,
                                        marker_color=["#00b4d8","#06d6a0","#ffd166","#ef476f"][idx]),
                           row=r + 1, col=c + 1)
        fig2.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Single option pricer ───────────────────────────────────────────────
with tabs[3]:
    st.subheader("Garman-Kohlhagen Single Option Pricer")
    pc1, pc2 = st.columns(2)

    with pc1:
        p_S = st.number_input("Spot (S)", value=11.50, step=0.01)
        p_K = st.number_input("Strike (K)", value=11.50, step=0.01)
        p_T = st.number_input("Tenor (days)", value=30, step=1)
        p_sigma = st.slider("Implied vol (%)", 1.0, 60.0, 12.0, step=0.5) / 100

    with pc2:
        p_rd = st.slider("Domestic rate (%)", 0.0, 15.0, 3.0, step=0.25) / 100
        p_rf = st.slider("Foreign rate (%)", 0.0, 15.0, 1.5, step=0.25) / 100
        p_type = st.radio("Option type", ["call", "put"], horizontal=True)

    T_yr = p_T / 365.25
    res = garman_kohlhagen(p_S, p_K, T_yr, p_rd, p_rf, p_sigma, p_type)

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Price", f"{res.price:.5f}")
    g2.metric("Delta", f"{res.delta:.4f}")
    g3.metric("Gamma", f"{res.gamma:.6f}")
    g4.metric("Vega (per 1%)", f"{res.vega:.5f}")
    g5.metric("Theta (per day)", f"{res.theta:.5f}")

    # Vol surface sensitivity
    st.subheader("Sensitivity: Price vs Spot")
    spots = np.linspace(p_S * 0.85, p_S * 1.15, 100)
    prices = [garman_kohlhagen(s, p_K, T_yr, p_rd, p_rf, p_sigma, p_type).price for s in spots]
    deltas = [garman_kohlhagen(s, p_K, T_yr, p_rd, p_rf, p_sigma, p_type).delta for s in spots]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=spots, y=prices, name="Price", line=dict(color="#00b4d8")))
    fig.add_trace(go.Scatter(x=spots, y=deltas, name="Delta", line=dict(color="#06d6a0", dash="dash")),
                  secondary_y=True)
    fig.add_vline(x=p_S, line_dash="dot", line_color="gray", annotation_text="Current S")
    fig.update_xaxes(title_text="Spot")
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Delta", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

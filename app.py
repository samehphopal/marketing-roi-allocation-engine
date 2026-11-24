import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize, curve_fit

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Marketing ROI & Budget Allocation Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        cursor: pointer !important;
    }
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }
    .main {
        background-color: #0b0f17;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .header-box {
        padding: 1.1rem 0 0.9rem 0;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.4rem;
    }
    .header-title {
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.96rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        font-weight: 400;
    }
    .metric-card {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.15rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.72rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.15;
        white-space: nowrap;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    }
    .metric-lift {
        color: #10b981;
        font-weight: 600;
    }
    .recommendation-card {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin-top: 1rem;
    }
    .rec-heading {
        color: #38bdf8;
        font-weight: 700;
        font-size: 0.84rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .rec-body {
        color: #e2e8f0;
        font-size: 0.91rem;
        line-height: 1.5;
        margin: 0;
    }
    .rec-highlight {
        color: #34d399;
        font-weight: 700;
    }
    .glossary-card {
        background: #131b2e;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 16px;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #cbd5e1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 0.2rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------
def fmt_signed_usd(val: float, decimals: int = 0) -> str:
    sign = "+" if val >= 0 else "-"
    return f"{sign}${abs(val):,.{decimals}f}"

def fmt_compact_usd(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"${val / 1_000:.0f}K"
    return f"${val:,.0f}"

# ---------------------------------------------------------
# Saturation Curves, Live CSV Calibration & SLSQP Engine
# ---------------------------------------------------------
DEFAULT_CHANNELS = {
    "Google Search": {"alpha": 9200.0, "beta": 0.0009, "default_spend": 4200.0},
    "Meta Ads": {"alpha": 8500.0, "beta": 0.0011, "default_spend": 2800.0},
    "YouTube Ads": {"alpha": 6800.0, "beta": 0.0013, "default_spend": 1400.0},
    "TikTok Ads": {"alpha": 5800.0, "beta": 0.0016, "default_spend": 1000.0}
}

SCENARIO_PRESETS = {
    "Balanced Growth Mix": {"Google Search": 4200.0, "Meta Ads": 2800.0, "YouTube Ads": 1400.0, "TikTok Ads": 1000.0},
    "Search-Heavy (Oversaturated)": {"Google Search": 6000.0, "Meta Ads": 2200.0, "YouTube Ads": 1000.0, "TikTok Ads": 800.0},
    "Aggressive Launch Push": {"Google Search": 3500.0, "Meta Ads": 4500.0, "YouTube Ads": 2500.0, "TikTok Ads": 1500.0},
}

def channel_revenue(spend, alpha, beta):
    return alpha * np.log(1.0 + beta * np.maximum(spend, 0.0))

def marginal_roas(spend, alpha, beta):
    return (alpha * beta) / (1.0 + beta * np.maximum(spend, 0.0))

def total_portfolio_revenue(spend_vector, channels_dict):
    return sum(
        channel_revenue(spend_vector[idx], params["alpha"], params["beta"])
        for idx, params in enumerate(channels_dict.values())
    )

def generate_sample_csv() -> bytes:
    rng = np.random.default_rng(42)
    rows = []
    for ch, p in DEFAULT_CHANNELS.items():
        for spend in np.linspace(500, 6000, 15):
            noise = rng.normal(1.0, 0.04)
            rev = channel_revenue(spend, p["alpha"], p["beta"]) * noise
            rows.append({"channel": ch, "spend": round(float(spend), 2), "revenue": round(float(rev), 2)})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")

def calibrate_from_csv(df: pd.DataFrame) -> dict:
    df_clean = df.copy()
    df_clean.columns = [c.strip().lower() for c in df_clean.columns]
    required = {"channel", "spend", "revenue"}
    if not required.issubset(set(df_clean.columns)):
        raise ValueError("CSV must contain columns: channel, spend, revenue")

    fitted = {}
    for ch_name, grp in df_clean.groupby("channel"):
        grp = grp.dropna(subset=["spend", "revenue"])
        x = grp["spend"].astype(float).values
        y = grp["revenue"].astype(float).values
        if len(x) < 3 or np.max(x) <= 0:
            continue
        try:
            popt, _ = curve_fit(
                channel_revenue,
                x,
                y,
                p0=[max(y.max(), 1000.0), 0.001],
                bounds=([100.0, 1e-6], [1e7, 0.1]),
                maxfev=5000
            )
            alpha_fit, beta_fit = float(popt[0]), float(popt[1])
        except Exception:
            avg_roas = max(float(y.sum() / max(x.sum(), 1.0)), 0.5)
            alpha_fit, beta_fit = float(y.mean() * 1.5), min(avg_roas / max(y.mean() * 1.5, 1.0), 0.01)

        fitted[str(ch_name)] = {
            "alpha": alpha_fit,
            "beta": beta_fit,
            "default_spend": round(float(np.mean(x)), 2)
        }
    if len(fitted) < 2:
        raise ValueError("Need at least 2 channels with 3+ rows each to run optimization.")
    return fitted

def run_slsqp_optimization(total_budget, channels_dict):
    n_channels = len(channels_dict)
    channel_names = list(channels_dict.keys())
    x0 = np.array([total_budget / n_channels] * n_channels)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - total_budget},)

    min_floor = min(100.0, total_budget / (n_channels * 2.0))
    bounds = [(min_floor, total_budget * 0.70) for _ in channel_names]

    result = minimize(
        lambda x: -total_portfolio_revenue(x, channels_dict),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-7, "maxiter": 500}
    )
    if result.success:
        return result.x
    return np.array([channels_dict[ch]["current_spend"] for ch in channel_names])

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Data Source")
    data_mode = st.radio(
        "Model Input",
        ["Industry Benchmark Data", "Upload Custom CSV"],
        index=0
    )

    active_channels = {k: dict(v) for k, v in DEFAULT_CHANNELS.items()}
    preset_choice = "Balanced Growth Mix"

    if data_mode == "Upload Custom CSV":
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=generate_sample_csv(),
            file_name="sample_marketing_channels.csv",
            mime="text/csv",
            use_container_width=True
        )
        uploaded_file = st.file_uploader("Upload Channel History (.csv)", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                active_channels = calibrate_from_csv(df_upload)
                st.success(f"Calibrated custom curves for {len(active_channels)} channels.")
            except Exception as e:
                st.error(f"CSV Error: {e}")
        else:
            st.caption("Upload a CSV with `channel, spend, revenue` columns to model your own data.")
    else:
        preset_choice = st.selectbox("Scenario Preset", list(SCENARIO_PRESETS.keys()), index=0)
        for ch, preset_val in SCENARIO_PRESETS[preset_choice].items():
            if ch in active_channels:
                active_channels[ch]["default_spend"] = preset_val

    st.markdown("---")
    st.markdown("### ⚙️ Daily Ad Spend")

    current_spends = {}
    for ch_name, params in active_channels.items():
        val = st.number_input(
            f"{ch_name} ($/day)",
            min_value=100.0,
            max_value=50000.0,
            value=float(params["default_spend"]),
            step=100.0,
            format="%.2f",
            key=f"spend_{data_mode}_{preset_choice}_{ch_name}"
        )
        current_spends[ch_name] = val
        active_channels[ch_name]["current_spend"] = val

    st.markdown("---")
    st.markdown("### 💼 Unit Economics")
    gross_margin_pct = st.slider(
        "Product Gross Margin (%)",
        min_value=20,
        max_value=90,
        value=60,
        step=5,
        help="The share of each sale you keep after paying to make or ship the product. For example, at 60% margin you keep $0.60 per $1 sold, so you need $1.67 in sales to pay back every $1 spent on ads."
    )
    gross_margin = gross_margin_pct / 100.0
    breakeven_mroas = 1.0 / gross_margin

    total_daily_spend = sum(current_spends.values())
    st.markdown("---")
    st.metric("Total Daily Budget", f"${total_daily_spend:,.2f}")
    st.caption(f"Monthly: **{fmt_compact_usd(total_daily_spend * 30)}** &bull; Annual: **{fmt_compact_usd(total_daily_spend * 365)}**")
    st.caption(f"Profit Breakeven Line: **${breakeven_mroas:.2f}** per $1")

# ---------------------------------------------------------
# Optimization Execution
# ---------------------------------------------------------
channel_names = list(active_channels.keys())
current_spend_arr = np.array([current_spends[ch] for ch in channel_names])
optimal_spend_arr = run_slsqp_optimization(total_daily_spend, active_channels)

current_daily_rev = total_portfolio_revenue(current_spend_arr, active_channels)
optimal_daily_rev = total_portfolio_revenue(optimal_spend_arr, active_channels)

daily_lift = max(0.0, optimal_daily_rev - current_daily_rev)
annual_lift = daily_lift * 365.0
annual_profit_lift = annual_lift * gross_margin
pct_lift = ((optimal_daily_rev / current_daily_rev) - 1.0) * 100 if current_daily_rev > 0 else 0.0
current_blended_roas = current_daily_rev / total_daily_spend if total_daily_spend > 0 else 0.0
optimal_blended_roas = optimal_daily_rev / total_daily_spend if total_daily_spend > 0 else 0.0

optimal_mroas_list = [
    marginal_roas(optimal_spend_arr[i], active_channels[ch]["alpha"], active_channels[ch]["beta"])
    for i, ch in enumerate(channel_names)
]
equilibrium_mroas = float(np.median(optimal_mroas_list))

# ---------------------------------------------------------
# Main Dashboard UI
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1 class="header-title">Marketing ROI & Budget Allocation Engine</h1>
    <div class="header-subtitle">
        See where your ad dollars are working hardest, and shift budget to drive more sales without spending extra.
    </div>
</div>
""", unsafe_allow_html=True)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Daily Ad Budget</div>
        <div class="metric-value">${total_daily_spend:,.0f}</div>
        <div class="metric-sub">Mo: {fmt_compact_usd(total_daily_spend * 30)} &bull; Yr: {fmt_compact_usd(total_daily_spend * 365)}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Daily Sales</div>
        <div class="metric-value">${current_daily_rev:,.0f}</div>
        <div class="metric-sub">Blended ROAS: <b>{current_blended_roas:.2f}x</b></div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Optimized Daily Sales</div>
        <div class="metric-value">${optimal_daily_rev:,.0f}</div>
        <div class="metric-sub"><span class="metric-lift">+${daily_lift:,.0f}/day</span> ({pct_lift:+.1f}% &bull; {optimal_blended_roas:.2f}x ROAS)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Extra Annual Sales</div>
        <div class="metric-value">+${annual_lift:,.0f}</div>
        <div class="metric-sub"><span class="metric-lift">+${annual_profit_lift:,.0f}/yr</span> Est. Gross Profit ({gross_margin_pct}% Margin)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

with st.expander("📖 How This Tool Works"):
    st.markdown(f"""
- **Every Ad Channel Hits a Ceiling:** Spending more on Google or Meta eventually reaches fewer new buyers, so each extra dollar brings back less than the one before it.
- **Follow the Next Dollar:** Instead of looking at past averages, this tool checks what your **next $1** will earn in each channel and moves budget out of crowded channels into ones with more room to grow.
- **Same Budget, More Profit:** Without adding a dollar to your total spend, the optimizer balances every channel to the same sweet spot (**${equilibrium_mroas:.2f}** per next $1), compared to your **${breakeven_mroas:.2f}** profit breakeven line at a {gross_margin_pct}% margin.
    """)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_opt, tab_curves, tab_audit = st.tabs([
    "🎯 Budget Plan & Strategy",
    "📈 Diminishing Returns Curves",
    "📋 Full Breakdown & Export"
])

CHART_CONFIG = {
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
    "responsive": True
}

# --- TAB 1 ---
with tab_opt:
    col_chart, col_rec = st.columns([1.45, 1.05], gap="large")

    with col_chart:
        st.subheader("Current vs. Recommended Daily Spend")

        realloc_df = pd.DataFrame({
            "Channel": channel_names,
            "Current Spend": current_spend_arr,
            "Optimal Spend": optimal_spend_arr,
            "Spend Delta": optimal_spend_arr - current_spend_arr
        })

        fig_realloc = go.Figure()
        fig_realloc.add_trace(go.Bar(
            x=realloc_df["Channel"],
            y=realloc_df["Current Spend"],
            name="Current Spend ($)",
            marker_color="#38bdf8",
            hovertemplate="<b>%{x}</b><br>Current Spend: $%{y:,.2f}<extra></extra>"
        ))
        fig_realloc.add_trace(go.Bar(
            x=realloc_df["Channel"],
            y=realloc_df["Optimal Spend"],
            name="Recommended Spend ($)",
            marker_color="#10b981",
            hovertemplate="<b>%{x}</b><br>Recommended: $%{y:,.2f}<br>Shift: $%{customdata:+,.2f}<extra></extra>",
            customdata=realloc_df["Spend Delta"]
        ))
        fig_realloc.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(color="#cbd5e1")),
            xaxis=dict(gridcolor="#1e293b", showgrid=False),
            yaxis=dict(gridcolor="#1e293b", title="Daily Budget ($)", showgrid=True),
            bargap=0.25,
            bargroupgap=0.08,
            height=375
        )
        st.plotly_chart(fig_realloc, use_container_width=True, config=CHART_CONFIG)

    with col_rec:
        st.subheader("Where to Shift Your Budget")

        marginal_data = []
        for idx, ch in enumerate(channel_names):
            p = active_channels[ch]
            cur_s = current_spend_arr[idx]
            opt_s = optimal_spend_arr[idx]
            delta_s = opt_s - cur_s
            m_roas = marginal_roas(cur_s, p["alpha"], p["beta"])

            if delta_s > max(50.0, cur_s * 0.05):
                status = "🟢 Scale Up"
            elif delta_s < -max(50.0, cur_s * 0.05):
                status = "🔴 Trim Spend"
            else:
                status = "⚪ On Target"

            marginal_data.append({
                "Channel": ch,
                "Next $1 Return": f"${m_roas:.2f}",
                "Suggested Shift": fmt_signed_usd(delta_s, 0),
                "Action": status
            })

        st.dataframe(pd.DataFrame(marginal_data), use_container_width=True, hide_index=True)

        reallocated_capital = float(np.sum(np.maximum(0.0, current_spend_arr - optimal_spend_arr)))
        if reallocated_capital < 50.0 or annual_lift < 1000.0:
            rec_html = (
                "Your budget split is already <b>on target</b>. "
                f"Every channel is returning roughly <span class='rec-highlight'>${equilibrium_mroas:.2f}</span> on its next dollar, so there is no major upside left from shifting spend."
            )
        else:
            top_cut = channel_names[int(np.argmin(optimal_spend_arr - current_spend_arr))]
            top_scale = channel_names[int(np.argmax(optimal_spend_arr - current_spend_arr))]
            rec_html = (
                f"Moving <span class='rec-highlight'>${reallocated_capital:,.0f}/day</span> (mostly from "
                f"<b>{top_cut}</b> into <b>{top_scale}</b>) unlocks <span class='rec-highlight'>+${annual_lift:,.0f}</span> in extra annual sales "
                f"(<b>+${annual_profit_lift:,.0f}</b> est. gross profit) without increasing your total ad budget."
            )

        st.markdown(f"""
        <div class="recommendation-card">
            <div class="rec-heading"><span>⚡</span> Bottom-Line Takeaway</div>
            <p class="rec-body">{rec_html}</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2 ---
with tab_curves:
    st.subheader("Diminishing Returns by Channel")
    st.caption("Circles show where you spend today. Diamonds show your optimal target.")

    spend_range = np.linspace(100, max(total_daily_spend * 0.85, 8000), 250)
    colors = ["#38bdf8", "#818cf8", "#f43f5e", "#f59e0b", "#10b981", "#a855f7"]
    curve_col1, curve_col2 = st.columns(2, gap="medium")

    with curve_col1:
        fig_sat = go.Figure()
        for idx, (ch, p) in enumerate(active_channels.items()):
            color = colors[idx % len(colors)]
            fig_sat.add_trace(go.Scatter(
                x=spend_range,
                y=channel_revenue(spend_range, p["alpha"], p["beta"]),
                mode="lines",
                name=ch,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{ch}</b><br>Spend: $%{{x:,.0f}}<br>Daily Sales: $%{{y:,.0f}}<extra></extra>"
            ))
            cur_s = current_spend_arr[idx]
            opt_s = optimal_spend_arr[idx]
            fig_sat.add_trace(go.Scatter(
                x=[cur_s],
                y=[channel_revenue(cur_s, p["alpha"], p["beta"])],
                mode="markers",
                marker=dict(size=8, color=color, symbol="circle", line=dict(width=1.5, color="#ffffff")),
                showlegend=False,
                hovertemplate=f"<b>{ch} (Today)</b><br>Spend: $%{{x:,.0f}}<br>Sales: $%{{y:,.0f}}<extra></extra>"
            ))
            fig_sat.add_trace(go.Scatter(
                x=[opt_s],
                y=[channel_revenue(opt_s, p["alpha"], p["beta"])],
                mode="markers",
                marker=dict(size=10, color=color, symbol="diamond", line=dict(width=1.5, color="#10b981")),
                showlegend=False,
                hovertemplate=f"<b>{ch} (Target)</b><br>Spend: $%{{x:,.0f}}<br>Sales: $%{{y:,.0f}}<extra></extra>"
            ))

        fig_sat.update_layout(
            title=dict(text="Total Daily Sales vs. Ad Spend", x=0, y=0.97, font=dict(size=15, color="#f8fafc")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=12),
            xaxis=dict(gridcolor="#1e293b", title="Daily Channel Spend ($)"),
            yaxis=dict(gridcolor="#1e293b", title="Forecasted Daily Sales ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=11, color="#cbd5e1")),
            margin=dict(l=20, r=20, t=80, b=20),
            height=410
        )
        st.plotly_chart(fig_sat, use_container_width=True, config=CHART_CONFIG)

    with curve_col2:
        fig_mroas = go.Figure()
        for idx, (ch, p) in enumerate(active_channels.items()):
            color = colors[idx % len(colors)]
            fig_mroas.add_trace(go.Scatter(
                x=spend_range,
                y=marginal_roas(spend_range, p["alpha"], p["beta"]),
                mode="lines",
                name=ch,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{ch}</b><br>Spend: $%{{x:,.0f}}<br>Next $1 Return: $%{{y:.2f}}<extra></extra>"
            ))
            cur_s = current_spend_arr[idx]
            opt_s = optimal_spend_arr[idx]
            fig_mroas.add_trace(go.Scatter(
                x=[cur_s],
                y=[marginal_roas(cur_s, p["alpha"], p["beta"])],
                mode="markers",
                marker=dict(size=8, color=color, symbol="circle", line=dict(width=1.5, color="#ffffff")),
                showlegend=False,
                hovertemplate=f"<b>{ch} (Today)</b><br>Spend: $%{{x:,.0f}}<br>Next $1: $%{{y:.2f}}<extra></extra>"
            ))
            fig_mroas.add_trace(go.Scatter(
                x=[opt_s],
                y=[marginal_roas(opt_s, p["alpha"], p["beta"])],
                mode="markers",
                marker=dict(size=10, color=color, symbol="diamond", line=dict(width=1.5, color="#10b981")),
                showlegend=False,
                hovertemplate=f"<b>{ch} (Target)</b><br>Spend: $%{{x:,.0f}}<br>Next $1: $%{{y:.2f}}<extra></extra>"
            ))

        # Place the higher line's label above and the lower line's label below to prevent overlap
        eq_pos = "top right" if equilibrium_mroas >= breakeven_mroas else "bottom right"
        be_pos = "bottom right" if equilibrium_mroas >= breakeven_mroas else "top right"

        fig_mroas.add_hline(
            y=equilibrium_mroas,
            line_dash="dot",
            line_color="#10b981",
            annotation_text=f"Sweet Spot (${equilibrium_mroas:.2f})",
            annotation_position=eq_pos,
            annotation_bgcolor="rgba(11, 15, 23, 0.88)",
            annotation_borderpad=3,
            annotation_font=dict(color="#34d399", size=10)
        )
        fig_mroas.add_hline(
            y=breakeven_mroas,
            line_dash="dash",
            line_color="#f87171",
            annotation_text=f"Profit Breakeven (${breakeven_mroas:.2f} @ {gross_margin_pct}% Margin)",
            annotation_position=be_pos,
            annotation_bgcolor="rgba(11, 15, 23, 0.88)",
            annotation_borderpad=3,
            annotation_font=dict(color="#fca5a5", size=10)
        )
        fig_mroas.update_layout(
            title=dict(text="Return on Your Next $1 (Marginal ROAS)", x=0, y=0.97, font=dict(size=15, color="#f8fafc")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=12),
            xaxis=dict(gridcolor="#1e293b", title="Daily Channel Spend ($)"),
            yaxis=dict(gridcolor="#1e293b", title="Return per Next $1 ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(size=11, color="#cbd5e1")),
            margin=dict(l=20, r=20, t=80, b=20),
            height=410
        )
        st.plotly_chart(fig_mroas, use_container_width=True, config=CHART_CONFIG)

# --- TAB 3 ---
with tab_audit:
    st.subheader("Full Channel Breakdown & Export")

    audit_table = pd.DataFrame({
        "Marketing Channel": channel_names,
        "Current Spend ($)": current_spend_arr,
        "Recommended Spend ($)": optimal_spend_arr,
        "Suggested Shift ($)": optimal_spend_arr - current_spend_arr,
        "Current Sales ($)": [channel_revenue(current_spends[ch], active_channels[ch]["alpha"], active_channels[ch]["beta"]) for ch in channel_names],
        "Optimized Sales ($)": [channel_revenue(optimal_spend_arr[i], active_channels[ch]["alpha"], active_channels[ch]["beta"]) for i, ch in enumerate(channel_names)],
        "Current Next $1": [marginal_roas(current_spends[ch], active_channels[ch]["alpha"], active_channels[ch]["beta"]) for ch in channel_names],
        "Target Next $1": optimal_mroas_list
    })

    st.dataframe(
        audit_table.style.format({
            "Current Spend ($)": "${:,.2f}",
            "Recommended Spend ($)": "${:,.2f}",
            "Suggested Shift ($)": lambda v: fmt_signed_usd(v, 2),
            "Current Sales ($)": "${:,.2f}",
            "Optimized Sales ($)": "${:,.2f}",
            "Current Next $1": "${:.2f}",
            "Target Next $1": "${:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    csv_buffer = io.StringIO()
    audit_table.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Recommended Budget Plan (CSV)",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="optimized_marketing_allocation_plan.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown(f"""
    <div class="glossary-card">
        <div style="color: #38bdf8; font-weight: 800; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;">💡 Key Metrics Guide</div>
        <div><strong>• Past Return vs. Next-Dollar Return (mROAS):</strong> An ad channel can have great past results and still waste money today if you have already reached everyone likely to buy. <strong>mROAS</strong> simply asks: <em>"If I put $1 more into this channel right now, how much do I get back?"</em></div>
        <div style="margin-top: 6px;"><strong>• The Sweet Spot (${equilibrium_mroas:.2f} per $1):</strong> You make the most money when every channel gives you the exact same return on its last dollar. If TikTok returns $3 on your next dollar and Google only returns $2, you are leaving money on the table until you shift budget over.</div>
        <div style="margin-top: 6px;"><strong>• The Profit Line (${breakeven_mroas:.2f} at {gross_margin_pct}% Margin):</strong> Making $1 in sales from $1 in ads actually loses money once you pay for the product itself. At a {gross_margin_pct}% margin, every new ad dollar needs to bring back at least <strong>${breakeven_mroas:.2f}</strong> just to break even.</div>
    </div>
    """, unsafe_allow_html=True)
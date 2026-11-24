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
        padding: 1.2rem 0 0.9rem 0;
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
        font-size: 0.95rem;
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
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.15;
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
        font-size: 0.85rem;
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
# Saturation Curves, Real CSV Calibration & SLSQP Engine
# ---------------------------------------------------------
DEFAULT_CHANNELS = {
    "Google Search": {"alpha": 9200.0, "beta": 0.0009, "default_spend": 4200.0},
    "Meta Ads": {"alpha": 8500.0, "beta": 0.0011, "default_spend": 2800.0},
    "YouTube Ads": {"alpha": 6800.0, "beta": 0.0013, "default_spend": 1400.0},
    "TikTok Ads": {"alpha": 5800.0, "beta": 0.0016, "default_spend": 1000.0}
}

SCENARIO_PRESETS = {
    "Balanced E-Commerce ($9.4k/day)": {"Google Search": 4200.0, "Meta Ads": 2800.0, "YouTube Ads": 1400.0, "TikTok Ads": 1000.0},
    "Over-Indexed on Search ($10k/day)": {"Google Search": 6000.0, "Meta Ads": 2200.0, "YouTube Ads": 1000.0, "TikTok Ads": 800.0},
    "Product Launch Push ($12k/day)": {"Google Search": 3500.0, "Meta Ads": 4500.0, "YouTube Ads": 2500.0, "TikTok Ads": 1500.0},
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
        raise ValueError("Need at least 2 channels with 3+ data rows each to run portfolio optimization.")
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
# Sidebar Controls & Ingestion
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Data Source")
    ingestion_mode = st.radio(
        "Calibration Mode",
        ["Benchmark Telemetry", "Upload Custom CSV"],
        index=0
    )

    active_channels = {k: dict(v) for k, v in DEFAULT_CHANNELS.items()}
    preset_choice = "Balanced E-Commerce ($9.4k/day)"

    if ingestion_mode == "Upload Custom CSV":
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
                st.success(f"Fitted saturation curves for {len(active_channels)} channels.")
            except Exception as e:
                st.error(f"CSV Error: {e}")
        else:
            st.caption("Upload a CSV with columns `channel, spend, revenue` to fit custom response curves.")
    else:
        preset_choice = st.selectbox("Budget Scenario Preset", list(SCENARIO_PRESETS.keys()), index=0)
        for ch, preset_val in SCENARIO_PRESETS[preset_choice].items():
            if ch in active_channels:
                active_channels[ch]["default_spend"] = preset_val

    st.markdown("---")
    st.markdown("### ⚙️ Daily Spend Allocation")

    current_spends = {}
    for ch_name, params in active_channels.items():
        val = st.number_input(
            f"{ch_name} ($/day)",
            min_value=100.0,
            max_value=50000.0,
            value=float(params["default_spend"]),
            step=100.0,
            format="%.2f",
            key=f"spend_{ingestion_mode}_{preset_choice}_{ch_name}"
        )
        current_spends[ch_name] = val
        active_channels[ch_name]["current_spend"] = val

    total_daily_spend = sum(current_spends.values())
    st.markdown("---")
    st.metric("Total Daily Budget", f"${total_daily_spend:,.2f}")
    st.caption(f"Annualized Ad Spend: **${total_daily_spend * 365:,.0f}**")

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
pct_lift = ((optimal_daily_rev / current_daily_rev) - 1.0) * 100 if current_daily_rev > 0 else 0.0
current_blended_roas = current_daily_rev / total_daily_spend if total_daily_spend > 0 else 0.0
optimal_blended_roas = optimal_daily_rev / total_daily_spend if total_daily_spend > 0 else 0.0

# ---------------------------------------------------------
# Main Dashboard UI
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <h1 class="header-title">Marketing ROI & Budget Allocation Engine</h1>
    <div class="header-subtitle">
        Marketing Mix Modeling (MMM) &bull; Diminishing Marginal Returns Optimization &bull; Real-Time Budget Reallocation
    </div>
</div>
""", unsafe_allow_html=True)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active Daily Spend</div>
        <div class="metric-value">${total_daily_spend:,.0f}</div>
        <div class="metric-sub">Annualized: ${total_daily_spend * 365:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Forecasted Rev</div>
        <div class="metric-value">${current_daily_rev:,.0f}</div>
        <div class="metric-sub">Blended ROAS: <b>{current_blended_roas:.2f}x</b></div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Optimized Forecasted Rev</div>
        <div class="metric-value">${optimal_daily_rev:,.0f}</div>
        <div class="metric-sub"><span class="metric-lift">+${daily_lift:,.0f}/day</span> ({pct_lift:+.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Annual Revenue Lift</div>
        <div class="metric-value">+${annual_lift:,.0f}</div>
        <div class="metric-sub"><span class="metric-lift">+{optimal_blended_roas - current_blended_roas:.2f}x</span> Blended ROAS Gain</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

with st.expander("📖 How the Optimizer Works (Plain English)"):
    st.markdown("""
- **Why Average ROAS Misleads Teams:** A channel can look strong on average while its *next* dollar spent is already hitting audience saturation.
- **Marginal ROAS (mROAS):** Measures the exact revenue return generated by the **next $1** spent in each channel.
- **Budget-Neutral Optimization (SLSQP):** Without increasing total ad spend, the solver shifts dollars out of lower-mROAS channels into higher-mROAS channels until marginal returns equalize across the portfolio—maximizing total revenue.
    """)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_opt, tab_curves, tab_audit = st.tabs([
    "🎯 Budget Reallocation & Strategy",
    "📈 Saturation & Marginal ROAS Curves",
    "📋 Channel Efficiency Audit"
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
        st.subheader("Current vs. Optimal Daily Spend")

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
            name="Optimized Spend ($)",
            marker_color="#10b981",
            hovertemplate="<b>%{x}</b><br>Optimal Spend: $%{y:,.2f}<br>Shift: $%{customdata:+,.2f}<extra></extra>",
            customdata=realloc_df["Spend Delta"]
        ))
        fig_realloc.update_layout(
            barmode="group",
            plot_bgcolor="#0b0f17",
            paper_bgcolor="#0b0f17",
            font=dict(color="#94a3b8", size=12),
            margin=dict(l=20, r=20, t=35, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#cbd5e1")),
            xaxis=dict(gridcolor="#1e293b", showgrid=False),
            yaxis=dict(gridcolor="#1e293b", title="Daily Budget ($)", showgrid=True),
            bargap=0.25,
            bargroupgap=0.08,
            height=370
        )
        st.plotly_chart(fig_realloc, use_container_width=True, config=CHART_CONFIG)

    with col_rec:
        st.subheader("Marginal Efficiency & Action Plan")

        marginal_data = []
        for idx, ch in enumerate(channel_names):
            p = active_channels[ch]
            cur_s = current_spend_arr[idx]
            opt_s = optimal_spend_arr[idx]
            delta_s = opt_s - cur_s
            m_roas = marginal_roas(cur_s, p["alpha"], p["beta"])

            if delta_s > max(50.0, cur_s * 0.05):
                status = "🟢 Scale Up (High mROAS)"
            elif delta_s < -max(50.0, cur_s * 0.05):
                status = "🔴 Trim Spend (Saturating)"
            else:
                status = "⚪ Near Optimal"

            marginal_data.append({
                "Channel": ch,
                "Next $1 Return": f"${m_roas:.2f}",
                "Suggested Shift": f"{delta_s:+,.0f}",
                "Action": status
            })

        st.dataframe(pd.DataFrame(marginal_data), use_container_width=True, hide_index=True)

        reallocated_capital = float(np.sum(np.maximum(0.0, current_spend_arr - optimal_spend_arr)))
        top_cut = channel_names[int(np.argmin(optimal_spend_arr - current_spend_arr))]
        top_scale = channel_names[int(np.argmax(optimal_spend_arr - current_spend_arr))]

        st.markdown(f"""
        <div class="recommendation-card">
            <div class="rec-heading"><span>⚡</span> Executive Takeaway</div>
            <p class="rec-body">
                Reallocating <span class="rec-highlight">${reallocated_capital:,.0f}/day</span>—primarily shifting spend from <b>{top_cut}</b> into <b>{top_scale}</b>—unlocks <span class="rec-highlight">+${annual_lift:,.0f}</span> in annualized revenue lift at <b>zero additional ad budget</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 2 ---
with tab_curves:
    st.subheader("Channel Response Dynamics (Diminishing Marginal Returns)")
    st.caption("Dots mark your active daily spend. Steeper curves indicate more room to scale efficiently.")

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
                hovertemplate=f"<b>{ch}</b><br>Spend: $%{{x:,.0f}}<br>Daily Rev: $%{{y:,.0f}}<extra></extra>"
            ))
            cur_s = current_spends[ch]
            fig_sat.add_trace(go.Scatter(
                x=[cur_s],
                y=[channel_revenue(cur_s, p["alpha"], p["beta"])],
                mode="markers",
                marker=dict(size=9, color=color, line=dict(width=1.5, color="#ffffff")),
                showlegend=False,
                hoverinfo="skip"
            ))

        fig_sat.update_layout(
            title="Total Daily Revenue vs. Spend",
            plot_bgcolor="#0b0f17",
            paper_bgcolor="#0b0f17",
            font=dict(color="#94a3b8", size=12),
            xaxis=dict(gridcolor="#1e293b", title="Daily Channel Spend ($)"),
            yaxis=dict(gridcolor="#1e293b", title="Forecasted Daily Revenue ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=55, b=20),
            height=380
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
            cur_s = current_spends[ch]
            fig_mroas.add_trace(go.Scatter(
                x=[cur_s],
                y=[marginal_roas(cur_s, p["alpha"], p["beta"])],
                mode="markers",
                marker=dict(size=9, color=color, line=dict(width=1.5, color="#ffffff")),
                showlegend=False,
                hoverinfo="skip"
            ))

        fig_mroas.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#e2e8f0",
            annotation_text="Breakeven ($1.00 mROAS)",
            annotation_position="bottom right",
            annotation_font=dict(color="#94a3b8", size=10)
        )
        fig_mroas.update_layout(
            title="Marginal ROAS (Return on Next $1)",
            plot_bgcolor="#0b0f17",
            paper_bgcolor="#0b0f17",
            font=dict(color="#94a3b8", size=12),
            xaxis=dict(gridcolor="#1e293b", title="Daily Channel Spend ($)"),
            yaxis=dict(gridcolor="#1e293b", title="Marginal Return ($)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=20, r=20, t=55, b=20),
            height=380
        )
        st.plotly_chart(fig_mroas, use_container_width=True, config=CHART_CONFIG)

# --- TAB 3 ---
with tab_audit:
    st.subheader("Full Channel Audit & Export")

    audit_table = pd.DataFrame({
        "Marketing Channel": channel_names,
        "Current Spend ($)": current_spend_arr,
        "Optimal Spend ($)": optimal_spend_arr,
        "Recommended Shift ($)": optimal_spend_arr - current_spend_arr,
        "Current Revenue ($)": [channel_revenue(current_spends[ch], active_channels[ch]["alpha"], active_channels[ch]["beta"]) for ch in channel_names],
        "Optimized Revenue ($)": [channel_revenue(optimal_spend_arr[i], active_channels[ch]["alpha"], active_channels[ch]["beta"]) for i, ch in enumerate(channel_names)],
        "Current mROAS": [marginal_roas(current_spends[ch], active_channels[ch]["alpha"], active_channels[ch]["beta"]) for ch in channel_names],
        "Optimal mROAS": [marginal_roas(optimal_spend_arr[i], active_channels[ch]["alpha"], active_channels[ch]["beta"]) for i, ch in enumerate(channel_names)]
    })

    st.dataframe(
        audit_table.style.format({
            "Current Spend ($)": "${:,.2f}",
            "Optimal Spend ($)": "${:,.2f}",
            "Recommended Shift ($)": "${:+,.2f}",
            "Current Revenue ($)": "${:,.2f}",
            "Optimized Revenue ($)": "${:,.2f}",
            "Current mROAS": "${:.2f}",
            "Optimal mROAS": "${:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    csv_buffer = io.StringIO()
    audit_table.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Optimized Budget Plan (CSV)",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="optimized_marketing_allocation_plan.csv",
        mime="text/csv",
        use_container_width=True
    )
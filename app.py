import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# ----------------- PAGE CONFIGURATION & ENTERPRISE STYLING -----------------
st.set_page_config(
    page_title="Enterprise Marketing Mix & Budget Allocator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: #1e222b;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #8b949e;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-delta-pos {
        color: #3fb950;
        font-size: 13px;
        font-weight: 600;
        margin-top: 4px;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA PIPELINE / BENCHMARK GENERATION -----------------
@st.cache_data
def generate_mmm_dataset():
    np.random.seed(42)
    periods = 180
    dates = pd.date_range(end=pd.Timestamp.today(), periods=periods)
    
    meta = np.random.uniform(1000, 4000, size=periods)
    google = np.random.uniform(1200, 5000, size=periods)
    tiktok = np.random.uniform(500, 2500, size=periods)
    
    meta_rev = 3800 * np.log1p(meta / 650)
    google_rev = 5200 * np.log1p(google / 950)
    tiktok_rev = 3000 * np.log1p(tiktok / 400)
    
    baseline_organic = 2500.0
    noise = np.random.normal(0, 400, size=periods)
    
    gross_revenue = baseline_organic + meta_rev + google_rev + tiktok_rev + noise
    total_spend = meta + google + tiktok
    conversions = np.round(gross_revenue / np.random.uniform(55, 75, size=periods)).astype(int)
    
    df = pd.DataFrame({
        "Date": dates,
        "Meta_Spend": meta,
        "Google_Spend": google,
        "TikTok_Spend": tiktok,
        "Total_Spend": total_spend,
        "Gross_Revenue": gross_revenue,
        "Conversions": conversions
    })
    df["Blended_ROAS"] = df["Gross_Revenue"] / df["Total_Spend"]
    df["CAC"] = df["Total_Spend"] / df["Conversions"]
    return df

# ----------------- MATHEMATICAL RESPONSE & CONVEX OPTIMIZATION -----------------
def channel_response(spend, channel):
    """Logarithmic response modeling diminishing returns per channel."""
    if channel == "Meta":
        return 3800 * np.log1p(spend / 650)
    elif channel == "Google":
        return 5200 * np.log1p(spend / 950)
    elif channel == "TikTok":
        return 3000 * np.log1p(spend / 400)
    return 0.0

def marginal_roas(spend, channel, delta=100.0):
    """First derivative approximation: incremental revenue of next $100 spent."""
    r_current = channel_response(spend, channel)
    r_next = channel_response(spend + delta, channel)
    return (r_next - r_current) / delta

def objective_function(weights):
    """Negative total revenue across channels for SLSQP minimization."""
    m_spend, g_spend, t_spend = weights
    rev = channel_response(m_spend, "Meta") + channel_response(g_spend, "Google") + channel_response(t_spend, "TikTok")
    return -rev

def compute_optimal_allocation(total_budget):
    """Bounded constrained optimization using Sequential Least Squares Programming (SLSQP)."""
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - total_budget})
    bounds = [(250.0, total_budget) for _ in range(3)]
    init_guess = [total_budget / 3.0] * 3
    
    result = minimize(objective_function, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### 📁 Data Source")
data_mode = st.sidebar.radio("Ingestion Mode", ["Benchmark Telemetry", "Upload Custom CSV"], index=0)

if data_mode == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Ad Spend CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = {"Date", "Meta_Spend", "Google_Spend", "TikTok_Spend", "Gross_Revenue", "Conversions"}
        if not required_cols.issubset(df.columns):
            st.sidebar.error("CSV missing required schema. Falling back to benchmark data.")
            df = generate_mmm_dataset()
    else:
        df = generate_mmm_dataset()
else:
    df = generate_mmm_dataset()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Allocation Engine")
input_mode = st.sidebar.radio("Allocation Controller", ["Direct Dollar Input ($)", "Percentage Split (%)"], index=0)

if input_mode == "Direct Dollar Input ($)":
    st.sidebar.markdown("#### Enter Daily Channel Spends")
    in_meta = st.sidebar.number_input("Meta Ads Spend ($)", min_value=100.0, max_value=25000.0, value=2500.0, step=100.0)
    in_google = st.sidebar.number_input("Google Search Spend ($)", min_value=100.0, max_value=25000.0, value=3500.0, step=100.0)
    in_tiktok = st.sidebar.number_input("TikTok Ads Spend ($)", min_value=100.0, max_value=25000.0, value=1200.0, step=100.0)
    
    current_meta, current_google, current_tiktok = in_meta, in_google, in_tiktok
    total_budget = in_meta + in_google + in_tiktok
    st.sidebar.metric("Total Daily Spend", f"${total_budget:,.2f}")
else:
    total_budget = st.sidebar.number_input("Total Daily Budget Cap ($)", min_value=500.0, max_value=50000.0, value=7200.0, step=250.0)
    st.sidebar.markdown("#### Adjust Percentage Allocations")
    pct_m = st.sidebar.slider("Meta Ads (%)", 0, 100, 35)
    pct_g = st.sidebar.slider("Google Search (%)", 0, 100, 45)
    pct_t = st.sidebar.slider("TikTok Ads (%)", 0, 100, 20)
    
    sum_pct = pct_m + pct_g + pct_t
    if sum_pct != 100 and sum_pct > 0:
        st.sidebar.caption(f"⚠️ Proportions sum to {sum_pct}%. Normalizing to 100%.")
        norm_factor = total_budget / sum_pct
        current_meta = pct_m * norm_factor
        current_google = pct_g * norm_factor
        current_tiktok = pct_t * norm_factor
    else:
        current_meta = (pct_m / 100.0) * total_budget
        current_google = (pct_g / 100.0) * total_budget
        current_tiktok = (pct_t / 100.0) * total_budget

# ----------------- ANALYTICS & ARBITRAGE COMPUTATIONS -----------------
ORGANIC_DAILY_REV = 2500.0

current_ad_rev = (channel_response(current_meta, "Meta") + 
                  channel_response(current_google, "Google") + 
                  channel_response(current_tiktok, "TikTok"))
current_total_rev = ORGANIC_DAILY_REV + current_ad_rev
current_roas = current_total_rev / total_budget

opt_meta, opt_google, opt_tiktok = compute_optimal_allocation(total_budget)
opt_ad_rev = -objective_function([opt_meta, opt_google, opt_tiktok])
opt_total_rev = ORGANIC_DAILY_REV + opt_ad_rev
opt_roas = opt_total_rev / total_budget

daily_uplift = opt_total_rev - current_total_rev
annual_uplift = daily_uplift * 365.0
roas_delta = opt_roas - current_roas

# ----------------- MAIN VIEWPORT -----------------
st.title("Enterprise Marketing Mix & Budget Optimization Engine")
st.caption("Media Mix Modeling (MMM) | Non-Linear Response Curves | Bounded SLSQP Convex Optimization")
st.markdown("<hr style='margin-top: 0px; margin-bottom: 24px;'>", unsafe_allow_html=True)

# KPI SCORECARD
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Daily Spend</div>
        <div class="metric-value">${total_budget:,.0f}</div>
        <div style="color: #8b949e; font-size: 12px; margin-top: 4px;">Annualized: ${total_budget*365:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current Forecasted Rev</div>
        <div class="metric-value">${current_total_rev:,.0f}</div>
        <div style="color: #8b949e; font-size: 12px; margin-top: 4px;">ROAS: {current_roas:.2f}x</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Optimized Forecasted Rev</div>
        <div class="metric-value">${opt_total_rev:,.0f}</div>
        <div class="metric-delta-pos">+{daily_uplift:,.0f}/day (+{(daily_uplift/current_total_rev)*100:.1f}%)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Arbitrage Potential</div>
        <div class="metric-value">${annual_uplift:,.0f}</div>
        <div class="metric-delta-pos">+{roas_delta:+.2f}x Blended Lift</div>
    </div>
    """, unsafe_allow_html=True)

# TABBED WORKSPACES
tab_opt, tab_curves, tab_audit = st.tabs([
    "🎯 Budget Arbitrage & Optimization", 
    "📈 Channel Saturation Curves (mROAS)", 
    "🗃️ Historical Data Audit"
])

with tab_opt:
    col_left, col_right = st.columns([3, 2])
    
    comparison_df = pd.DataFrame({
        "Channel": ["Meta Ads", "Google Search", "TikTok Ads"],
        "Current Spend ($)": [current_meta, current_google, current_tiktok],
        "Optimized Spend ($)": [opt_meta, opt_google, opt_tiktok],
        "Variance ($)": [opt_meta - current_meta, opt_google - current_google, opt_tiktok - current_tiktok]
    })
    
    with col_left:
        st.markdown('<div class="section-header">Channel Reallocation Strategy</div>', unsafe_allow_html=True)
        fig_bar = px.bar(
            comparison_df.melt(id_vars="Channel", value_vars=["Current Spend ($)", "Optimized Spend ($)"], var_name="Strategy", value_name="Budget ($)"),
            x="Channel", y="Budget ($)", color="Strategy", barmode="group",
            color_discrete_map={"Current Spend ($)": "#4c78a8", "Optimized Spend ($)": "#54a24b"},
            template="plotly_dark"
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Marginal Efficiency (mROAS)</div>', unsafe_allow_html=True)
        meta_m = marginal_roas(current_meta, "Meta")
        google_m = marginal_roas(current_google, "Google")
        tiktok_m = marginal_roas(current_tiktok, "TikTok")
        
        mroas_df = pd.DataFrame({
            "Channel": ["Meta Ads", "Google Search", "TikTok Ads"],
            "Next $1 Return": [f"${meta_m:.2f}", f"${google_m:.2f}", f"${tiktok_m:.2f}"],
            "Status": [
                "Over-saturated" if meta_m < 0.9 else ("Under-funded" if meta_m > 1.3 else "Efficient"),
                "Over-saturated" if google_m < 0.9 else ("Under-funded" if google_m > 1.3 else "Efficient"),
                "Over-saturated" if tiktok_m < 0.9 else ("Under-funded" if tiktok_m > 1.3 else "Efficient")
            ]
        })
        st.dataframe(mroas_df, use_container_width=True, hide_index=True)
        
        st.info(
            f"**Actionable Recommendation:** Reallocating **${abs(opt_meta - current_meta):,.0f}** away from saturated channels into underfunded avenues unlocks **${annual_uplift:,.0f}** in annual profit with zero extra ad budget."
        )

    st.markdown("---")
    csv_export = comparison_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Optimized Allocation Plan to CSV",
        data=csv_export,
        file_name="optimal_marketing_budget_plan.csv",
        mime="text/csv",
        use_container_width=True
    )

with tab_curves:
    st.markdown('<div class="section-header">Diminishing Marginal Returns Curves</div>', unsafe_allow_html=True)
    
    spend_range = np.linspace(100, 8000, 200)
    meta_curve = [channel_response(s, "Meta") for s in spend_range]
    google_curve = [channel_response(s, "Google") for s in spend_range]
    tiktok_curve = [channel_response(s, "TikTok") for s in spend_range]
    
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=spend_range, y=meta_curve, mode="lines", name="Meta Ads", line=dict(color="#1877F2", width=2.5)))
    fig_curve.add_trace(go.Scatter(x=spend_range, y=google_curve, mode="lines", name="Google Search", line=dict(color="#34A853", width=2.5)))
    fig_curve.add_trace(go.Scatter(x=spend_range, y=tiktok_curve, mode="lines", name="TikTok Ads", line=dict(color="#EE1D52", width=2.5)))
    
    fig_curve.add_trace(go.Scatter(x=[current_meta], y=[channel_response(current_meta, "Meta")], mode="markers", name="Current Meta", marker=dict(size=10, color="#1877F2", symbol="diamond")))
    fig_curve.add_trace(go.Scatter(x=[current_google], y=[channel_response(current_google, "Google")], mode="markers", name="Current Google", marker=dict(size=10, color="#34A853", symbol="diamond")))
    fig_curve.add_trace(go.Scatter(x=[current_tiktok], y=[channel_response(current_tiktok, "TikTok")], mode="markers", name="Current TikTok", marker=dict(size=10, color="#EE1D52", symbol="diamond")))
    
    fig_curve.update_layout(
        template="plotly_dark",
        xaxis_title="Daily Spend ($)",
        yaxis_title="Projected Gross Revenue ($)",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified"
    )
    st.plotly_chart(fig_curve, use_container_width=True)

with tab_audit:
    st.markdown('<div class="section-header">Ingested Daily Campaign Telemetry</div>', unsafe_allow_html=True)
    st.dataframe(
        df.style.format({
            "Meta_Spend": "${:,.2f}",
            "Google_Spend": "${:,.2f}",
            "TikTok_Spend": "${:,.2f}",
            "Total_Spend": "${:,.2f}",
            "Gross_Revenue": "${:,.2f}",
            "Blended_ROAS": "{:.2f}x",
            "CAC": "${:,.2f}"
        }),
        use_container_width=True,
        height=380
    )
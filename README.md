# Multi-Channel Marketing ROI & Budget Allocation Engine

An applied business intelligence and predictive modeling platform designed to evaluate multi-channel ad spend, identify diminishing marginal returns across marketing channels, and optimize budget distributions to maximize Return on Ad Spend (ROAS).

## Overview & Business Context
Modern marketing organizations face non-linear return curves across paid acquisition channels (Meta, Google Ads, TikTok). This engine ingests historical campaign performance data, models non-linear spend-to-revenue relationships, and provides an interactive scenario simulator for marketing budget allocation.

## Technical Architecture & Methodology
- **Data Ingestion & Cleaning:** Engineered structured time-series datasets aggregating daily spend, conversion volumes, and blended gross revenue.
- **Predictive Modeling:** Deployed polynomial regression pipelines utilizing `scikit-learn` to capture channel-level saturation and marginal revenue decay.
- **Interactive Simulation:** Built a reactive frontend in `Streamlit` with `Plotly` integrations, enabling stakeholders to adjust channel allocations dynamically and evaluate real-time forecasted ROAS and CAC impact.

## Key Insights
- Demonstrated that scaling Google Ads beyond $4,200/day yields diminishing marginal ROAS due to query saturation in high-intent keyword auctions.
- Identified higher budget resilience in TikTok campaigns for top-of-funnel conversion generation compared to Meta re-targeting pipelines.

## Installation & Local Execution
```bash
pip install -r requirements.txt
streamlit run app.py
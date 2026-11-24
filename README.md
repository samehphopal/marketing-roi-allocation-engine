# 📈 Marketing ROI & Budget Allocation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sameh-marketing-allocator.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Optimization](https://img.shields.io/badge/Optimization-SciPy%20SLSQP-green.svg)

**[🚀 Launch Live Interactive App](https://sameh-marketing-allocator.streamlit.app)**

An interactive Marketing Mix Modeling (MMM) and ad-budget simulator built to help growth and marketing teams answer one question: **how do we drive more sales and profit without increasing total ad spend?**

---

## Why I Built This

Most marketing teams allocate budgets using **average ROAS**—which quietly wastes money. A channel like Google Search can look great on average even after it hits audience saturation, meaning the *next* dollar spent there earns far less than a dollar shifted into an underfunded channel like YouTube or TikTok.

This tool turns diminishing-returns math into a clean, self-serve dashboard any marketer or executive can use in 30 seconds:
- **See the next-dollar return (mROAS)** for every channel in real time.
- **Reallocate budget automatically** to maximize revenue and gross profit at zero extra spend.
- **Test your own numbers** with one-click presets, a gross-margin profit slider, or your own uploaded CSV.

---

## Core Features

- **4-Channel Mix & One-Click Presets:** Models Google Search, Meta Ads, YouTube Ads, and TikTok Ads with instant scenarios (*Balanced Growth Mix*, *Search-Heavy Saturation*, *Aggressive Launch Push*).
- **Live Custom CSV Curve-Fitting:** Upload any `channel, spend, revenue` CSV (or download the built-in sample CSV) to fit custom diminishing-returns curves on the fly via `scipy.optimize.curve_fit`.
- **True Profit Breakeven:** Adjustable Product Gross Margin slider calculates the real contribution breakeven line ($1 / \text{Margin}$) and estimated annual gross profit lift.
- **Visual Diagnostics:** Interactive Plotly charts comparing current spend (circles) vs. optimal targets (diamonds) against the portfolio sweet spot and profit breakeven line.
- **Plain-English UX & CSV Export:** Built-in plain-English metrics guide for non-technical stakeholders plus one-click CSV budget plan export.

---

## Under the Hood (Math & Tech Stack)

- **Diminishing Returns Curve:** $R_i(x_i) = \alpha_i \cdot \ln(1 + \beta_i \cdot x_i)$
- **Next-Dollar Return (mROAS):** $\frac{dR_i}{dx_i} = \frac{\alpha_i \cdot \beta_i}{1 + \beta_i \cdot x_i}$
- **Optimization:** Constrained convex optimization via `scipy.optimize.minimize` (SLSQP), shifting spend across channels until marginal returns equalize while keeping total budget fixed.
- **Stack:** Python, SciPy, NumPy, Pandas, Plotly, Streamlit, Streamlit Community Cloud.

---

## Run Locally

```bash
git clone [https://github.com/samehphopal/marketing-roi-allocation-engine.git](https://github.com/samehphopal/marketing-roi-allocation-engine.git)
cd marketing-roi-allocation-engine
pip install -r requirements.txt
streamlit run app.py
# 📈 Marketing ROI & Budget Allocation Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sameh-marketing-allocator.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Optimization](https://img.shields.io/badge/Optimization-SciPy%20SLSQP-green.svg)

**[🚀 Launch Live Interactive App](https://sameh-marketing-allocator.streamlit.app)**

An interactive Marketing Mix Modeling (MMM) and budget-allocation platform built with Python, SciPy, Plotly, and Streamlit. The engine models non-linear diminishing returns across paid acquisition channels and solves a constrained convex optimization problem (SLSQP) to maximize revenue and gross profit without increasing total ad spend.

---

## 1. Business Problem & Product Solution

Performance marketing teams routinely allocate budgets based on **average ROAS**, overlooking **diminishing marginal returns**:

- **The Problem:** High-intent channels like Google Search saturate as daily spend climbs—so average ROAS looks healthy even while the *next* dollar deployed delivers lower incremental return than underfunded channels like YouTube or TikTok.
- **The Solution:** This self-serve simulator calculates exact **marginal ROAS (mROAS)** for every channel, benchmarks it against true **gross-margin contribution breakeven**, and automatically computes the budget-neutral spend split that equalizes marginal efficiency across the portfolio.
- **Stakeholder Usability:** Includes one-click executive scenario presets, a product gross-margin slider, live custom CSV curve-fitting (`scipy.optimize.curve_fit`), and one-click CSV budget plan export.

---

## 2. Mathematical Formulation

### Diminishing Returns Response Curve
Channel revenue is modeled using a concave logarithmic saturation curve:

$$R_i(x_i) = \alpha_i \cdot \ln(1 + \beta_i \cdot x_i)$$

Where $x_i$ is daily spend in channel $i$, $\alpha_i$ is the saturation scale parameter, and $\beta_i$ is the channel responsiveness coefficient.

### Marginal ROAS (mROAS) & Margin Breakeven
The instantaneous return on the next $1 spent is the first derivative with respect to spend:

$$\text{mROAS}_i(x_i) = \frac{d R_i(x_i)}{d x_i} = \frac{\alpha_i \cdot \beta_i}{1 + \beta_i \cdot x_i}$$

Given a product gross margin $m \in (0, 1]$, incremental spend remains contribution-positive only while $\text{mROAS}_i(x_i) \ge \frac{1}{m}$.

### Constrained Convex Optimization (SLSQP)
The optimizer maximizes total portfolio revenue subject to strict budget conservation and realistic channel bounds:

$$\max_{\{x_i\}} \sum_{i=1}^{n} R_i(x_i) \quad \text{subject to} \quad \sum_{i=1}^{n} x_i = B_{\text{total}}, \quad x_{i,\min} \le x_i \le x_{i,\max}$$

At optimality, marginal returns across all unbounded channels equalize to a single portfolio equilibrium:

$$\text{mROAS}_1(x_1^*) = \text{mROAS}_2(x_2^*) = \dots = \text{mROAS}_n(x_n^*)$$

---

## 3. Core Features

- **4-Channel Benchmark Mix & Presets:** Out-of-the-box simulation across Google Search, Meta Ads, YouTube Ads, and TikTok Ads with one-click scenario presets (*Balanced Growth Mix*, *Search-Heavy Saturation*, *Aggressive Launch Push*).
- **Live Custom CSV Curve-Fitting:** Upload any historical `channel, spend, revenue` CSV (or download the built-in sample CSV) to fit custom non-linear response curves on the fly using `scipy.optimize.curve_fit`.
- **Unit Economics & Profit Lift:** Interactive gross-margin control calculates true contribution breakeven mROAS and annualized gross profit lift.
- **Visual Saturation Diagnostics:** Interactive Plotly curves comparing current spend (circles) vs. optimal targets (diamonds) against both the optimal equilibrium and margin breakeven thresholds.
- **Executive Audit & CSV Export:** Full channel-by-channel reallocation matrix with one-click CSV export and a plain-English metrics glossary.

---

## 4. Tech Stack

- **Optimization & Modeling:** Python, `scipy.optimize` (`minimize` SLSQP & `curve_fit`), NumPy, Pandas
- **Visualization & UI:** Plotly, Streamlit, Custom CSS, Streamlit Community Cloud

---

## 5. Run Locally

```bash
git clone [https://github.com/samehphopal/marketing-roi-allocation-engine.git](https://github.com/samehphopal/marketing-roi-allocation-engine.git)
cd marketing-roi-allocation-engine
pip install -r requirements.txt
streamlit run app.py
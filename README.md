# Institutional Quantitative Portfolio Management & Risk Engine

An end-to-end Python-based quantitative portfolio optimization, backtesting, and risk management platform built with Streamlit, SciPy, and Plotly.

## Key Features

- **Advanced Allocation Algorithms:**
  - Exact Sharpe Ratio Maximization & Minimum Variance (SciPy SLSQP)
  - Hierarchical Risk Parity (HRP) using Ward's Hierarchical Clustering
  - Equal Risk Contribution / Risk Parity
  - Simplex Dirichlet Monte Carlo Optimization
- **Out-of-Sample Validation:**
  - Chronological train/test split to prevent in-sample overfitting and data snooping bias.
- **Dynamic Backtesting & Real-World Friction:**
  - Dynamic weight drift modeling with user-defined rebalancing frequencies (Monthly, Quarterly, Annual).
  - Transaction cost penalty based on portfolio turnover.
- **Stochastic Projections & Stress Testing:**
  - Geometric Brownian Motion (GBM) Monte Carlo paths (up to 10,000 simulations) with Itô's lemma drift correction.
  - Historical crisis stress testing (2008 GFC, 2020 COVID sell-off, 2022 Inflation bear market, 2023 SVB banking panic).
- **Institutional PDF Reporting:**
  - Automated PDF report generation with corporate layout, asset breakdown, and risk disclaimers via `fpdf2`.

## Mathematical Formulations

### Portfolio Risk

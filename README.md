# Quantitative Portfolio Optimization Engine & Predictive Risk Simulator

A professional-grade financial analytics program written in Python designed to eliminate arbitrary manual asset guessing. This engine combines Modern Portfolio Theory (MPT) with an inflation-adjusted Monte Carlo matrix to engineer mathematically optimal portfolios tailored to client risk boundaries.

## 🛠️ Financial Engineering Capabilities
* **Dynamic Weight Optimization:** Employs `scipy.optimize` sequential least squares programming (SLSQP) to establish exact asset allocations that maximize the Sharpe Ratio.
* **Fisher Effect Adjustment:** Converts nominal historical returns into True Purchasing Power (Real Rate of Return) to combat multi-decade tracking illusions.
* **Risk Probability Matrix:** Simulates 10,000 parallel economic paths utilizing Geometric Brownian Motion across a 30-year operational horizon.
* **Drawdown Vulnerability Metric:** Tracks real-time peak-to-trough pathing to expose the single worst-case market corrections.

## 📈 Optimization & Simulation Output
The tool calculates exact portfolio percentages across historical covariance horizons and renders an interactive visual projection:

* **Optimal Sharpe Weights:** Automated calculation based on the daily historical performance of chosen assets.
* **Real Wealth Projection Box:** Displays live metrics factoring in inflation to track long-term viability.

## 🚀 Technical Requirements & Installation

Execute standard virtual environment assembly and install dependencies:

```bash
pip install -r requirements.txt
```

*Required Dependencies:* `numpy`, `pandas`, `scipy`, `yfinance`, `matplotlib`

## 💻 Program Architecture & Invocation

Run the central optimization suite directly from your terminal:

```bash
python portfolio_optimization.py
```

## 🎓 Academic Integration
Developed to exceed standard portfolio design benchmarks, this engine functions as an analytical alternative framework for corporate asset allocation models and advanced predictive analysis.

import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize as sco
import yfinance as yf
import matplotlib.pyplot as plt

# Web App Title Headers
st.title("Quantitative Portfolio Optimization Engine")
st.write("A predictive risk simulator factoring inflation and asset covariance.")

# Sidebar Controls for the User
st.sidebar.header("Simulation Settings")
initial_investment = st.sidebar.number_input("Initial Investment ($)", value=100000, step=1000)
time_horizon = st.sidebar.slider("Years in Retirement", 5, 50, 30)
inflation_rate = st.sidebar.slider("Annual Inflation Rate", 0.0, 0.10, 0.03, step=0.01)

# Core Math Logic (Same as your script)
assets = ['SPY', 'BND', 'QQQ', 'GLD']
data = yf.download(assets, start="2021-01-01", end="2026-06-01")['Adj Close']
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252  
cov_matrix = returns.cov() * 252    

def portfolio_performance(weights, mean_returns, cov_matrix):
    perf_returns = np.sum(mean_returns * weights)
    perf_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return perf_returns, perf_std

def negate_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(mean_returns)))
initial_guess = len(mean_returns) * [1. / len(mean_returns)]

opts = sco.minimize(negate_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opts['x']
portfolio_return, portfolio_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

# Run Monte Carlo
rrr = ((1 + portfolio_return) / (1 + inflation_rate)) - 1
num_simulations = 5000  # Kept at 5,000 for faster web performance
np.random.seed(42)
sim_results = np.zeros((time_horizon + 1, num_simulations))
sim_results = initial_investment

for t in range(1, time_horizon + 1):
    random_shocks = np.random.normal(0, 1, num_simulations)
    nominal_value = sim_results[t-1] * np.exp((portfolio_return - 0.5 * portfolio_volatility**2) + portfolio_volatility * random_shocks)
    sim_results[t] = nominal_value / (1 + inflation_rate)

years = np.arange(0, time_horizon + 1)
p90 = np.percentile(sim_results, 90, axis=1)
p50 = np.percentile(sim_results, 50, axis=1)
p10 = np.percentile(sim_results, 10, axis=1)

# Drawdowns
peaks = np.maximum.accumulate(sim_results, axis=0)
drawdowns = (sim_results - peaks) / peaks
max_drawdown_overall = np.min(drawdowns)
median_max_drawdown = np.median(np.min(drawdowns, axis=0))

# Display Metrics on Webpage Layout
col1, col2 = st.columns(2)
with col1:
    st.metric("Expected Real Value (Year 30)", f"${p50[-1]:,.0f}")
    st.metric("Real Rate of Return (RRR)", f"{rrr*100:.1f}%")
with col2:
    st.metric("Median Market Correction", f"{median_max_drawdown*100:.1f}%")
    st.metric("Worst-Case Market Crash", f"{max_drawdown_overall*100:.1f}%")

# Generate and Display Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years, p50, color='#1f77b4', linewidth=2, label='Median Outcome')
ax.fill_between(years, p10, p90, color='#1f77b4', alpha=0.15, label='80% Confidence Interval')
ax.set_title('30-Year Portfolio Wealth Projection')
ax.set_xlabel('Years in Retirement')
ax.set_ylabel('Portfolio Value (Real USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
st.pyplot(fig)

# Display Allocation Weights Table
st.subheader("Optimal Asset Allocation Weights")
weight_df = pd.DataFrame({'Asset': assets, 'Optimal Weight': [f"{w*100:.2f}%" for w in optimal_weights]})
st.table(weight_df)

import streamlit as st
import numpy as np
import pandas as pd
import scipy.optimize as sco
import yfinance as yf
import matplotlib.pyplot as plt

# =============================================================================
# WEB INTERFACE CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

st.title("📊 Quantitative Portfolio Optimization Engine")
st.write("A predictive risk simulator factoring inflation and asset covariance for client capstone models.")

# Create a clean sidebar for user interactive sliders
st.sidebar.header("⚙️ Client Simulation Parameters")
initial_investment = st.sidebar.number_input("Initial Investment ($)", value=100000, step=5000)
time_horizon = st.sidebar.slider("Years in Retirement", min_value=5, max_value=50, value=30)
inflation_rate = st.sidebar.slider("Annual Inflation Rate", min_value=0.0, max_value=0.15, value=0.03, step=0.01)

# Add a visual divider line
st.divider()

# =============================================================================
# CORE MATHEMATICAL ENGINE
# =============================================================================
assets = ['SPY', 'BND', 'QQQ', 'GLD']

# Cache the data download so the webpage loads blazing fast for users
@st.cache_data
def load_market_data():
    return yf.download(assets, start="2021-01-01", end="2026-06-01", auto_adjust=False)['Adj Close']

data = load_market_data()
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

# Optimization Matrix
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(mean_returns)))
initial_guess = len(mean_returns) * [1. / len(mean_returns)]

opts = sco.minimize(negate_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opts['x']
portfolio_return, portfolio_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

# Run Monte Carlo Model based on slider numbers
rrr = ((1 + portfolio_return) / (1 + inflation_rate)) - 1
num_simulations = 5000  # Kept at 5k for swift web calculation speed
np.random.seed(42)
sim_results = np.zeros((time_horizon + 1, num_simulations))
sim_results[0] = initial_investment

for t in range(1, time_horizon + 1):
    random_shocks = np.random.normal(0, 1, num_simulations)
    nominal_value = sim_results[t-1] * np.exp((portfolio_return - 0.5 * portfolio_volatility**2) + portfolio_volatility * random_shocks)
    sim_results[t] = nominal_value / (1 + inflation_rate)

years = np.arange(0, time_horizon + 1)
p90 = np.percentile(sim_results, 90, axis=1)
p50 = np.percentile(sim_results, 50, axis=1)
p10 = np.percentile(sim_results, 10, axis=1)

# Advanced Drawdowns
peaks = np.maximum.accumulate(sim_results, axis=0)
drawdowns = (sim_results - peaks) / peaks
max_drawdown_overall = np.min(drawdowns)
median_max_drawdown = np.median(np.min(drawdowns, axis=0))

# =============================================================================
# DATA PRESENTATION LAYOUT
# =============================================================================
# Layout Key Performance Metrics into 4 clean cards side-by-side
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="Expected Year 30 Real Wealth", value=f"${p50[-1]:,.0f}")
with m2:
    st.metric(label="Real Rate of Return (RRR)", value=f"{rrr*100:.1f}%")
with m3:
    st.metric(label="Median Expected Correction", value=f"{median_max_drawdown*100:.1f}%")
with m4:
    st.metric(label="Worst-Case Systemic Crash", value=f"{max_drawdown_overall*100:.1f}%")

st.write("") # Whitespace spacing

# Display Chart and Weights Table side-by-side
chart_col, table_col = st.columns([2, 1])

with chart_col:
    st.subheader("🔮 30-Year Portfolio Wealth Projection Cone")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, p50, color='#1f77b4', linewidth=2.5, label='Median Outcome (Real Purchasing Power)')
    ax.fill_between(years, p10, p90, color='#1f77b4', alpha=0.15, label='80% Confidence Interval Boundaries')
    ax.set_xlabel('Years in Retirement')
    ax.set_ylabel('Portfolio Balance (Real Purchasing Power USD)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper left')
    st.pyplot(fig)

with table_col:
    st.subheader("🎯 Mathematically Optimal Weights")
    st.write("Calculated utilizing MPT historical covariance matching.")
    weight_df = pd.DataFrame({
        'Asset Ticker': assets, 
        'Calculated Allocation Target': [f"{w*100:.2f}%" for w in optimal_weights]
    })
    st.dataframe(weight_df, hide_index=True, use_container_width=True)

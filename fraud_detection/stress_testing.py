# Install required libraries (run in Jupyter or terminal if needed)
# !pip install yfinance numpy pandas matplotlib seaborn arch

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy.optimize import minimize
import os

# Set matplotlib style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')

# Create directory for saving plots
if not os.path.exists("outputs_stress"):
    os.makedirs("outputs_stress")

# 1. Download stock data (20 major stocks)
tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "MA",
    "WMT", "PG", "KO", "PEP", "XOM", "CVX", "JNJ", "PFE", "BAC", "C"
]
data = yf.download(tickers, start="2020-01-01", end="2023-12-31", auto_adjust=True)["Close"]

# Check data integrity
print("First 5 rows of data:\n", data.head())
print("\nMissing values:\n", data.isna().sum())

# Handle missing values
data = data.ffill().dropna()

# 2. Calculate daily returns
returns = data.pct_change().dropna()

# 3. Optimize portfolio weights using Sharpe Ratio
def portfolio_performance(weights, returns, cov_matrix, risk_free_rate=0.01):
    port_returns = returns.dot(weights)
    port_mean = port_returns.mean() * 252
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_mean - risk_free_rate) / port_std
    return -sharpe

cov_matrix = returns.cov()
initial_weights = np.array([1/len(tickers)] * len(tickers))
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))
opt_result = minimize(portfolio_performance, initial_weights, args=(returns, cov_matrix),
                     method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_result.x
print("\nOptimal Portfolio Weights:", dict(zip(tickers, np.round(optimal_weights, 4))))

# Calculate portfolio returns with optimized weights
port_returns = returns.dot(optimal_weights)

# 4. Stress Testing: 2% Interest Rate Shock (assume 5% return drop)
stress_shock = -0.05  # 5% drop due to interest rate increase
stressed_returns = port_returns + stress_shock

# 5. Historical VaR under Normal and Stress Scenarios (1-day and 5-day)
VaR_95_hist_1d_normal = np.percentile(port_returns, 5)
VaR_99_hist_1d_normal = np.percentile(port_returns, 1)
VaR_95_hist_1d_stress = np.percentile(stressed_returns, 5)
VaR_99_hist_1d_stress = np.percentile(stressed_returns, 1)

port_returns_5d = returns.rolling(window=5).sum().dropna().dot(optimal_weights)
stressed_returns_5d = port_returns_5d + stress_shock * 5  # Scaled shock for 5 days
VaR_95_hist_5d_normal = np.percentile(port_returns_5d, 5)
VaR_99_hist_5d_normal = np.percentile(port_returns_5d, 1)
VaR_95_hist_5d_stress = np.percentile(stressed_returns_5d, 5)
VaR_99_hist_5d_stress = np.percentile(stressed_returns_5d, 1)

# 6. Print results
print("\n1-Day VaR Results:")
print(f"Normal Historical VaR 95%: {VaR_95_hist_1d_normal:.4f}")
print(f"Normal Historical VaR 99%: {VaR_99_hist_1d_normal:.4f}")
print(f"Stressed Historical VaR 95%: {VaR_95_hist_1d_stress:.4f}")
print(f"Stressed Historical VaR 99%: {VaR_99_hist_1d_stress:.4f}")
print("\n5-Day VaR Results:")
print(f"Normal Historical VaR 95%: {VaR_95_hist_5d_normal:.4f}")
print(f"Normal Historical VaR 99%: {VaR_99_hist_5d_normal:.4f}")
print(f"Stressed Historical VaR 95%: {VaR_95_hist_5d_stress:.4f}")
print(f"Stressed Historical VaR 99%: {VaR_99_hist_5d_stress:.4f}")

# 7. Plot distribution comparison
plt.figure(figsize=(12, 7))
plt.hist(port_returns, bins=50, alpha=0.5, color="dodgerblue", label="Normal Returns")
plt.hist(stressed_returns, bins=50, alpha=0.5, color="red", label="Stressed Returns")
plt.axvline(VaR_95_hist_1d_normal, color="blue", linestyle="--", linewidth=2, label=f"Normal VaR 95%: {VaR_95_hist_1d_normal:.4f}")
plt.axvline(VaR_95_hist_1d_stress, color="orange", linestyle="--", linewidth=2, label=f"Stressed VaR 95%: {VaR_95_hist_1d_stress:.4f}")
plt.title("Stress Testing: Normal vs. 2% Interest Rate Shock (1-Day)", fontsize=16, weight="bold")
plt.xlabel("Daily Returns", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("outputs_stress/stress_histogram_1d.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 7))
plt.hist(port_returns_5d, bins=50, alpha=0.5, color="dodgerblue", label="Normal 5-Day Returns")
plt.hist(stressed_returns_5d, bins=50, alpha=0.5, color="red", label="Stressed 5-Day Returns")
plt.axvline(VaR_95_hist_5d_normal, color="blue", linestyle="--", linewidth=2, label=f"Normal VaR 95%: {VaR_95_hist_5d_normal:.4f}")
plt.axvline(VaR_95_hist_5d_stress, color="orange", linestyle="--", linewidth=2, label=f"Stressed VaR 95%: {VaR_95_hist_5d_stress:.4f}")
plt.title("Stress Testing: Normal vs. 2% Interest Rate Shock (5-Day)", fontsize=16, weight="bold")
plt.xlabel("5-Day Returns", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("outputs_stress/stress_histogram_5d.png", dpi=300, bbox_inches="tight")
plt.show()
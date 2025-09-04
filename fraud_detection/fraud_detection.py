# Install required libraries (run in Jupyter or terminal if needed)
# !pip install yfinance numpy pandas matplotlib seaborn scikit-learn

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from scipy.optimize import minimize
import os

try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('ggplot')

if not os.path.exists("outputs_fraud"):
    os.makedirs("outputs_fraud")

tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "MA", "WMT", "PG", "KO", "PEP", "XOM", "CVX", "JNJ", "PFE", "BAC", "C"]
data = yf.download(tickers, start="2020-01-01", end="2023-12-31", auto_adjust=True)["Close"]

print("First 5 rows of data:\n", data.head())
print("\nMissing values:\n", data.isna().sum())

data = data.ffill().dropna()
returns = data.pct_change().dropna()

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
opt_result = minimize(portfolio_performance, initial_weights, args=(returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_result.x
print("\nOptimal Portfolio Weights:", dict(zip(tickers, np.round(optimal_weights, 4))))

port_returns = returns.dot(optimal_weights)

# Simulate transaction data with anomalies
np.random.seed(42)
transaction_data = np.random.normal(0, port_returns.std(), 1000)
anomalies = np.random.choice([0, 1], 1000, p=[0.95, 0.05])
transaction_data[anomalies == 1] += np.random.uniform(5, 10, sum(anomalies))

# Fraud detection with Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
predictions = iso_forest.fit_predict(transaction_data.reshape(-1, 1))

# Anomalies are -1, normal are 1
anomaly_indices = np.where(predictions == -1)[0]
print(f"Detected {len(anomaly_indices)} fraudulent transactions out of 1000.")

# Plot results
plt.figure(figsize=(12, 7))
plt.scatter(range(len(transaction_data)), transaction_data, c=predictions, cmap='coolwarm', label="Transactions")
plt.scatter(anomaly_indices, transaction_data[anomaly_indices], c='red', label="Fraudulent", s=100)
plt.title("Fraud Detection in Portfolio Transactions", fontsize=16, weight="bold")
plt.xlabel("Transaction Index", fontsize=14)
plt.ylabel("Return Value", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig("outputs_fraud/fraud_detection.png", dpi=300, bbox_inches="tight")
plt.show()
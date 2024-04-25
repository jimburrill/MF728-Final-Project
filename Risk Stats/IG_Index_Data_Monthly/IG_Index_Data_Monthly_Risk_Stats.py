import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load original portfolio data
portfolio_data = pd.read_csv('IG_Index_Data_Monthly.csv')

# Adjust risk-free rate
risk_free_rate = 3 * portfolio_data['RF']

# Calculate excess returns
excess_returns = portfolio_data['Returns'] - risk_free_rate

# Calculate volatility
volatility = np.std(excess_returns)

# Calculate Sharpe Ratio
sharpe_ratio = np.mean(excess_returns) / volatility

# Calculate VaR
var_5 = np.percentile(excess_returns, 5)
var_1 = np.percentile(excess_returns, 1)

# Calculate max drawdown
max_drawdown = np.min(excess_returns - excess_returns.cummax())

# Add calculated metrics as new columns
portfolio_data['Excess Returns Quarterly'] = excess_returns
portfolio_data['Volatility'] = np.where(portfolio_data.index == 0, volatility, np.nan)
portfolio_data['Sharpe Ratio'] = np.where(portfolio_data.index == 0, sharpe_ratio, np.nan)
portfolio_data['5% VaR'] = np.where(portfolio_data.index == 0, -var_5, np.nan)
portfolio_data['1% VaR'] = np.where(portfolio_data.index == 0, -var_1, np.nan)
portfolio_data['Max Drawdown'] = np.where(portfolio_data.index == 0, max_drawdown, np.nan)

# Save updated DataFrame to CSV
portfolio_data.to_csv('IG_Index_Data_Monthly_Updated.csv', index=False)

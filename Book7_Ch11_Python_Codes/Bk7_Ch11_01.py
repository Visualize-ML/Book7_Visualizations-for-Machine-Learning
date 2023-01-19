

###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# initializations and download results 
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas_datareader as web
tickers = ['TSLA','TSM','COST','NVDA','FB','AMZN','AAPL','NFLX','GOOGL','^GSPC'];
stock_levels_df = web.get_data_yahoo(tickers, start = '2020-08-01', end = '2021-08-01')

print(stock_levels_df.round(2).head())
print(stock_levels_df.round(2).tail())

#%% Plot lineplot of stock prices

# normalize the initial stock price levels to 1
normalized_stock_levels = stock_levels_df['Adj Close']/stock_levels_df['Adj Close'].iloc[0]

fig = sns.relplot(data=normalized_stock_levels,dashes = False,
            kind="line") # , palette="coolwarm"
fig.set_axis_labels('Date','Normalized closing price')

#%% daily log return

daily_log_r = stock_levels_df['Adj Close'].apply(lambda x: np.log(x) - np.log(x.shift(1)))

daily_log_r = daily_log_r.dropna()

#%% Variance-covariance matrix

sns.set_theme(style="white")

daily_log_r = daily_log_r.rename(columns={"^GSPC": "Predictive"})

# Compute the covariance matrix
cov_SIGMA = daily_log_r.cov()

# Set up the matplotlib figure
fig, ax = plt.subplots()


sns.heatmap(cov_SIGMA, cmap="coolwarm",
            square=True, linewidths=.05)
plt.title('Covariance matrix of historical data')

# Compute the covariance matrix
rho_SIGMA = daily_log_r.corr()

# Set up the matplotlib figure
fig, ax = plt.subplots()


sns.heatmap(rho_SIGMA, cmap="coolwarm",
            square=True, linewidths=.05,
            vmax = 1, vmin = 0)
plt.title('Correlation matrix of historical data')

#%% define a stress event s2
SIGMA = cov_SIGMA.to_numpy()

SIGMA_12 = np.array(SIGMA[-1,0:-1])[:, None] 
SIGMA_22 = np.matrix(SIGMA[-1,-1])

s2_up   = np.array([[0.1]])
s2_down = np.array([[-0.05]])

# predicted risk factors s1
s1_up   = SIGMA_12@np.linalg.inv(SIGMA_22)@s2_up
s1_down = SIGMA_12@np.linalg.inv(SIGMA_22)@s2_down

fig, ax = plt.subplots()

plt.bar(tickers[0:-1],np.array(s1_up.flat))
plt.axhline(y=s2_up, color='r', linestyle='-')

fig, ax = plt.subplots()

plt.bar(tickers[0:-1],np.array(s1_down.flat))
plt.axhline(y=s2_down, color='r', linestyle='-')

# verify
b = np.sqrt(np.diag(cov_SIGMA)[0:-1])/np.sqrt(np.diag(cov_SIGMA)[-1])*np.array(rho_SIGMA)[-1,0:-1]
print(b)

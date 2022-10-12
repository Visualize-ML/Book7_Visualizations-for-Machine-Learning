
###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# initializations
import pandas as pd
import pandas_datareader as web
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# load historical price levels for 12 stocks
stock_levels_df = web.get_data_yahoo(['TSLA','WMT','MCD','USB',
                                      'YUM','NFLX','JPM','PFE',
                                      'F','GM','COST','JNJ'],
                                      start = '2020-01-01', end = '2021-05-31')
stock_levels_df.to_csv("12_stocks_level.csv")
stock_levels_df.round(2).head()

# calculate daily returns
daily_returns_df = stock_levels_df['Adj Close'].pct_change()

#%% Lineplot of stock prices

plt.close('all')

sns.set_style("whitegrid") 
sns.set_theme(font = 'Times New Roman')

# normalize the initial stock price levels to 1
normalized_stock_levels = stock_levels_df['Adj Close']/stock_levels_df['Adj Close'].iloc[0]

g = sns.relplot(data=normalized_stock_levels,dashes = False,
                kind="line") # , palette="coolwarm"
g.set_xlabels('Date')
g.set_ylabels('Adjusted closing price')
g.set_xticklabels(rotation=45)

#%% Heatmap of correlation matrix

fig, ax = plt.subplots()
# Compute the correlation matrix
corr_P = daily_returns_df.corr()

sns.heatmap(corr_P, cmap="coolwarm",
            square=True, linewidths=.05, 
            annot=True)


#%% Cluster map based on correlation

g = sns.clustermap(corr_P, cmap="coolwarm", 
                   annot=True)
g.ax_row_dendrogram.remove()



###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets

# load iris data
iris = datasets.load_iris()
X = iris.data[:, [0,1]]

# generate pairwise RBF affinity matrix
rbf_X = rbf_kernel(X)

# heatmap of RBF affinity matrix
fig, ax = plt.subplots()

sns.heatmap(rbf_X, cmap="coolwarm",
            square=True, linewidths=.05)

# lower triangle for heatmap of RBF affinity matrix
fig, ax = plt.subplots()

mask = np.triu(np.ones_like(rbf_X, dtype=bool))

sns.heatmap(rbf_X, cmap="coolwarm",
            mask = mask,
            square=True, linewidths=.05)

# Cluster map based on affinity matrix

fig, ax = plt.subplots()

g = sns.clustermap(rbf_X, cmap="coolwarm")
g.ax_row_dendrogram.remove()

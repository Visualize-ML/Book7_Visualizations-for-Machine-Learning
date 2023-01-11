

###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans

# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X_train = iris.data[:, :2]

# Vector of labels
y_train = iris.target

distortions = []
for i in range(1, 11):
    
    # GK-Means
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0)
    # train the parameters
    km.fit(X_train)
    distortions.append(km.inertia_)

fig, ax = plt.subplots()
plt.plot(range(1, 11), distortions, marker='x')
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
ax.set_xticks(range(1, 11))
plt.show()

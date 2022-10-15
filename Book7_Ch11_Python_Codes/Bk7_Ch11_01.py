

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

# Create color maps
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.

cmap_light = ListedColormap(rgb)

# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X_train = iris.data[:, :2]

# Vector of labels
y_train = iris.target

# GK-Means
kmeans = KMeans(n_clusters=3)

# train the parameters
kmeans.fit(X_train)

# Generate mesh
plot_step = 0.02
xx, yy = np.meshgrid(np.arange(4, 8+plot_step, plot_step),
                     np.arange(1.5, 4.5+plot_step, plot_step))

# predict clusters
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()

# plot regions
plt.contourf(xx, yy, Z, cmap=cmap_light)

# plot sample data
plt.scatter(x=X_train[:, 0], y=X_train[:, 1], color=np.array([0, 68, 138])/255., alpha=1.0, 
                linewidth = 1, edgecolor=[1,1,1])

# plot decision boundaries
plt.contour(xx, yy, Z, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)

# plot centroids
centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, linewidths=1.5,
            color="k")

ax.set_xticks(np.arange(4, 8.5, 0.5))
ax.set_yticks(np.arange(1.5, 5, 0.5))
ax.set_xlim(4, 8)
ax.set_ylim(1.5, 4.5)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_aspect('equal')
plt.show()

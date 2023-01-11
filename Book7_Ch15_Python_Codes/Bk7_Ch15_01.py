

###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap

# Create color maps
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.

cmap_light = ListedColormap(rgb)

iris = load_iris()
X = iris.data[:, :2]

clustering_algorithms = (
    ('Single linkage', 'single'),
    ('Average linkage', 'average'),
    ('Complete linkage', 'complete'),
    ('Ward linkage', 'ward'),
)

for name, method in clustering_algorithms:

    # visualization
    fig, ax = plt.subplots()
    
    # plot dendrogram
    plt.title(name)
    dend = dendrogram(linkage(X, method = method))
    
    # Agglomerative Clustering
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage=method)
    
    cluster.fit(X)
    # Generate mesh
    plot_step = 0.02
    xx, yy = np.meshgrid(np.arange(4, 8+plot_step, plot_step),
                         np.arange(1.5, 4.5+plot_step, plot_step))
    
    # predict clusters
    Z = cluster.fit_predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots()
    plt.title(name)
    # plot regions
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    
    # plot sample data
    plt.scatter(x=X[:, 0], y=X[:, 1], color='r', alpha=1.0, 
                    linewidth = 1, edgecolor=[1,1,1])
    
    # plot decision boundaries
    plt.contour(xx, yy, Z, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)
    
    ax.set_xticks(np.arange(4, 8.5, 0.5))
    ax.set_yticks(np.arange(1.5, 5, 0.5))
    ax.set_xlim(4, 8)
    ax.set_ylim(1.5, 4.5)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_aspect('equal')
    plt.show()

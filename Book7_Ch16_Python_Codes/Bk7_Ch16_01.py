

###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)

n_samples = 500;
# number of sample data

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
dataset = noisy_circles;

X, y = dataset

# normalize dataset
X = StandardScaler().fit_transform(X)

for eps in np.array([0.1,0.2,0.4,0.6]):
    
    dbscan = cluster.DBSCAN(eps=eps,min_samples=10)

    y_pred = dbscan.fit_predict(X)

    fig, ax = plt.subplots()

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y_pred) + 1))))
    # add black color for outliers
    colors = np.append(colors, ["#000000"])
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    # colors[-1], black, 
    # noisy samples are given the label -1, the last element in an array
    
    plt.title('eps = %0.2f' % eps)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.axis('equal')

plt.show()

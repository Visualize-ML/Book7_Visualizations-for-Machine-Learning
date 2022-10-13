

###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import svm, datasets
from matplotlib import cm

def make_meshgrid(x, y, h=.02):

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# Generate data

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [linearly_separable,
            make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1)]

#%% iterate over datasets

for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)

    C = 3  # SVM regularization parameter

    models = (svm.SVC(kernel='linear', C=C),
              svm.SVC(kernel='poly', degree=2, gamma='auto', C=C),
              svm.SVC(kernel='poly', degree=3, gamma='auto', C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='sigmoid', gamma=0.5, C=C))

    models = (clf.fit(X, y) for clf in models)
    
    # title for the plots
    titles = ('linear',
              'Polynomial, d = 2',
              'Polynomial, d = 3',
              'RBF',
              'Sigmoid')

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    # iterate over models
    for clf, title in zip(models, titles):
        
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        Z_0 = clf.decision_function(X);
        
        fig = plt.figure()
        ax = fig.add_subplot(1,2,1)
        
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='w')

        x1_sp_vec = clf.support_vectors_[:, 0];
        x2_sp_vec = clf.support_vectors_[:, 1];
        
        plt.scatter(x1_sp_vec, x2_sp_vec, s=50,
                    facecolors='none', edgecolors='k')
        plt.contour(xx, yy, -Z, levels = 30, cmap=plt.cm.RdBu,linewidths = 0.5)
        plt.contour(xx, yy, -Z, [0], alpha = 1, colors='k',linewidths = 1.25)
        plt.contour(xx, yy, -Z, [-1,1], alpha = 1, colors='k',linewidths = 1.25,linestyles = '--')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xticks(np.linspace(xx.min(), xx.max(), 5))
        ax.set_yticks(np.linspace(yy.min(), yy.max(), 5))
        ax.set_title(title)
        plt.tight_layout()
        plt.axis('square')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        
        ax = fig.add_subplot(1,2,2, projection='3d')
        norm = plt.Normalize(-Z.max(),-Z.min())
        colors = cm.coolwarm(norm(-Z))
        
        surf = ax.plot_surface(xx, yy, -Z,linewidths = 0.25,
                               facecolors=colors, shade=False,rstride=10,cstride=10)
        surf.set_facecolor((0,0,0,0))

        # ax.contour3D(xx, yy, -Z, levels = 30, cmap=plt.cm.RdBu,linewidths = 0.5)
        ax.contour3D(xx, yy, -Z, [0], colors='k',linewidths = 1.25)
        ax.contour3D(xx, yy, -Z, [-1,1], colors='k',linewidths = 1.25,linestyles = '--')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.tight_layout()
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(np.linspace(xx.min(), xx.max(), 5))
        ax.set_yticks(np.linspace(yy.min(), yy.max(), 5))
        ax.set_zticks(np.linspace(-Z.max(), -Z.min(), 5))



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
from sklearn.mixture import GaussianMixture

# Create color maps
rgb = [[255, 238, 255],  # red
       [228, 228, 228],  # blue
       [219, 238, 244]]  # black
rgb = np.array(rgb)/255.

cmap_light = ListedColormap(rgb)
cmap_bold = [[255, 51, 0], [0, 153, 255],[138,138,138]]
cmap_bold = np.array(cmap_bold)/255.


def make_ellipses(gmm, ax):
    for n in range(0,3):
        
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        
        # eigen decomposition of covariance matrix
        v_, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        
        # OR use SVD
        U, s, Vt = np.linalg.svd(covariances)
        major, minor = 2 * np.sqrt(s)
        
        print(gmm.covariance_type)
        print('=== major ==='); print(major)
        print('=== minor ==='); print(minor)
        
        # rotating angle of the ellipse
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  
        # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v_)
        # length of the major/minor axis
        
        for scale in np.array([1, 0.8, 0.6, 0.4, 0.2]):
            
            # center of a component
            plt.plot(gmm.means_[n, 0],gmm.means_[n, 1],
                     color = 'k',marker = 'x',markersize = 10)
            
            # Plot vectors, i.e., directions of minor and major axis
            plt.quiver(gmm.means_[n,0],gmm.means_[n,1],
                       w[0,0], w[1,0], scale = 5/minor)
            
            plt.quiver(gmm.means_[n,0],gmm.means_[n,1], 
                       w[0,1], w[1,1], scale = 5/major)

            # plot five layers of ellipse
            ell = mpl.patches.Ellipse(gmm.means_[n, :2], scale*minor, scale*major,
                                      180 + angle, color=rgb[n,:])
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ell.set_edgecolor(cmap_bold[n,:])
            ax.add_artist(ell)
            plt.show()


# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X_train = iris.data[:, :2]

# Vector of labels
y_train = iris.target

# GMMs using four types of covariances.
estimators = {cov_type: GaussianMixture(n_components=3,
              covariance_type=cov_type, max_iter=20, random_state=0)
              for cov_type in ['tied', 'spherical', 'diag', 'full']} 

for index, (name, gmm) in enumerate(estimators.items()):
    
    # visualization
    fig, ax = plt.subplots()
    plt.gca().set_adjustable("box")
    
    # train the parameters of GMM
    gmm.fit(X_train)
    
    # scatter plot the sample data
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=iris.target_names[y_train],
                    palette=cmap_bold, alpha=1.0, 
                    linewidth = 1, edgecolor=[1,1,1])
    
    # plot ellipses
    make_ellipses(gmm, ax)

    plot_step = 0.02
    xx, yy = np.meshgrid(np.arange(4, 8, plot_step),
                         np.arange(1.5, 4.5, plot_step))
    plt.title(name)
    ax.set_xticks(np.arange(4, 8.5, 0.5))
    ax.set_yticks(np.arange(1.5, 5, 0.5))
    ax.set_xlim(4, 8)
    ax.set_ylim(1.5, 4.5)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_aspect('equal')

    fig, ax = plt.subplots()
    
    # predict clusters
    Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # plot regions
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # plot decision boundaries
    plt.contour(xx, yy, Z, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)
    
    sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=iris.target_names[y_train],
                    palette=cmap_bold, alpha=1.0, 
                    linewidth = 1, edgecolor=[1,1,1])

    plt.title(name)
    ax.set_xticks(np.arange(4, 8.5, 0.5))
    ax.set_yticks(np.arange(1.5, 5, 0.5))
    ax.set_xlim(4, 8)
    ax.set_ylim(1.5, 4.5)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_aspect('equal')
    plt.show()

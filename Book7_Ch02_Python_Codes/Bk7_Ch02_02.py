

###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from matplotlib.widgets import Slider, Button
from matplotlib.font_manager import FontProperties

# Self-defined utility functions

def knn(k_NN,X,y,xx1,xx2):

    # kNN classification, weight = uniform
    clf = neighbors.KNeighborsClassifier(k_NN)
    
    # Fit the data
    clf.fit(X, y)
    
    # query points
    q = np.c_[xx1.ravel(), xx2.ravel()];
    # numpy.c_() Stack 1-D arrays as columns into a 2-D array.
    # numpy.ravel() Return a contiguous flattened array
    
    # Predict; query points are the meshgrid points
    y_predict = clf.predict(q)
    
    # Put the result into a color plot
    y_predict = y_predict.reshape(xx1.shape)

    return y_predict


def plot_contour(X,y,xx1,xx2,y_predict,cmap_light,cmap_bold):

    fig, ax = plt.subplots()

    # plot decision regions
    cntr1 = ax.contourf(xx1, xx2, y_predict, cmap=cmap_light)
    
    # plot decision boundaries
    cntr2 = ax.contour(xx1, xx2, y_predict, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)

    # Plot data points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels[y],
                       palette=cmap_bold, alpha=1.0, 
                       linewidth = 1, edgecolor=[1,1,1],ax = ax)

    plt.axis('equal')
    plt.show()
    
    return fig, ax

#%% Main function

plt.close('all')

# Create color maps
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.

cmap_light = ListedColormap(rgb)
cmap_bold = [[255, 51, 0], [0, 153, 255],[138,138,138]]
cmap_bold = np.array(cmap_bold)/255.

# number of nearest neighbors, initial input
k_NN_0 = 2;

# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X = iris.data[:, :2]
y = iris.target
labels = iris.target_names;

# generate mesh
h = .02  # step size in the mesh
x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))

# kNN predict using utility function
y_predict = knn(k_NN_0,X,y,xx1,xx2)


# Interactive plot
C_plot,C_ax = plot_contour(X,y,xx1,xx2,y_predict,cmap_light,cmap_bold)

# plot a axis for slider
axcolor = 'lightgoldenrodyellow'

ax_k_NN = plt.axes([0.2, 0.9, 0.5, 0.03], facecolor=axcolor)
#  [left, bottom, width, height]

s_k_NN = Slider(ax_k_NN, 'k-NN', 2, 60, valinit=k_NN_0, valstep=2)
# Define a slider, value in range of [2, 60], step size = 2

# function to update decision boundary

def update(val):

    k_NN = s_k_NN.val
    new_y_predict = knn(k_NN,X,y,xx1,xx2)
    C_ax.cla()
    C_ax.contourf(xx1, xx2, new_y_predict, cmap=cmap_light)
    C_ax.contour( xx1, xx2, new_y_predict, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)
    # Plot data points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels[y],
                       palette=cmap_bold, alpha=1.0, 
                       linewidth = 1, edgecolor=[1,1,1],ax = C_ax)

    # Figure decorations
    C_ax.set_xlim(xx1.min(), xx1.max())
    C_ax.set_ylim(xx2.min(), xx2.max())
    
    C_ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    C_ax.set_xlabel(iris.feature_names[0])
    C_ax.set_ylabel(iris.feature_names[1])

    plt.axis('equal')
    plt.show()


s_k_NN.on_changed(update)

# add a reset button
reset_ax = plt.axes([0.8, 0.90, 0.075, 0.03])
button = Button(reset_ax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_k_NN.reset()

button.on_clicked(reset)

# Figure decorations
C_ax.set_xlim(xx1.min(), xx1.max())
C_ax.set_ylim(xx2.min(), xx2.max())

C_ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
C_ax.set_xlabel(iris.feature_names[0])
C_ax.set_ylabel(iris.feature_names[1])

plt.axis('equal')
plt.show()

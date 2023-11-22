import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from matplotlib.widgets import Slider, Button
from matplotlib.font_manager import FontProperties
import streamlit as st

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

with st.sidebar:
    st.title('kNN分类')
    k_NN_0 = st.slider('k', 1,20,5,1)

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

    
    ax.set_xlim(xx1.min(),xx1.max())
    ax.set_ylim(xx2.min(),xx2.max())

    
    return fig, ax

#%% Main function

# plt.close('all')

# Create color maps
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.

cmap_light = ListedColormap(rgb)
cmap_bold = [[255, 51, 0], [0, 153, 255],[138,138,138]]
cmap_bold = np.array(cmap_bold)/255.

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

C_plot,C_ax = plot_contour(X,y,xx1,xx2,y_predict,cmap_light,cmap_bold)

st.pyplot(C_plot)

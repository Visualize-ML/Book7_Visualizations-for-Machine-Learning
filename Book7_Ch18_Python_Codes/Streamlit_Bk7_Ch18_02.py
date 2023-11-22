import streamlit as st
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np

with st.sidebar:
    st.title('核主成分分析')
    gamma = st.slider('Gamma', 0.0, 2.0, 0.01)
    
    
# 装饰
import matplotlib.pyplot as plt
p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

X_original, y = make_circles(n_samples=200, factor=0.3, noise=0.05, random_state=0)


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_original)

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1, 2, 1)

ax.scatter(X[:, 0], X[:, 1], 
           c=y, cmap = 'cool', 
           edgecolors = ['k'], alpha = 0.5)
ax.set_xlabel("STD feature 1")
ax.set_ylabel("STD feature 2")

from sklearn.decomposition import KernelPCA
SK_PCA = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
SK_PC_X = SK_PCA.fit_transform(X)

ax = fig.add_subplot(1, 2, 2)

ax.scatter(SK_PC_X[:, 0], SK_PC_X[:, 1], 
           c=y, cmap = 'cool', 
           edgecolors = ['k'], alpha = 0.5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
st.pyplot(fig)

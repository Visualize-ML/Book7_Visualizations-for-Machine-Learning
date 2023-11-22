import streamlit as st
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np

with st.sidebar:
    st.title('核主成分分析')
    gamma = st.slider('Gamma', 0.0, 30.0, 0.0, 0.2)
    
    
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

np.random.seed(0)
# 大球
num = 500
theta = np.random.uniform(0, np.pi*1, num)
phi = np.random.uniform(0, np.pi*2, num)
r = 10
x1_big = r*np.sin(theta)*np.cos(phi)
x2_big = r*np.sin(theta)*np.sin(phi)
x3_big = r*np.cos(theta)

X_big = np.column_stack((x1_big,x2_big,x3_big))
y_big = np.ones(len(X_big))

# 小球
num = 500
theta = np.random.uniform(0, np.pi*1, num)
phi = np.random.uniform(0, np.pi*2, num)
r = 5
x1_small = r*np.sin(theta)*np.cos(phi)
x2_small = r*np.sin(theta)*np.sin(phi)
x3_small = r*np.cos(theta)

X_small = np.column_stack((x1_small,x2_small,x3_small))
y_smal = np.zeros(len(X_small))

X_original = np.row_stack((X_big,X_small))
y = np.concatenate((y_big,y_smal))


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X_original)

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(X[:,0],X[:,1],X[:,2], 
           c = y, cmap = 'cool',
           edgecolors = ['k'], alpha = 0.5)

ax.set_proj_type('ortho')
ax.view_init(azim=-120, elev=30) 
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))


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

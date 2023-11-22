import matplotlib.pyplot as plt
import numpy as np
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
    st.title('SVD降维图片')
    rank = st.slider('rank', 1, 20, 1, 1)

# Load image
# img = plt.imread("iris_photo.jpg")

# # Donwsample and encode RGBa image as matrix of intensities, X
# DOWNSAMPLE = 4
# R = img[::DOWNSAMPLE, ::DOWNSAMPLE, 0]
# G = img[::DOWNSAMPLE, ::DOWNSAMPLE, 1]
# B = img[::DOWNSAMPLE, ::DOWNSAMPLE, 2] 
# X = 0.2989 * R + 0.5870 * G + 0.1140 * B

from skimage import color
from skimage import io

X = color.rgb2gray(io.imread('iris_photo.jpg'))
# DOWNSAMPLE = 5
# X = X[::DOWNSAMPLE, ::DOWNSAMPLE]
# Calculate the rank of the data matrix, X

# Run SVD on Image
U, S, V = np.linalg.svd(X)

n_components = len(S)
component_idx = range(1,  n_components + 1)

lambda_i = np.square(S)/(X.shape[0] - 1)
# approximation, given that X is not centered

#%% Image Reconstruction

# Reconstruct image with increasing number of singular vectors/values


# Reconstructed Image
X_reconstruction = U[:, :rank] * S[:rank] @ V[:rank,:]

fig, axs = plt.subplots(1, 2)
axs[0].imshow(X_reconstruction, cmap='gray')
axs[0].set_title('X_reproduced with ' + str(rank) + ' PCs')

## Reconstruction error

axs[1].imshow(X - X_reconstruction, cmap='gray')
axs[1].set_title('Error')

st.pyplot(fig)
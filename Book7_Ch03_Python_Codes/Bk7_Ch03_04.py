
###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.spatial import distance
import numpy as np
from numpy.linalg import inv

# Variance-covariance matrix
SIGMA = np.array([[2, 1], [1, 2]])

q   = [0, 0];       # query point
x_1 = [-3.5, -4];   # data point 1
x_2 = [2.75, -1.5]; # data point 1

# Calculate Mahal distances

d_1 = distance.mahalanobis(q, x_1, inv(SIGMA))
d_2 = distance.mahalanobis(q, x_2, inv(SIGMA))

# Note1: the output of the function is Mahal distance, not Mahal distance squared
# Note2: input is the inverse of the covariance matrix


###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.spatial import distance
import numpy as np

# Variance-covariance matrix
SIGMA = np.array([[2, 1], [1, 2]])

q   = [0, 0];       # query point
x_1 = [-3.5, -4];   # data point 1
x_2 = [2.75, -1.5]; # data point 1

# Calculate standardized Euclidean distances

d_1 = distance.seuclidean(q, x_1, np.diag(SIGMA))
d_2 = distance.seuclidean(q, x_2, np.diag(SIGMA))

# Note1: V is an 1-D array of component variances

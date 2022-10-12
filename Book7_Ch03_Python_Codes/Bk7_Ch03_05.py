
###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.spatial import distance
from sklearn.metrics import pairwise_distances

# Sample data points
X = [[-5, 0], [4, 3], [3, -4]]

# Query point
q = [[0, 0]]

# Compute the City Block (Manhattan) distance.
d_1 = distance.cityblock(q, X[0])
d_2 = distance.cityblock(q, X[1])
d_3 = distance.cityblock(q, X[2])

# pairwise distances between rows of X and q
dst_pairwise_X_q = pairwise_distances(X, q, metric='cityblock')
print('Pairwise City Block distances between X and q')
print(dst_pairwise_X_q)

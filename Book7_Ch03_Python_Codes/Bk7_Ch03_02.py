

###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from sklearn.metrics.pairwise import euclidean_distances

# Sample data points
X = [[-5, 0], [4, 3], [3, -4]]

# Query point
q = [[0, 0]]

# pairwise distances between rows of X and q
dst_pairwise_X_q = euclidean_distances(X, q)
print('Pairwise distances between X and q')
print(dst_pairwise_X_q)

# pairwise distances between rows of X and itself
dst_pairwise_X_X = euclidean_distances(X, X)
print('Pairwise distances between X and X')
print(dst_pairwise_X_X)

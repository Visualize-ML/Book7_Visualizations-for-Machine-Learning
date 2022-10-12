

###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

from scipy.spatial import distance
import numpy as np

x_i = (0, 0, 0) # data point
q   = (4, 8, 6) # query point

# calculate Euclidean distance
dst_1 = distance.euclidean(x_i, q)

dst_2 = np.linalg.norm(np.array(x_i) - np.array(q))

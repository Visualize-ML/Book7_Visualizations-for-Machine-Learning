
###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
from scipy.spatial import distance
x = np.array([[8, 2]])
q = np.array([[7, 9]])

d_x_q = distance.correlation(x,q)
print(d_x_q)

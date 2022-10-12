
###############
# Authored by Weisheng Jiang
# Book 7  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

x = np.array([[8, 2]])
q = np.array([[7, 9]])

k_x_q = cosine_similarity(x,q)
print(k_x_q)

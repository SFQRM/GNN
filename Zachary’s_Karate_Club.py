# @author: SFQRM
# first edit time = 2019-10-8
# function:
#    Zachary’s karate club is a commonly used social network where nodes represent members of
# a karate club and the edges their mutual relations. While Zachary was studying the karate club,
# a conflict arose between the administrator and the instructor which resulted in the club splitting in two.
# The code below shows the graph representation of the network and nodes are labeled according to
# which part of the club. The administrator and instructor are marked with ‘A’ and ‘I’, respectively.


import numpy as np
from networkx import karate_club_graph, to_numpy_matrix

zkc = karate_club_graph()                                                       # zkc:数据集
order = sorted(list(zkc.nodes()))                                               # 排序
# print(sorted(list(zkc.nodes())))

"""
    to_numpy_matrix(G, nodelist=None, dtype=None, order=None, multigraph_weight=<built-in function sum>, weight='weight')
    功能：以NumPy矩阵的形式返回图邻接矩阵。
"""
A = to_numpy_matrix(zkc, nodelist=order)                                        #
I = np.eye(zkc.number_of_nodes())
print(A)
print(zkc.number_of_nodes())
print(I)


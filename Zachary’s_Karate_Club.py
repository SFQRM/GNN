# @author: SFQRM
# first edit time = 2019-10-8
# function:
#    Zachary’s karate club is a commonly used social network where nodes represent members of
# a karate club and the edges their mutual relations. While Zachary was studying the karate club,
# a conflict arose between the administrator and the instructor which resulted in the club splitting in two.
# The code below shows the graph representation of the network and nodes are labeled according to
# which part of the club. The administrator and instructor are marked with ‘A’ and ‘I’, respectively.


import numpy as np
# import matplotlib.pyplot as plt
from networkx import karate_club_graph, to_numpy_matrix


zkc = karate_club_graph()                                                       # zkc: 数据集
# print(zkc)
# print(zkc.edges)
# Output:
#     Zachary's Karate Club
#     [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30), (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33), (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32), (31, 33), (32, 33)]

order = sorted(list(zkc.nodes()))                                               # 排序
# print(sorted(list(zkc.nodes())))

"""
    函数原型：to_numpy_matrix(G, nodelist=None, dtype=None, order=None, multigraph_weight=<built-in function sum>, weight='weight')
    功能：以NumPy矩阵的形式返回图邻接矩阵。 
"""
A = to_numpy_matrix(zkc, nodelist=order)                                        # A: 数据集zkc的邻接矩阵
I = np.eye(zkc.number_of_nodes())                                               # I：单位矩阵
# print(A)
# print(zkc.number_of_nodes())
# print(I)

A_hat = A+I                                                                     # A_hat: 加入自环后的邻接矩阵
D_hat = np.array(np.sum(A_hat, axis=0))[0]                                      # D_hat: 度序列
'''np.array()[0]的目的在于结果多一个中括号，去掉中括号'''
D_hat = np.matrix(np.diag(D_hat))                                               # D_hat: 度矩阵
# print(D_hat)

"""
    函数原型：numpy.random.normal(loc=0.0, scale=1.0, size=None)
    参数：
        loc：float
            此概率分布的均值（对应着整个分布的中心centre）
        scale：float
            此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
        size：int or tuple of ints
            输出的shape，默认为None，只输出一个值
"""
W_1 = np.random.normal(loc=0,                                                   # W_1: 随机权重
                       scale=1,
                       size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(loc=0,                                                   # W_1: 随机权重
                       size=(W_1.shape[1], 2))
# print(W_1)
# print(W_2)


def relu(x):
    s = np.where(x < 0, 0, x)
    return s


# """
#     @name: gcn_layer
#     功能：添加隐藏层
#     参数:
#         A_hat: 邻接矩阵
#         D_hat: 度矩阵
#         X: 特征矩阵
#         W: 权重矩阵
# """
def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat**-1 * A_hat * X * W)


H_1 = gcn_layer(A_hat, D_hat, I, W_1)                                           # 第一层掩藏层
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)                                         # 第二层隐藏层

output = H_2

# for node in zkc.nodes:
# print(np.array(output))

feature_representation = {
    node: np.array(output)[node]
    for node in zkc.nodes
}
# print(feature_representation)
# print(feature_representation.get(0))

'''
fig = plt.figure()                                                              # 创建画布
ax1 = fig.add_subplot(111)                                                      # 创建子图
for i in feature_representation.keys():
    # print(i)
    x = feature_representation.get(i)[0]                                        # x: x坐标值
    y = feature_representation.get(i)[1]                                        # y: y坐标值
    # print(x,y)
    ax1.scatter(x, y, c='r', marker='o')

plt.show()                                                                      # 画图
'''

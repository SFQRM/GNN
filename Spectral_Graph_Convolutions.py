# @author: SFQRM
# first edit time: 2019.10.10
# function:
#     EN:
#       https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0

import numpy as np
from numpy import linalg as la


def exponentiation(matrix):
    v, P = la.eig(matrix)                                       # v:特征值，P:特征向量
    # print(v)
    # print(P)
    V = np.diag(v**(1/2))                                       # V:特征值1/2幂运算之后，再矩阵化
    B = P**-1 * V * P                                           # B = A^(0.5)
    # print(B)
    return B**-1                                                # 返回结果:A^(-0.5)


def exponentiation1(matrix):
    v, P = la.eig(matrix)                                       # v:特征值，P:特征向量
    # print(v)
    # print(P)
    V = np.diag(v**(-1/2))                                      # V:特征值1/2幂运算之后，再矩阵化
    B = P * V * P**-1                                           # B = A^(0.5)
    # print(B)
    return B                                                    # 返回结果:A^(-0.5)


A = np.matrix([                                                 # A:邻接矩阵
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float                                                 # 定义类型
)
# print("邻接矩阵A：\n",A)

X = np.matrix([                                                 # X:特征矩阵
    [i, -i]
    for i in range(A.shape[0])
    ],
    dtype=float)                                                # 定义类型
# print("特征矩阵X：\n",X)

# print(A*X)                                                      # 一个最简单的卷积层
                                                                # 每一行是其邻居节点特征的总和

I = np.matrix(np.eye(A.shape[0]))                               # I:单位矩阵
# print("单位矩阵I：\n", I)

A_hat = A+I                                                     # 对邻接矩阵加一个单位矩阵
# print(A_hat)

D = np.array(np.sum(A, axis=0))[0]                              # D:A的度序列
D = np.matrix(np.diag(D))                                       # D:A的（入）度矩阵
# print("A的（入）度矩阵：\n", D)
# print(D**-1)
# print(D**-1 * A * X)

# D_ = np.array(np.sum(A_hat, axis=0))[0]                         # D:A_hat的度序列
# D_ = np.matrix(np.diag(D_))                                     # D:A_hat的（入）度矩阵
# print("A_hat的（入）度矩阵：\n", D)
D_hat = exponentiation(D)                                       # D_hat:D^(-0.5)
# D_hat1 = exponentiation1(D)                                     # D_hat:D^(-0.5)
# print(D_hat)
# print(D_hat1)

hideLayer1 = A*X                                                # hideLayer1: 隐藏层
print("Rule 1隐藏层:\n", hideLayer1)

hideLayer2 = D**-1 * A * X                                      # hideLayer1: 隐藏层
print("Rule 2隐藏层:\n", hideLayer2)

hideLayer3 = D_hat * A * D_hat * X                              # hideLayer3: 隐藏层
print("Rule 3隐藏层:\n", hideLayer3)

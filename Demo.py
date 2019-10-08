import numpy as np
# import tensorflow as tf


def relu(x):
    s = np.where(x < 0, 0, x)
    return s


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
# print("卷积结果：\n", A_hat*X)

'''
    sum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue)
    a是要进行加法运算的向量/数组/矩阵
    axis的值可以为None,也可以为整数和元组
        axis=None，即将数组/矩阵中的元素全部加起来，得到一个和
        axis=0，即压缩列，得到一行
        axis=1，即压缩行，得到一列
'''
D = np.array(np.sum(A, axis=0))[0]                              # D:A的度序列
D = np.matrix(np.diag(D))                                       # D:A的（入）度矩阵
# print("（入）度矩阵：\n", D)

# print(D**-1 * A)
# print(D**-1 * A * X)

D_hat = np.array(np.sum(A_hat, axis=0))[0]                      # D_hat:A_hat的度序列
D_hat = np.matrix(np.diag(D_hat))                               # D_hat:A的（入）度矩阵
# print(D_hat)

W = np.matrix([
    [1,-1],
    [-1,1]
])
# print(W)
# print(D_hat**-1 * A_hat * X * W)
print(relu(D_hat**-1 * A_hat * X * W))

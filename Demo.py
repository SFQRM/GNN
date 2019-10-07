import numpy as np


A = np.matrix([                                                 # A:临界矩阵
    [0, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [1, 0, 1, 0]],
    dtype=float                                                 # 定义类型
)

print(A)

X = np.matrix([                                                 # X:特征矩阵
    [i, -i]
    for i in range(A.shape[0])
    ],
    dtype=float)                                                # 定义类型
print(X)

print(A*X)                                                      # 一个最简单的卷积层
                                                                # 每一行是其邻居节点特征的总和


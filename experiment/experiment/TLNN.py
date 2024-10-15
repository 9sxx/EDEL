import numpy as np
from sklearn.neighbors import KDTree


def TLNN_Search(data,k=10):
    # initialize parameters
    n = data.shape[0]
    r = 1
    tag = True
    TLNN = [[] for _ in range(n)]
    RN = np.zeros(n, dtype=bool)
    KNN = [[] for _ in range(n)]
    TLNN_idx = [[] for _ in range(n)]
    kdtree = KDTree(data)
    # 只查询 k 个最近邻，而不是 n
    dist, index = kdtree.query(data, k=k+1)
    index[:, 0] = -1  # exclude self
    while tag:
        # search KNN and TLNN by round rth
        KNN_idx = index[:, r]
        TLNN_idx = [[] for _ in range(n)]
        for i in range(n):
            if not RN[i]:
                KNN[i].append(KNN_idx[i])
                if r > 1:
                    TLNN[i] = list(set(TLNN[i]).intersection(set(KNN[KNN_idx[i]])))
                TLNN_idx[KNN_idx[i]].append(i)
        # update RN
        RN[[i for i in range(n) if len(TLNN[i]) > 0]] = True
        # stopping condition
        cnt = np.sum(~RN)
        if r > 2 and cnt == prev_cnt:
            tag = False
            r -= 1
        prev_cnt = cnt
        r += 1
        if r == n:
            tag = False
    # convert TLNN_idx to TLNN
    for i in range(n):
        TLNN[i] = TLNN_idx[i]
    return TLNN

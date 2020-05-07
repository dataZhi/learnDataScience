import numpy as np
from preprocessing_ import LabelEncoder_


"""
回归评价指标：
    1. MAE，平均绝对误差
    2. MSE， 均方差
    3. R2，最为常用，R2 = 1 - MSE / VAR
"""


def mean_absolute_error_(y_true, y_pre):
    assert len(y_true) == len(y_pre) and y_true.ndim == 1 and y_pre.ndim == 1, 'y_true or y_pre has wrong dimensions'
    return np.sum(np.abs(y_true - y_pre)) / len(y_pre)


def mean_squared_error_(y_true, y_pre):
    """评价指标：均方根误差，适用于回归"""
    assert len(y_true) == len(y_pre) and y_true.ndim == 1 and y_pre.ndim == 1, 'y_true or y_pre has wrong dimensions'
    return np.sum((y_true - y_pre) ** 2) / len(y_pre)


def r2_score_(y_true, y_pre):
    """评价指标：R2，即1 - 均方根误差/方差，适用于回归"""
    assert len(y_true) == len(y_pre) and y_true.ndim == 1 and y_pre.ndim == 1, 'y_true or y_pre has wrong dimensions'
    u = mean_squared_error_(y_true, y_pre)
    v = np.var(y_true)
    return 1 - u / v


"""
分类评价指标：
    1. 准确率，适用于样本数量无偏情况
    2. 混淆矩阵，二分类情况下较为经典
    3. 精准率，Precision，TP/(TP+FP)，所有分类为1的样本中，真实也为1的比例
    4. 召回率，Recall，TP/(TP+FN)，所有真实为1的样本中，实际分类为1的比例
    5. F1score，倒数的均值，即1/F1 = (1/P + 1/R)/2
"""


def accuracy_score_(y_true, y_pre):
    """评价指标：准确率，适用于分类"""
    assert len(y_true) == len(y_pre) and y_true.ndim == 1 and y_pre.ndim == 1, 'y_true or y_pre has wrong dimensions'
    return np.mean(y_true == y_pre)


def confusion_matrix_(y_true, y_pre):
    labels = np.unique(y_true)
    cm = [[0] * len(labels) for _ in range(len(labels))]
    for i in range(len(labels)):
        for j in range(len(labels)):
            cm[i][j] = np.sum((y_true == labels[i]) & (y_pre == labels[j]))
    return cm


def precision_score_(y_true, y_pre):
    try:
        return np.sum((y_true == 1) & (y_pre == 1)) / np.sum(y_pre == 1)
    except:
        return 0.


def recall_score_(y_true, y_pre):
    try:
        return np.sum((y_true == 1) & (y_pre == 1)) / np.sum(y_true == 1)
    except:
        return 0.


def f1_score_(y_true, y_pre):
    precision = precision_score_(y_true, y_pre)
    recall = recall_score_(y_true, y_pre)
    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.


"""
聚类评价指标：
    1. 轮廓系数：簇内平均距离记为a，距离最近的簇外平均距离记为b，则单样本轮廓系数为(b-a)/max(a,b)，总的轮廓系数为全样本轮廓系数均值
    2. 调整兰德指数：
        a. 兰德指数: RI = （TP+TN）/（TP+FP+FN+TN）
            TP：同类样本被分到同一个簇
            TN：不同类样本被分到不同簇
            FP：不同类样本被分到同一个簇
            FN：同类样本被分到不同簇
        b. 兰德指数的缺陷：分布于0~1之间，对于绝对随机的聚类结果指数评价不为0，无法准确表征聚类效果
        c. 调整兰德指数: ARI = ( RI - E(RI) ) / ( max(RI) - E(RI) )
            RI = ∑i,j(nij,2)
            E(RI)=E(∑i,j(nij,2))=[∑i(ni,2)∑j(nj,2)]/(n,2)
            max(RI)=[∑i(ni,2)+∑j(nj,2)] / 2
        d. 调整兰德指数优点：相当于去均值化的兰德指数，分布于-1~1之间，指数越大越好，随机情况下取0，最优,取1，表示完全一致，最坏取-1，表示完全独立。
"""


def silhouette_score_(X, y):
    silhouettes = np.empty(len(X))
    y_unique = np.unique(y)
    for i, xx in enumerate(X):
        y_self = y[i]
        a = np.sum(np.linalg.norm(xx - X[y == y_self], axis=1)) / (np.sum(y == y_self) - 1)
        b = float('inf')
        for yy in y_unique:
            if yy == y_self:
                continue
            b = min(b, np.mean(np.linalg.norm(xx - X[y == yy], axis=1)))
        silhouettes[i] = (b - a) / max(a, b)
    return np.mean(silhouettes)


def _comb2(n):
    n = int(n)
    return n*(n-1)//2 if n>0 else 0


def _contingency_matrix_(labels_true, labels_pred):
    y_true = LabelEncoder_().fit_transform(labels_true)
    y_pred = LabelEncoder_().fit_transform(labels_pred)
    C = np.unique(y_true)
    K = np.unique(y_pred)
    cm = np.zeros((len(C), len(K)), dtype=int)
    for i, j in zip(y_true, y_pred):
        cm[i, j] += 1
    return cm


def adjusted_rand_score_(labels_true, labels_pred):
    cm = _contingency_matrix_(labels_true, labels_pred)
    sum_comb_c = sum(_comb2(n_c) for n_c in np.sum(cm, axis=1))
    sum_comb_k = sum(_comb2(n_k) for n_k in np.sum(cm, axis=0))
    sum_comb = sum(_comb2(n_ij) for n_ij in cm.flat)
    prod_comb = (sum_comb_c * sum_comb_k) / _comb2(len(labels_true))
    mean_comb = (sum_comb_k + sum_comb_c) / 2.
    return (sum_comb - prod_comb) / (mean_comb - prod_comb)

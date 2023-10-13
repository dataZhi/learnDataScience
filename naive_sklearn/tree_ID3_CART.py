import numpy as np
from collections import Counter
from metrics_ import *
from sklearn.metrics import r2_score


class TreeNodeID3:
    """
    fea_idx:该属性在原始数据集中的索引
    val：如果是叶子节点，其对应的分类结果
    """
    def __init__(self, fea_idx=-1, val=None):
        self.fea_idx = fea_idx
        self.val = val
        self.childs = {}
        self.repr = {'val':val, 'fea_idx':fea_idx, 'childs':self.childs}#用于显示输出

    def __repr__(self):
        return '{}'.format(self.repr)


class DecisionTreeClassifierID3:
    """
    当前仅支持ID3算法实现
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = None
        self.tree_ = None

    def entropy(self, y):
        """根据某一属性计算熵值，当不指定属性时，则计算原始数据集的熵"""
        counter = Counter(y)
        probs = np.array([val / len(y) for val in counter.values()])
        return -np.sum(probs * np.log(probs))

    def best_fea_idx(self, X, y):
        # 对所有特征求解最佳切割特征和最小基尼
        total_entropy = self.entropy(y)
        best_fea_idx, best_entropy = None, float('inf')
        if X.shape[1] < 1:  # 无特征可选
            return None
        if len(np.unique(y)) == 1:  # 样本纯净
            return None
        if self.min_samples_split and len(y) < self.min_samples_split:  ## 小于最小切分样本数
            return None
        for fea_idx in range(X.shape[1]):
            cur_entropy = 0
            for feature in np.unique(X[:, fea_idx]):
                yy = y[X[:, fea_idx] == feature]
                cur_entropy += len(yy) / len(y) * self.entropy(yy)
            if cur_entropy < best_entropy:
                best_fea_idx, best_entropy = fea_idx, cur_entropy
        if self.min_impurity_decrease and (total_entropy - best_entropy) < self.min_impurity_decrease:  # 小于最小切割基尼增益
            return None
        return best_fea_idx

    def build_tree(self, X, y, cur_depth=1):
        best_idx = self.best_fea_idx(X, y)
        if best_idx is None or (self.max_depth and cur_depth > self.max_depth):  # 终止切分条件
            val = Counter(y).most_common(1)[0][0]
            return TreeNodeID3(val=val)
        root = TreeNodeID3(fea_idx=best_idx)
        XX = np.delete(X, best_idx, axis=1)
        for feature in np.unique(X[:, best_idx]):
            idxs = (X[:, best_idx] == feature)
            if self.min_samples_leaf is not None and len(idxs) <= self.min_samples_leaf:
                # 当前特征切分的某个特征值出现最小切片的叶子节点数小于指定数目
                root.val = Counter(y).most_common(1)[0][0]
                root.fea_idx = -1
                root.childs = {}
                return root
            node = self.build_tree(XX[idxs], y[idxs], cur_depth + 1)
            root.childs[feature] = node
        return root

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree_ = self.build_tree(X, y)
        return self

    def predict_one(self, x):
        node = self.tree_
        while node.val is None:
            feature = x[node.fea_idx]
            if feature in node.childs:
                node = node.childs[feature]
            else:
                return -1
        return node.val

    def predict(self, X):
        assert X.ndim == 2 and X.shape[1] == self.n_features, 'X has the wrong dimensions'
        labels = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            labels[i] = self.predict_one(x)
        return labels

    def score(self, X_test, y_test):
        y_pre = self.predict(X_test)
        return np.mean(y_test == y_pre)


class TreeNode:
    """
    单个树节点
    """

    def __init__(self, fea_idx=-1, threshold=None, val=None, left=None, right=None):
        self.fea_idx = fea_idx  # 分割特征索引
        self.threshold = threshold  # 对应特征的取值
        self.val = val  # 如果是叶子节点，则对应标签
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.repr = {'fea_idx': fea_idx, 'feature': threshold, 'val': val, 'left': left, 'right': right}

    def __repr__(self):
        return '{}'.format(self.repr)


class DecisionTreeClassifierD_:
    """
    基于CART的分类树
    """

    def __init__(self, max_depth=None, min_samples_split=None, min_samples_leaf=1, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = None
        self.tree_ = None

    def gini(self, y):
        # 对给定标签列表计算当前基尼指数
        counter = Counter(y)
        return 1 - np.sum([(val / len(y)) ** 2 for val in counter.values()])

    def best_split(self, X, y):
        # 对所有特征求解最佳切割特征和最小基尼
        total_gini = self.gini(y)
        best_fea_idx, best_gini, best_feature = -1, float('inf'), None
        if len(np.unique(y)) == 1:  # 样本纯净
            return best_fea_idx, best_feature
        if self.min_samples_split and len(y) < self.min_samples_split:  ## 小于最小切割样本数
            return best_fea_idx, best_feature
        for fea_idx in range(X.shape[1]):
            for feature in np.unique(X[:, fea_idx]):
                y1 = y[X[:, fea_idx] == feature]
                y0 = y[X[:, fea_idx] != feature]
                if self.min_samples_leaf is not None and (
                        len(y1) < self.min_samples_leaf or len(y0) < self.min_samples_leaf):
                    continue
                cur_gini = len(y1) / len(y) * self.gini(y1) + len(y0) / len(y) * self.gini(y0)
                if cur_gini < best_gini:
                    best_fea_idx, best_gini, best_feature = fea_idx, cur_gini, feature
        if self.min_impurity_decrease and (total_gini - best_gini) < self.min_impurity_decrease:  # 小于最小切割基尼增益
            return -1, None
        return best_fea_idx, best_feature

    def build_tree(self, X, y, cur_depth=1):
        best_fea_idx, best_feature = self.best_split(X, y)
        if best_feature is None or (self.max_depth and cur_depth > self.max_depth):  # 终止切分条件
            val = Counter(y).most_common(1)[0][0]
            return TreeNode(val=val)
        idx_left = np.array(X[:, best_fea_idx] == best_feature)
        idx_right = np.array(X[:, best_fea_idx] != best_feature)
        left = self.build_tree(X[idx_left], y[idx_left], cur_depth + 1)
        right = self.build_tree(X[idx_right], y[idx_right], cur_depth + 1)
        return TreeNode(fea_idx=best_fea_idx, threshold=best_feature, left=left, right=right)

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree_ = self.build_tree(X, y)
        return self

    def predict_one(self, x):
        node = self.tree_
        while node.val is None:
            node = node.left if x[node.fea_idx] == node.threshold else node.right
        return node.val

    def predict(self, X):
        assert X.ndim == 2 and X.shape[1] == self.n_features, 'X has the wrong dimensions'
        labels = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            labels[i] = self.predict_one(x)
        return labels

    def score(self, X_test, y_test):
        y_pre = self.predict(X_test)
        return np.mean(y_test == y_pre)


class DecisionTreeClassifierC_:
    """
    基于CART的分类树
    """

    def __init__(self, max_depth=None, min_samples_split=None, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = None
        self.tree_ = None

    def gini(self, y):
        # 对给定标签列表计算当前基尼指数
        counter = Counter(y)
        return 1 - np.sum([(val / len(y)) ** 2 for val in counter.values()])

    def best_split_con(self, X, y):
        # 对连续取值的特征求解最佳切割阈值和最小基尼
        if len(np.unique(y)) < 2:  # 样本纯净
            return -1, None
        if self.min_samples_split and len(y) < self.min_samples_split:  ## 小于最小切割样本数
            return -1, None
        total_gini = self.gini(y)
        best_fea_idx, best_gini, best_threshold = -1, 1, None
        for fea_idx in range(X.shape[1]):  # 所有特征列索引
            thresholds = (np.sort(X[:, fea_idx])[:-1] + np.sort(X[:, fea_idx])[1:]) / 2
            pre_threshold = None
            for threshold in thresholds:
                if threshold == pre_threshold:
                    continue
                pre_threshold = threshold
                y1 = y[X[:, fea_idx] >= threshold]
                y0 = y[X[:, fea_idx] < threshold]
                cur_gini = len(y1) / len(y) * self.gini(y1) + len(y0) / len(y) * self.gini(y0)
                if cur_gini < best_gini:
                    best_fea_idx, best_gini, best_threshold = fea_idx, cur_gini, threshold
        if self.min_impurity_decrease and (total_gini - best_gini) < self.min_impurity_decrease:  # 小于最小切割基尼增益
            return -1, None
        return best_fea_idx, best_threshold

    def build_tree(self, X, y, cur_depth=1):
        best_fea_idx, best_threshold = self.best_split_con(X, y)
        if best_threshold is None or (self.max_depth and cur_depth > self.max_depth):  # 终止切分条件
            val = Counter(y).most_common(1)[0][0]
            return TreeNode(val=val)
        idx_left = (X[:, best_fea_idx] <= best_threshold)
        idx_right = (X[:, best_fea_idx] > best_threshold)
        left = self.build_tree(X[idx_left], y[idx_left], cur_depth + 1)
        right = self.build_tree(X[idx_right], y[idx_right], cur_depth + 1)
        return TreeNode(fea_idx=best_fea_idx, threshold=best_threshold, left=left, right=right)

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree_ = self.build_tree(X, y)
        return self

    def predict_one(self, x):
        node = self.tree_
        while node.val is None:
            node = node.left if x[node.fea_idx] <= node.threshold else node.right
        return node.val

    def predict(self, X):
        assert X.ndim == 2 and X.shape[1] == self.n_features, 'X has the wrong dimensions'
        labels = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            labels[i] = self.predict_one(x)
        return labels

    def score(self, X_test, y_test):
        y_pre = self.predict(X_test)
        return np.mean(y_test==y_pre)


class DecisionTreeRegressor_:
    """
    基于CART的回归树
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.n_features = None
        self.tree_ = None

    def square_error(self, y):
        # 对给定回归值判断和方差
        return np.sum((y - np.mean(y)) ** 2)

    def best_split_con(self, X, y):
        # 对连续取值的特征求解最佳切割阈值和最小基尼
        if len(np.unique(y)) < 2:  # 样本纯净
            return -1, None
        if self.min_samples_split and len(y) < self.min_samples_split:  ## 小于最小切割样本数
            return -1, None
        total_err = self.square_error(y)
        best_fea_idx, best_err, best_threshold = -1, float('inf'), None
        for fea_idx in range(X.shape[1]):  # 所有特征列索引
            thresholds = (np.sort(X[:, fea_idx])[:-1] + np.sort(X[:, fea_idx])[1:]) / 2
            pre_threshold = None
            for threshold in thresholds:
                if threshold == pre_threshold:
                    continue
                pre_threshold = threshold
                y1 = y[X[:, fea_idx] > threshold]
                y0 = y[X[:, fea_idx] <= threshold]
                if self.min_samples_leaf is not None and (len(y1) < self.min_samples_leaf or len(y0) <= self.min_samples_leaf):
                    continue
                cur_err = len(y1) / len(y) * self.square_error(y1) + len(y0) / len(y) * self.square_error(y0)
                if cur_err < best_err:
                    best_fea_idx, best_err, best_threshold = fea_idx, cur_err, threshold
        if self.min_impurity_decrease and (total_err - best_err) < self.min_impurity_decrease:  # 小于最小切割基尼增益
            return -1, None
        return best_fea_idx, best_threshold

    def build_tree(self, X, y, cur_depth=1):
        best_fea_idx, best_threshold = self.best_split_con(X, y)
        if best_threshold is None or (self.max_depth and cur_depth > self.max_depth):  # 终止切分条件
            val = np.mean(y)
            return TreeNode(val=val)
        idx_left = (X[:, best_fea_idx] <= best_threshold)
        idx_right = (X[:, best_fea_idx] > best_threshold)
        left = self.build_tree(X[idx_left], y[idx_left], cur_depth + 1)
        right = self.build_tree(X[idx_right], y[idx_right], cur_depth + 1)
        return TreeNode(fea_idx=best_fea_idx, threshold=best_threshold, left=left, right=right)

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree_ = self.build_tree(X, y)
        return self

    def predict_one(self, x):
        node = self.tree_
        while node.val is None:
            node = node.left if x[node.fea_idx] <= node.threshold else node.right
        return node.val

    def predict(self, X):
        assert X.ndim == 2 and X.shape[1] == self.n_features, 'X has the wrong dimensions'
        labels = np.empty(len(X), dtype=int)
        for i, x in enumerate(X):
            labels[i] = self.predict_one(x)
        return labels

    def score(self, X_test, y_test):
        y_pre = self.predict(X_test)
        return r2_score(y_test, y_pre)

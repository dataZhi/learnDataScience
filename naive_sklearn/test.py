from preprocessing_ import *
from model_selection_ import *
from linear_model_ import LinearRegression_, Ridge_, Lasso_, LogisticRegression_
from neighbors_ import KNeighborsClassifier_, KNeighborsRegressor_
from tree_ import DecisionTreeClassifier_, DecisionTreeRegressor_
from naive_bayes_ import GaussianNB_, MultinomialNB_
from cluster_ import KMeans_
from ensemble_ import RandomForestClassifier_, RandomForestRegressor_
from sklearn.datasets import load_diabetes, load_digits, make_blobs, load_iris
from metrics_ import *
from sklearn.metrics import *
from sklearn.cluster import KMeans


# 回归任务
Xr, yr = load_diabetes(return_X_y=True)
Xr_train, Xr_test, yr_train, yr_test = train_test_split_(Xr, yr)

myLR_lsq = LinearRegression_()
myLR_lsq.fit(Xr_train, yr_train)
score_lsq = myLR_lsq.score(Xr_test, yr_test)
print('糖尿病预测，线性回归评分: ', score_lsq)

myRidge_sgd = Ridge_(solver='sgd', alpha=0.01)
myRidge_sgd.fit(Xr_train, yr_train)
score_ridge_sgd = myRidge_sgd.score(Xr_test, yr_test)
print('糖尿病预测，随机梯度下降岭回归评分: ', score_ridge_sgd)

myLasso = Lasso_(solver='gd', alpha=0.01)
myLasso.fit(Xr_train, yr_train)
score_lasso = myLasso.score(Xr_test, yr_test)
print('糖尿病预测，LASSO回归评分: ', score_lasso)

knn_reg = KNeighborsRegressor_(weights='distance')
knn_reg.fit(Xr_train, yr_train)
score_knn_reg = knn_reg.score(Xr_test, yr_test)
print('糖尿病预测，KNN回归评分: ', score_knn_reg)

dtr = DecisionTreeRegressor_(max_depth=3)
dtr.fit(Xr_train, yr_train)
print('糖尿病预测，决策树回归评分：', dtr.score(Xr_test, yr_test))

rfr = RandomForestRegressor_(10)
rfr.fit(Xr_train, yr_train)
print('糖尿病预测，RF回归评分：', rfr.score(Xr_test, yr_test))

# 二分类任务
Xc, yc = load_iris(return_X_y=True)
Xc_train, Xc_test, yc_train, yc_test = train_test_split_(Xc, yc)

knn = KNeighborsClassifier_()
knn.fit(Xc_train, yc_train)
print('鸢尾花分类，KNN评分: ', knn.score(Xc_test, yc_test))

gnb = GaussianNB_()
gnb.fit(Xc_train, yc_train)
print('鸢尾花分类，高斯朴素贝叶斯评分: ', gnb.score(Xc_test, yc_test))

rf = RandomForestClassifier_(10)
rf.fit(Xc_train, yc_train)
print("鸢尾花数据集，RF评分：", rf.score(Xc_test, yc_test))

# 多分类任务
Xcm, ycm = load_digits(return_X_y=True)
Xcm_train, Xcm_test, ycm_train, ycm_test = train_test_split_(Xcm, ycm)

dt = DecisionTreeClassifier_(max_depth=10)
dt.fit(Xcm_train, ycm_train)
print('手写数字识别，决策树分类评分：', dt.score(Xcm_test, ycm_test))

mnb = MultinomialNB_()
mnb.fit(Xcm_train, ycm_train)
print('手写数字识别，多项式回归朴素贝叶斯分类评分：', mnb.score(Xcm_test, ycm_test))

# 聚类任务
Xcl, ycl = make_blobs(centers=4, n_samples=500, n_features=2, cluster_std=2)
Xcl_train, Xcl_test, ycl_train, ycl_test = train_test_split_(Xcl, ycl)
km_ = KMeans_(n_clusters=4)
km_.fit(Xcl_train)
print('自定义聚类轮廓系数', km_.score(Xcl_test))
km = KMeans(n_clusters=4)
km.fit(Xcl_train)
y_pre = km.predict(Xcl_test)
print('系统KMeans系统轮廓系数', silhouette_score(Xcl_test, y_pre))
print('系统KMeans自算轮廓系数', silhouette_score_(Xcl_test, y_pre))

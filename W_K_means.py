import numpy as np
import random
import math
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def InitCentroids(X, K):
    n = np.size(X, 0)
    rands_index = np.array(random.sample(range(1, n), K))
    centriod = X[rands_index, :]
    return centriod


def findClosestCentroids(X, w, centroids, num):
    K = np.size(centroids, 0)
    idx = np.zeros((np.size(X, 0)), dtype=int)
    n = X.shape[0]  # n 表示样本个数
    d = X.shape[1]  # d 表示样本的维度
    for i in range(n):
        subs = np.zeros((K, d), dtype=float)
        #对于category，计算汉明距离，对于num，计算欧式距离
        for j in range(d):
            if j in num:
                subs[:, j] = centroids[:, j] - X[i, j]
            else:
                for k in range(K):
                    if X[i, j] == centroids[k, j]:
                        subs[k, j] = 0
                    else:
                        subs[k, j] = 1
        # subs = centroids - X[i, :] 
        dimension2 = np.power(subs, 2)
        w_dimension2 = np.multiply(w, dimension2)
        w_distance2 = np.sum(w_dimension2, axis=1)
        if math.isnan(w_distance2.sum()) or math.isinf(w_distance2.sum()):
            w_distance2 = np.zeros(K)
            # print 'the situation that w_distance2 is nan or inf'
        idx[i] = np.where(w_distance2 == w_distance2.min())[0][0]
    return idx


def computeCentroids(X, idx, K, num):
    n, m = X.shape
    centriod = np.zeros((K, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]  # 一个簇一个簇的分开来计算
        if np.size(index) == 0:
            continue
        temp = X[index, :]  # ? by m # 每次先取出一个簇中的所有样本
        # s = np.sum(temp, axis=0)
        # centriod[k, :] = s / np.size(index)     ###
        for j in range(m):
            if j in num:
                centriod[k, j] = np.sum(temp[:, j]) / np.size(index)
            else:
                #对于category，取众数
                centriod[k, j] = np.argmax(np.bincount(temp[:, j].astype(int)))
    return centriod


def computeWeight(X, centroid, idx, K, belta, num):
    n, m = X.shape
    weight = np.zeros((1, m), dtype=float)
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]  # 取第k个簇的所有样本
        distance2 = np.zeros((temp.shape[0], m), dtype=float)
        for j in range(m):
            if j in num:
                distance2[:, j] = np.power((temp[:, j] - centroid[k, j]), 2)
            else:
                for i in range(temp.shape[0]):
                    if temp[i, j] == centroid[k, j]:
                        distance2[i, j] = 0
                    else:
                        distance2[i, j] = 1
        # distance2 = np.power((temp - centroid[k, :]), 2)  # ? by m
        D = D + np.sum(distance2, axis=0) #按列求和
    e = 1 / float(belta - 1)
    for j in range(m):
        temp = D[0][j] / D[0]
        weight[0][j] = 1 / np.sum((np.power(temp, e)), axis=0)
    return weight


def costFunction(X, K, centroids, idx, w, belta, num):
    n, m = X.shape
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]
        distance2 = np.zeros((temp.shape[0], m), dtype=float)
        for i in range(m):
            if i in num:
                distance2[:, i] = np.power((temp[:, i] - centroids[k, i]), 2)
            else:
                for j in range(temp.shape[0]):
                    if temp[j, i] == centroids[k, i]:
                        distance2[j, i] = 0
                    else:
                        distance2[j, i] = 1
        # distance2 = np.power((temp - centroids[k, :]), 2)  # ? by m
        D = D + np.sum(distance2, axis=0)
    cost = np.sum(w ** belta * D)
    return cost


def isConvergence(costF, max_iter):
    if math.isnan(np.sum(costF)):
        return False
    index = np.size(costF)
    for i in range(index - 1):
        if costF[i] < costF[i + 1]:
            return False
    if index >= max_iter:
        return True
    elif costF[index - 1] == costF[index - 2] == costF[index - 3]:
        return True
    return 'continue'


def wkmeans(X, K, belta, max_iter, num):
    n, m = X.shape
    costF = []
    r = np.random.rand(1, m)
    w = np.divide(r, r.sum())
    idx = np.zeros((np.size(X, 0)), dtype=int)
    c = 0
    centroids = InitCentroids(X, K)
    for i in range(max_iter):
        idx = findClosestCentroids(X, w, centroids, num)
        centroids = computeCentroids(X, idx, K, num)
        w = computeWeight(X, centroids, idx, K, belta, num)
        c = costFunction(X, K, centroids, idx, w, belta, num)
        costF.append(round(c, 4))
        if i < 2:
            continue
        flag = isConvergence(costF, max_iter)
        if flag == 'continue':
            continue
        elif flag:
            best_labels = idx
            best_centers = centroids
            isConverge = True
            return isConverge, best_labels, best_centers, costF
        else:
            isConverge = False
            return isConverge, None, None, costF


class WKMeans:

    def __init__(self, n_clusters=2, max_iter=20, belta=7.0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.belta = belta

    def fit(self, X, num):
        self.isConverge, self.best_labels, self.best_centers, self.cost = wkmeans(
            X=X, K=self.n_clusters, max_iter=self.max_iter, belta=self.belta, num=num
        )
        return self

    def fit_predict(self, X, num, y=None):
        if self.fit(X, num).isConverge:
            return self.best_labels
        else:
            return 'Not convergence with current parameter ' \
                   'or centroids,Please try again'

    def get_params(self):
        return self.isConverge, self.n_clusters, self.belta, 'WKME'

    def get_cost(self):
        return self.cost

#计算Purity
def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def load_data():
    data = load_iris()
    x, y = data.data, data.target
    return x, y


if __name__ == '__main__':
    from sklearn.cluster import KMeans
    #x, y = load_data()
    data_AC = pd.read_csv("datasets/AC/AC.csv")          #获取数据mixed
    Atr_AC = {}
    Atr_AC['num'] = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    Atr_AC['all'] = ["A{}".format(i+1) for i in range(14)]
    Atr_AC['cat'] = list(set(Atr_AC['all']) - set(Atr_AC['num']))
    #num存储的是数值属性的列索引
    num = np.array([data_AC.columns.get_loc(c) for c in data_AC.columns if c in Atr_AC['num']])
    #对num数据进行归一化
    scaler = MinMaxScaler()
    data_AC[Atr_AC['num']] = scaler.fit_transform(data_AC[Atr_AC['num']])
    Atr_AC['label'] = 'class'
    k = 2
    size_num = data_AC.shape[0]
    size_dim = data_AC.shape[1]
    #取LG_data的前size_dim-1列
    x = data_AC[Atr_AC['all']]
    # x = data_LG.iloc[:, 0:size_dim - 1]
    x = x.values
    #将x归一化
    # x = StandardScaler().fit_transform(x)
    #y为数据最后一列
    y = data_AC[Atr_AC['label']]
    y = y.values
    #将y转换为一维数组
    y = y.ravel()
    
    #用kmeans聚类
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x)
    y_pred = kmeans.labels_
    acc_record = []
    ari_record = []
    nmi_record = []
    silhouette_record = []  #轮廓系数SC
    ch_record = []  #Calinski-Harabasz指数CH
    for i in range(20):
        model = WKMeans(n_clusters=k, belta=3)
        while True:
            y_pred = model.fit_predict(x, num)
            if model.isConverge == True:
                y_pred = y_pred + 1 #将y_pred所有元素加一，使得y_pred的元素与y的元素相同
                #计算聚类指标accurrancy
                accu = acc(y, y_pred)
                acc_record.append(accu)
                #计算聚类指标ARI
                ari = metrics.adjusted_rand_score(y, y_pred)
                ari_record.append(ari)
                #计算聚类指标NMI
                nmi = metrics.normalized_mutual_info_score(y, y_pred)
                nmi_record.append(nmi)
                #计算轮廓系数
                silhouette = metrics.silhouette_score(x, y_pred)
                silhouette_record.append(silhouette)
                #计算Calinski-Harabasz指数
                ch = metrics.calinski_harabasz_score(x, y_pred)
                ch_record.append(ch)
                break 
    print("accurrancy's mean by wkmeans: ", np.mean(acc_record))
    print("accurrancy's std by wkmeans: ", np.std(acc_record))
    print("ARI's mean by wkmeans: ", np.mean(ari_record))
    print("ARI's std by wkmeans: ", np.std(ari_record))
    print("NMI's mean by wkmeans: ", np.mean(nmi_record))
    print("NMI's std by wkmeans: ", np.std(nmi_record))
    print("SC's mean by wkmeans: ", np.mean(silhouette_record))
    print("SC's std by wkmeans: ", np.std(silhouette_record))
    print("CH's mean by wkmeans: ", np.mean(ch_record))
    print("CH's std by wkmeans: ", np.std(ch_record))
    # X_tsne = TSNE()
    # X_tsne.fit_transform(x) #进行数据降维 
    # X_tsne = TSNE(n_components=2,learning_rate=200).fit_transform(x)
    # X_tsne = visual(x)
    # 将数据归一化
    # scaler = MinMaxScaler()
    # x_normalized = scaler.fit_transform(x)

    # # 使用TSNE进行降维
    # tsne = TSNE(n_components=2, init='pca', learning_rate=200, random_state=0)
    # X_tsne = tsne.fit_transform(x_normalized)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred)
    plt.savefig("wkmeans.png")
    plt.show() 
    #画class图
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
    plt.savefig("class.png")
    
    plt.show()


# result:
# NMI by sklearn:  0.7581756800057784
# NMI by wkmeans:  0.8130427037493443

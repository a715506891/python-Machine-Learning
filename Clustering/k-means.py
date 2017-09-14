# 层次聚类
#按照数据间的距离分类
from sklearn.metrics import adjusted_rand_score
from sklearn import cluster
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt

# 产生数据的函数


def create_data(centers, num=100, std=0.7):
    # scikit中的make_blobs方法常被用来生成聚类算法的测试数据
    # http://datahref.com/archives/191
    x, labels_true = make_blobs(
        n_samples=num, centers=centers, cluster_std=std)
    return x, labels_true


centers = [[1, 1], [2, 3], [4, 5], [10, 20]]
x, labels_true = create_data(centers, 1000, 0.5)


def test_Kmeans(*data):
    '''
    测试 KMeans 的用法
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    clst = cluster.KMeans()
    clst.fit(X)
    predicted_labels = clst.predict(X)
    print("ARI:%s" % adjusted_rand_score(labels_true, predicted_labels))
    print("Sum center distance %s" % clst.inertia_)


test_Kmeans(x, labels_true)


def test_Kmeans_nclusters(*data):
    '''
    测试 KMeans 的聚类结果随 n_clusters 参数的影响
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    nums = range(1, 50)
    ARIs = []
    Distances = []
    for num in nums:
        clst = cluster.KMeans(n_clusters=num)
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances.append(clst.inertia_)

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, ARIs, marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, Distances, marker='o')
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("inertia_")
    fig.suptitle("KMeans")
    plt.show()


test_Kmeans_nclusters(x, labels_true)  # 4最佳


def test_Kmeans_n_init(*data):
    '''
    测试 KMeans 的聚类结果随 n_init 和 init  参数的影响
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    nums = range(1, 50)
    # 绘图
    fig = plt.figure()

    ARIs_k = []
    Distances_k = []
    ARIs_r = []
    Distances_r = []
    for num in nums:
        clst = cluster.KMeans(n_init=num, init='k-means++')
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ARIs_k.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances_k.append(clst.inertia_)

        clst = cluster.KMeans(n_init=num, init='random')
        clst.fit(X)
        predicted_labels = clst.predict(X)
        ARIs_r.append(adjusted_rand_score(labels_true, predicted_labels))
        Distances_r.append(clst.inertia_)

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(nums, ARIs_k, marker="+", label="k-means++")
    ax.plot(nums, ARIs_r, marker="+", label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("ARI")
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(nums, Distances_k, marker='o', label="k-means++")
    ax.plot(nums, Distances_r, marker='o', label="random")
    ax.set_xlabel("n_init")
    ax.set_ylabel("inertia_")
    ax.legend(loc='best')

    fig.suptitle("KMeans")
    plt.show()


test_Kmeans_n_init(x, labels_true)  # 影响变化不大


# 单独模拟
clst = cluster.KMeans(n_clusters=4, n_init=50, init='k-means++')
clst.fit(x)
predicted_labels = clst.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=predicted_labels)
plt.show()

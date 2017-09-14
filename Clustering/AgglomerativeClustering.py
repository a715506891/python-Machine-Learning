# 层次聚类
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


def test_AgglomerativeClustering(*data):
    '''
    测试 AgglomerativeClustering 的用法
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    clst = cluster.AgglomerativeClustering()
    predicted_labels = clst.fit_predict(X)
    print("ARI:%s" % adjusted_rand_score(labels_true, predicted_labels))


test_AgglomerativeClustering(x, labels_true)


def test_AgglomerativeClustering_nclusters(*data):
    '''
    测试 AgglomerativeClustering 的聚类结果随 n_clusters 参数的影响
    :param data:  可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    nums = range(1, 50)
    ARIs = []
    for num in nums:
        clst = cluster.AgglomerativeClustering(n_clusters=num)
        predicted_labels = clst.fit_predict(X)
        ARIs.append(adjusted_rand_score(labels_true, predicted_labels))

    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(nums, ARIs, marker="+")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    fig.suptitle("AgglomerativeClustering")
    plt.show()


test_AgglomerativeClustering_nclusters(x, labels_true)


def test_AgglomerativeClustering_linkage(*data):
    '''
    测试 AgglomerativeClustering 的聚类结果随链接方式的影响
    :param data:  可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    :return: None
    '''
    X, labels_true = data
    nums = range(1, 50)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    linkages = ['ward', 'complete', 'average']
    markers = "+o*"
    for i, linkage in enumerate(linkages):
        ARIs = []
        for num in nums:
            clst = cluster.AgglomerativeClustering(
                n_clusters=num, linkage=linkage)
            predicted_labels = clst.fit_predict(X)
            ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
        ax.plot(nums, ARIs, marker=markers[i], label="linkage:%s" % linkage)

    ax.set_xlabel("n_clusters")
    ax.set_ylabel("ARI")
    ax.legend(loc="best")
    fig.suptitle("AgglomerativeClustering")
    plt.show()


test_AgglomerativeClustering_linkage(x, labels_true)


# 最佳分类
clst = cluster.AgglomerativeClustering(n_clusters=4, linkage='ward')
predicted_labels = clst.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=predicted_labels)
plt.show()

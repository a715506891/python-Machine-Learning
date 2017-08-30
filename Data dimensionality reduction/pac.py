'''
pca类
n_components:降维后的维数
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold


# 降维并没有一个标准
iris = datasets.load_iris()  # 自带数据集
x = iris.data
y = iris.target


# n_components制定降维后的维数，none，则选择它的值为min(n_samples,n_features)
pca = decomposition.PCA(n_components=None)
pca.fit(x)
print('特征值占比:%s' % str(pca.explained_variance_ratio_))
# 特征值占比:[ 0.92461621  0.05301557  0.01718514  0.00518309]
# 可以由累计特征值看出特征值可以由4维降至2维
pca_do = decomposition.PCA(n_components=2)
pca_do.fit(x)
x_do = pca_do.transform(x)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
colors = ['teal', 'yellowgreen', 'gold']
a = zip(list(np.unique(y)), colors)  # 压缩种类及颜色坐标
for label, color in a:
    position = y == label  # 等于坐标的位置
    ax.scatter(x_do[position, 0], x_do[position, 1],
               label="targer=%d" % label, color=color)


plt.show()

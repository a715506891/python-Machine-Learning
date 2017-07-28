# 最小二乘法
from sklearn import linear_model
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
clf = linear_model.LinearRegression()
clf.fit(X, y)
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
print(clf.predict([3, 3]))  # 求预测值
print(clf.decision_function(X))  # 求预测，等同predict
print(clf.score(X, y))  # R^2
print(clf.get_params())  # 获取参数信息
print(clf.set_params(fit_intercept=False))  # 重新设置参数

# 岭回归（Ridge 回归）
from sklearn import linear_model
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
clf = linear_model.Ridge(alpha=0.1)  # 设置k值
clf.fit(X, y)  # 参数拟合
print(clf.coef_)  # 系数
print(clf.intercept_)  # 常量
print(clf.predict([[3, 3]]))  # 求预测值
print(clf.decision_function(X))  # 求预测，等同predict
print(clf.score(X, y))  # R^2，拟合优度
print(clf.get_params())  # 获取参数信息
print(clf.set_params(fit_intercept=False))  # 重新设置参数


# 岭回归广义交叉验证
from sklearn import linear_model
# import numpy as np
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
# np.array([0.1, 1, 10])
# 教程中的alpha 应改为 alphas ,列表数组均可使用
clf = linear_model.RidgeCV(alphas=[0.1, 1, 10])
clf.fit(X, y)
print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[3, 3]]))
print(clf.decision_function(X))
print(clf.score(X, y))
print(clf.get_params())
print(clf.set_params(fit_intercept=False))

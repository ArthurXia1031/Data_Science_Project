# -*- coding:utf-8 -*-
# @Time       :11/28/22 & 9:53 AM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

RF = RandomForestClassifier(n_estimators=100, n_jobs=4, oob_score=True)
# import the dataset

iris = load_iris()
x = iris.data[:, :2]
# 为了可视化 只取两个特征

# print(iris)
# {'data': array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        [4.6, 3.1, 1.5, 0.2],
#        [5. , 3.6, 1.4, 0.2],
#        [5.4, 3.9, 1.7, 0.4],
#        [4.6, 3.4, 1.4, 0.3],
#        [5. , 3.4, 1.5, 0.2],
#        [4.4, 2.9, 1.4, 0.2],
#        [4.9, 3.1, 1.5, 0.1],
#        [5.4, 3.7, 1.5, 0.2],

y = iris.target

# 训练方法
RF.fit(x, y)

# 设置网格尺寸
h = 0.02

# 创建maps的颜色
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

# print(x)
# print(y)
# visualization
for weights in ["uniform", "distance"]:
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = RF.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20)
    plt.xlim(xx.min(), xx.max())

    plt.title('RandomForestClassifier')

plt.show()
print('RandomForestClassifier', RF.score(x, y))
# RandomForestClassifier 0.9266666666666666

# 使用随机森林对数据进行回归
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)

regr = RandomForestRegressor(max_depth=2, random_state=0,
                             n_estimators=100)

regr.fit(X, y)

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,
                      oob_score=False, random_state=0, verbose=0, warm_start=False)

print(regr.feature_importances_)
print(regr.predict([[0, 0, 0, 0]]))
# [0.18146984 0.81473937 0.00145312 0.00233767]
# [-8.32987858]

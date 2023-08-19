# -*- coding:utf-8 -*-
# @Time       :11/28/22 & 1:50 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
import warnings

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)

warnings.filterwarnings('ignore')

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# load the dataset
# sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
# random uniform set the train and test set
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head())
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  is_train species
# 0                5.1               3.5                1.4               0.2      True  setosa
# 1                4.9               3.0                1.4               0.2      True  setosa
# 2                4.7               3.2                1.3               0.2     False  setosa
# 3                4.6               3.1                1.5               0.2      True  setosa
# 4                5.0               3.6                1.4               0.2      True  setosa


train, test = df[df['is_train'] == True], df[df['is_train'] == False]
# 判断语句

features = df.columns[:4]
# set the features
clf = RandomForestClassifier(n_jobs=2)
# set the n_jobs

y, _ = pd.factorize(train['species'])
# codeing the y according to the feature of speices
# pandas.factorize(values, sort=False, order=None, na_sentinel=-1, size_hint=None)

clf.fit(train[features], y)
# so the result would be the y and the features...

preds = iris.target_names[clf.predict(test[features])]
pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])
# pandas 分组统计 列联表pd.crosstab()
print('----------------------DIVIDED-------------------------')

# print(pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds']))
# preds       setosa  versicolor  virginica
# actual
# setosa          11           0          0
# versicolor       0          14          1
# virginica        0           2          7

h = 0.02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
# each method

classifiers = [
    KNeighborsClassifier(3),
    # knn
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    # decision Tree
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # random forest
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

# methods

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

# X, y = make_classification(n_samples=10000, # 样本个数
# n_features=25, # 特征个数
# n_informative=3, # 有效特征个数
# n_redundant=2, # 冗余特征个数（有效特征的随机组合）
# n_repeated=0, # 重复特征个数（有效特征和冗余特征的随机组合）
# n_classes=3, # 样本类别
# n_clusters_per_class=1, # 簇的个数
# random_state=0)

# 二元分类，创建分类数据make_classification

rng = np.random.RandomState(2)
# seed 设置每次测试 数据相同

X += 2 * rng.uniform(size=X.shape)
# uniform 函数 表示在 一定范围内生成下一个随机数
#  uniform(self, low=0.0, high=1.0, size=None)

# print(X)
# print('----------------------DIVIDED-------------------------')

linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()

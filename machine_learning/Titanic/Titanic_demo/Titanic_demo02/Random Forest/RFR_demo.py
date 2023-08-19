# -*- coding:utf-8 -*-
# @Time       :11/12/22 & 2:53 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split


def load_data_regression():
    diabetes = datasets.load_diabetes()
    return train_test_split(diabetes.train, datasets.test, test_size=0.25, random_state=0)
    # 拆分成训练集和测试集，测试集大小为原始数据集大小的 1/4


def test_RandomForestregressor(*data):
    # 测试 RandomForestRegressor 的用法
    X_train, X_test, y_train, y_test = data
    regr = ensemble.RandomForestRegressor()
    regr.fit(X_train, y_train)
    print('Traing Score: %f' % regr.score(X_train, y_train))
    print("Testing Score: %f" % regr.score(X_test, y_test))


def test_RandomForestRegressor_num(*data):
    # 测试 RandomForestRegressor 的预测性能随  n_estimators 参数的影响
    X_train, X_test, y_train, y_test = data
    nums = np.arange(1, 100, step=2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    testing_scores = []
    training_scores = []
    for num in nums:
        regr = ensemble.RandomForestRegressor(n_estimators=num)
        regr.fit(X_train, y_train)
        training_scores.append(regr.score(X_train, y_train))
        testing_scores.append(regr.score(X_test, y_test))

    ax.plot(nums, training_scores, label='Training Score')
    ax.plot(nums, testing_scores, label='Testing Score')

    ax.set_xlabel('estimator num')
    ax.set_ylabel('score')

    ax.legend(loc='best')

    ax.set_ylim(-1, 1)

    plt.suptitle('RandomForestRegressor')
    plt.show()


def test_RandomForestRegressor_max_depth(*data):
    # 测试 RandomForestRegressor 的预测性能随  max_depth 参数的影响

    X_train, X_test, y_train, y_test = data
    maxdepths = range(1, 20)
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)
    testing_scores = []
    training_scores = []

    for max_depth in maxdepths:
        regr = ensemble.RandomForestRegressor(max_depth=max_depth)
        regr.fit(X_train, y_train)
        training_scores.append(regr.score(X_train, y_train))
        testing_scores.append(regr.score(X_test, y_test))

    ax.plot(maxdepths, training_scores, label='Training Score')
    ax.plot(maxdepths, testing_scores, label='Testing Score')

    ax.set_xlabel('Max_depth')
    ax.set_ylabel('Score')

    ax.legend(loc='best')
    ax.set_ylim(0, 1.05)

    plt.suptitle('RandomForestRegressor')
    plt.grid(axis='x', linestyle='-.')
    plt.show()


def test_RandomForestRegressor_max_features(*data):
    X_train, X_test, y_train, y_test = data
    max_features = np.linspace(0.01, 1, 0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    testing_scores = []
    training_scores = []

    for max_feature in max_features:
        regr = ensemble.RandomForestRegressor(max_features=max_feature)
        regr.fit(X_train, y_train)

        training_scores.append(regr.score(X_train, y_train))
        testing_scores.append(regr.score(X_test, y_test))

    ax.plot(max_features, training_scores, label='Training Score')
    ax.plot(max_features, testing_scores, label='Testing Score')

    ax.set_xlabel('Max_feature')
    ax.set_ylabel('Score')
    ax.legend(loc='best')

    ax.set_ylim(0, 1.05)

    plt.suptitle("RandomForestRegressor")

    plt.grid(axsi='x', linestyle='-.')

    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data_regression()  # get the data
    #    test_RandomForestRegressor(X_train,X_test,y_train,y_test) # 调用 test_RandomForestRegressor
    #    test_RandomForestRegressor_num(X_train,X_test,y_train,y_test) # 调用 test_RandomForestRegressor_num
    #    test_RandomForestRegressor_max_depth(X_train,X_test,y_train,y_test) # 调用 test_RandomForestRegressor_max_depth
    test_RandomForestRegressor_max_features(X_train, X_test, y_train,
                                            y_test)  # 调用 test_RandomForestRegressor_max_features







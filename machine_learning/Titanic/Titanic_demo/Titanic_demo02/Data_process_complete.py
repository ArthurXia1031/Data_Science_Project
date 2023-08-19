# -*- coding:utf-8 -*-
# @Time       :11/12/22 & 2:46 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains


import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

train = pd.read_csv('train.csv')

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 设置1000列的时候才换行
pd.set_option('display.width', 100)


def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进RandomForestRegressor
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0]
    x = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    #     用得到的模型来进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'

    return df


train = set_missing_ages(train)

print(train)

print('----------------------DIVIDED-------------------------')

train_over = set_Cabin_type(train)

print(train_over)

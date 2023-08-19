# -*- coding:utf-8 -*-
# @Time       :11/30/22 & 1:35 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains
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
pd.set_option('display.width', 1000)


def set_missing_ages(df):
    # 把已有的数据型特征取出来丢进RandomForestRegressor
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0]
    x = unknown_age[:, 0]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    # njobs = -1 means all cores
    rfr.fit(x, y)

    #     用得到的模型来进行未知年龄预测

    predictedAges = rfr.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Age'] = 'No'
    return df


train = set_missing_ages(train)
print(train)

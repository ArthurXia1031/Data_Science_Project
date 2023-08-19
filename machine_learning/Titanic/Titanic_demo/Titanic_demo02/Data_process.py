# -*- coding:utf-8 -*-
# @Time       :11/10/22 & 3:20 PM
# @AUTHOR     :Arthur Xia
# @SOFTWARE   :JetBrains

# the most important part
# simply pre-process of the data

# 下一步干啥？咱们该处理处理这些数据，为机器学习建模做点准备了。
# 对了，我这里说的数据预处理，其实就包括了很多Kaggler津津乐道的feature engineering过程，灰常灰常有必要！
# 『特征工程(feature engineering)太重要了！』

# https://blog.csdn.net/jxq0816/article/details/118276316

# Cabin & Age 是数据属性最有问题的，数据丢失最多的。
# 先从Cabin入手，由于cabin数据缺失太多，需要直接把Cabin归结于yes or no

# 对于age：
# 通常遇到缺值的情况，我们会有几种常见的处理方式
# 如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
# 如果缺值的样本适中，而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中
# 如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，然后把它离散化，之后把NaN作为一个type加到属性类目中。
# 有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。
# 本例中，后两种处理方式应该都是可行的，我们先试试拟合补全吧(虽然说没有特别多的背景可供我们拟合，这不一定是一个多么好的选择)

# 我们这里用scikit-learn中的RandomForest来拟合一下缺失的年龄数据
# (注：RandomForest是一个用在原始数据中做不同采样，建立多颗DecisionTree，再进行average等等来降低过拟合现象，提高结果的机器学习算法，我们之后会介绍到

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
    # 把已有的数值型特征都取出来丢进RandomForestRegressor
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknow_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0]
    x = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    # 用得到的模型来进行未知年龄结果预测
    predictedAges = rfr.predict(unknow_age[:, 1::])
    df.loc[(df.Age.isnull(), 'Age')] = predictedAges

    return df


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull(), 'Cabin')] = 'Yes'
    df.loc[(df.Cabin.isnull(), 'Cabin')] = 'No'
    return df


train = set_missing_ages(train)
print(train)

train_over = set_Cabin_type(train)
print(train_over)

print('----------------------DIVIDED-------------------------')

# 因为逻辑回归建模时，需要输入的特征都是数值型特征，我们通常会先对类目型的特征因子化。
# 什么叫做因子化呢？举个例子：
# 以Cabin为例，原本一个属性维度，因为其取值可以是[‘yes’,‘no’]，而将其平展开为’Cabin_yes’,'Cabin_no’两个属性

# 原本Cabin取值为yes的，在此处的"Cabin_yes"下取值为1，在"Cabin_no"下取值为0
# 原本Cabin取值为no的，在此处的"Cabin_yes"下取值为0，在"Cabin_no"下取值为1
# 我们使用pandas的"get_dummies"来完成这个工作，并拼接在原来的"train"之上，如下所示。


dumies_Cabin = pd.get_dummies(train_over['Cabin'], prefix='Cabin')

dumies_Embarked = pd.get_dummies(train_over['Embarked'], prefix='Embarked')

dumies_Sex = pd.get_dummies(train_over['Sex'], prefix='Sex')

dumies_Pclass = pd.get_dummies(train_over['Pclass'], prefix='Pclass')

df = pd.concat([train_over, dumies_Cabin, dumies_Embarked, dumies_Sex, dumies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

print(df)

print('----------------------DIVIDED-------------------------')

# transfer them into 1/0 format

# 仔细看看Age和Fare两个属性，乘客的数值幅度变化，seems like a little large
# 如果大家了解逻辑回归与梯度下降的话，会知道，各属性值之间scale差距太大，将对收敛速度造成几万点伤害值！
# 甚至不收敛！ 所以我们先用scikit-learn里面的preprocessing模块对这俩货做一个scaling
# 所谓scaling，其实就是将一些变化幅度较大的特征化到[-1,1]之内。

from sklearn import preprocessing

scalar = preprocessing.StandardScaler()
df['Age_scaled'] = scalar.fit_transform(df[['Age']])
df['Fare_scaled'] = scalar.fit_transform(df[['Fare']])

print(df)

print('----------------------DIVIDED-------------------------')

# 逻辑回归模型

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

from sklearn import linear_model

# fit到RandomForestRegressor之中

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6, solver='liblinear')

clf.fit(X, y)

print(clf)

print('----------------------DIVIDED-------------------------')

# pre-progress of test.csv

test = pd.read_csv('test.csv')
test.loc[(test.Fare.isnull(), 'Fare')] = 0

test = set_missing_ages(test)

test_over = set_Cabin_type(test)

print(test_over)

dumies_Cabin = pd.get_dummies(test_over['Cabin'], prefix='Cabin')
dumies_Embarked = pd.get_dummies(test_over['Embarked'], prefix='Embarked')
dumies_Sex = pd.get_dummies(test_over['Sex'], prefix='Sex')
dumies_Pclass = pd.get_dummies(test_over['Pclass'], prefix='Pclass')

df = pd.concat([test_over, dumies_Cabin, dumies_Embarked, dumies_Sex, dumies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

scalar = preprocessing.StandardScaler()

df['Age_scaled'] = scalar.fit_transform(df[['Age']])
df['Fare_scaled'] = scalar.fit_transform(df[['Fare']])








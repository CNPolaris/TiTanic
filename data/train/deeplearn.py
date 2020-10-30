# -*- coding: utf-8 -*-
# @Time    : 2020/9/28 22:03
# @FileName: deeplearn.py
# @Author  : CNTian
# @GitHub  ：https://github.com/CNPolaris


import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
# 数据加载
train_data = pd.read_csv('../../dataset/train.csv')
test_data = pd.read_csv('../../dataset/test.csv')
# 数据探索
# print(train_data.info())
# print('-' * 30)
# print(train_data.describe())
# print('-' * 30)
# print(train_data.describe(include=['O']))
# print('-' * 30)
# print(train_data.head())
# print('-' * 30)
# print(train_data.tail())

"""
数据清洗
"""
# 使用平均年龄来填充年龄中的 nan 值
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
# 使用票价的均值填充票价中的 nan 值
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
# print(train_data['Embarked'].value_counts())

# 使用登录最多的港口来填充登录港口的 nan 值
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)
# 将一些字符转换成数字
combine = [train_data, test_data]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)  # 将性别变为数字的格式

"""
构造模型
"""
# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

dvec = DictVectorizer(sparse=False)
train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
# print(dvec.feature_names_)

# 构造 ID3 决策树
clf = DecisionTreeClassifier(criterion='entropy')

"""
模型训练
"""
# 决策树训练
clf.fit(train_features, train_labels)

test_features = dvec.transform(test_features.to_dict(orient='record'))
# 决策树预测
pred_labels = clf.predict(test_features)

# 得到决策树准确率
acc_decision_tree = round(clf.score(train_features, train_labels), 6)
print('score 准确率为 %.4lf' % acc_decision_tree)

"""
使用随机森林进行生死预测
"""
x_train = train_data[features]
y_train = train_data['Survived']
x_test = test_data[features].copy()

# 选择模型
clf = RandomForestClassifier(n_estimators=100)
# 训练
clf.fit(x_train, y_train)
# 预测
predict = clf.predict(x_test)
clf.score(x_train, y_train)
# 准确度
print(round(clf.score(x_train, y_train) * 100, 2))

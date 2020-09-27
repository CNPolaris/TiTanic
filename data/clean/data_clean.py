# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 20:42
# @FileName: data_clean.py
# @Author  : CNTian
# @GitHub  ：https://github.com/CNPolaris


import numpy as np
import pandas as pd

import seaborn as sns
from sklearn.metrics import *

import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from scipy.interpolate import lagrange


class DataDeal():
    def __init__(self):
        self.df_train = pd.read_csv('../../dataset/train.csv')
        self.df_test = pd.read_csv('../../dataset/test.csv')
        self.data_clean = self.df_train.copy(deep=True)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        sns.set(font='SimHei', style='whitegrid', context='talk')
        self.clean_data = pd.read_csv('../../dataset/data_clean.csv')

    def dataPrint(self):
        print(self.df_train.head())
        print("========================")
        print(self.df_train.describe())
        print("========================")
        print(self.df_train.isnull().sum())
        print(self.data_clean)

    def ployinterp_column(self, n, k=5):
        y = self[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
        y = y[y.notnull()]  # 剔除空值
        return lagrange(y.index, list(y))(n)  # 插值并返回插值结果

    def dataExam(self):
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(1, 1, 1)
        data = self.df_train['Survived'].value_counts()
        plt.xlim('yes', 'no')
        data.plot(kind='bar', ax=ax, color=['#FF6600', '#666699'])
        print(data)
        plt.savefig("../../datapicture/surviedabouttotal.png")
        plt.show()

    def viveData(self):
        # print("==========查看形状==========")
        # print(self.df_train.head())

        # print("==========详情打印==========")
        # print(self.df_train.describe())

        # print("==========数据类型==========")
        # print(self.df_train.info())

        # print("==========数据缺失==========")
        # print(self.df_train.isnull())
        # print(self.df_train.isnull().sum())
        # # Age 177 Cabin 687 Embarked 2 #

        # print("==========性别唯一数==========")
        # print(self.df_train['Sex'].value_counts())

        # print("==========性别死亡率==========")
        # print(self.df_train['Survived'].groupby(self.df_train['Sex']).value_counts())
        # print(self.df_train['Survived'].groupby(self.df_train['Sex']).value_counts(normalize = True))
        # sexSurvived = self.df_train['Survived'].groupby(self.df_train['Sex']).value_counts()
        # labels = ['女性存活', '女性死亡', '男性死亡', '男性存活']
        # sexSurvived.plot(kind='pie', labels=labels, subplots=True, figsize=(8, 8), cmap=plt.cm.rainbow, autopct="%3.1f%%",
        #                  fontsize=12)
        # plt.title('男女各自的生存和死亡的比率')
        # plt.legend()
        # plt.savefig('../../datapicture/男女各自的生存和死亡的比率饼图.png')
        # plt.show()

        # print("==========不同舱位死亡生存比例==========")
        # print(self.df_train['Survived'].groupby(self.df_train['Cabin']).value_counts(normalize=True))
        # cabinSurvived = self.df_train['Survived'].groupby(self.df_train['Cabin']).value_counts(normalize=True)
        # print(self.df_train[self.df_train['Survived'] == 1].groupby(self.df_train['Cabin']))

        print("==========不同年龄死亡生存比例==========")
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111)
        survived = self.df_train[self.df_train['Survived'] == 1]
        death = self.df_train[self.df_train['Survived'] == 0]

        sns.distplot(survived['Age'], bins=40, kde=False)
        sns.distplot(death['Age'], bins=40, kde=False, color='red')
        plt.title("不同年龄死亡生存比例")
        plt.savefig("../../datapicture/不同年龄死亡生存比例.png")
        plt.show()

    # 保存数据
    def dataSave(self):
        save = pd.DataFrame(self.clean_data,
                            columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                                     'Parch', 'Ticket', 'Fare', 'Embarked'])
        save.to_csv('../../dataset/data_clean.csv')

    # 数据清洗
    # 删除缺失值比例超过60%的列
    def dataClean1(self):
        print(self.data_clean.isnull().sum())
        for column in self.data_clean.columns:
            miss = self.data_clean[column].isnull().sum(axis=0) / self.data_clean.shape[0]
            if miss > 0.6:
                self.data_clean.drop(column, axis=1, inplace=True)
        # print(self.data_clean.describe())
        print(self.data_clean.isnull().sum())
        print(self.data_clean.columns)
        self.dataSave()

    # 数据填充
    def completion1(self):
        # 填充Embarked
        # print(self.data_clean['Embarked'].isnull().sum())
        # # 考虑到众数可能不止一个， 所以取第一个
        self.clean_data['Embarked'].fillna(self.clean_data['Embarked'].mode()[0], inplace=True)
        print(self.clean_data['Embarked'].isnull().sum())
        print(self.clean_data.isnull().sum())
        self.dataSave()

    def completion2(self):
        # 填充Age
        self.clean_data['Age'].fillna(self.clean_data['Age'].mean(), inplace=True)
        print(self.clean_data.isnull().sum())
        self.dataSave()

        # 拉格朗日填充法

    #  合并Parch和SibSp
    def mergeParchSibSp(self):
        self.clean_data['family'] = self.clean_data['Parch'] + self.clean_data['SibSp']
        save = pd.DataFrame(self.clean_data,
                            columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Ticket', 'Fare',
                                     'Embarked', 'family'])
        save.to_csv('../../dataset/data_clean.csv')
        print(self.clean_data.info())

    # 删除name Id
    def dropNameId(self):
        self.clean_data.drop(['Name', 'PassengerId'], axis=1, inplace=True)
        save = pd.DataFrame(self.clean_data,
                            columns=['Survived', 'Pclass', 'Sex', 'Age', 'Ticket', 'Fare',
                                     'Embarked', 'family'])
        save.to_csv('../../dataset/data_clean.csv')

    # 更换性别表达式
    def changeSex(self):
        self.clean_data['Sex'].replace('male', '1', inplace=True)
        self.clean_data['Sex'].replace('female', '0', inplace=True)
        # print(self.clean_data['Sex'])
        save = pd.DataFrame(self.clean_data,
                            columns=['Survived', 'Pclass', 'Sex', 'Age', 'Ticket', 'Fare',
                                     'Embarked', 'family'])
        save.to_csv('../../dataset/data_clean.csv')


if __name__ == '__main__':
    dataDeal = DataDeal()
    # dataDeal.dataPrint()
    # dataDeal.dataExam()
    # dataDeal.viveData()
    # dataDeal.dataClean1()
    # dataDeal.completion1()
    # dataDeal.completion2()
    # dataDeal.mergeParchSibSp()
    # dataDeal.dropNameId()
    dataDeal.changeSex()

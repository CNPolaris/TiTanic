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

    # 数据填充
    def completion(self):
        for column in self.data_clean.columns:
            for i in range(len(self.data_clean)):
                if (self.data_clean[column].isnull())[i]:
                    self.data_clean[column][i] = self.ployinterp_column(self.data_clean[column], i)
        print(self.data_clean.isnull().sum())

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
        sns.distplot(death['Age'], bins=40, kde=False,color='red')
        plt.title("不同年龄死亡生存比例")
        plt.savefig("../../datapicture/不同年龄死亡生存比例.png")
        plt.show()


if __name__ == '__main__':
    dataDeal = DataDeal()
    # dataDeal.dataPrint()
    # dataDeal.dataExam()
    # dataDeal.completion()
    dataDeal.viveData()

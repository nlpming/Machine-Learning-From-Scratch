#-*- coding:utf-8 -*-
from __future__ import print_function
import sys
import os
import numpy as np
import pandas as pd
import pydotplus

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# 西瓜书表4.1数据集
def create_watermelon_data():
    datasets = [
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],

        # ----------------------------------------------------
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]

    labels = [u'色泽', u'根蒂', u'敲击', u'纹理', u'脐部', u'触感', u'好瓜/坏瓜']
    feature_dict = {
        u"色泽": [u"青绿", u"乌黑", u"浅白"],
        u"根蒂": [u"蜷缩", u"稍蜷", u"硬挺"],
        u"敲击": [u"浊响", u"沉闷", u"清脆"],
        u"纹理": [u"清晰", u"稍糊", u"模糊"],
        u"脐部": [u"平坦", u"稍凹", u"凹陷"],
        u"触感": [u"硬滑", u"软粘"],
    }

    # 返回数据集和每个维度的名称
    return datasets, labels, feature_dict

def process_datasets(datasets, labels, feature_dict):

    X_train = []
    y_train = []
    for data in datasets:
        res = []
        for i,item in enumerate(data[:-1]):
            feature_values = feature_dict[labels[i]]
            res.append(feature_values.index(item.decode('utf-8')))
        X_train.append(res)

        label = 1 if data[-1] == '好瓜' else 0
        y_train.append(label)

    return np.array(X_train), np.array(y_train)

if __name__ == "__main__":

    # 创建数据集
    datasets, labels, feature_dict = create_watermelon_data()
    X_train, y_train = process_datasets(datasets, labels, feature_dict)
    print(X_train)
    print(y_train)

    # 训练过程
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    print(clf)


#-*- coding:utf-8 -*-
"""
决策树算法，ID3分类决策树
环境：python3
"""
from __future__ import print_function
from __future__ import division
import sys
import os
import numpy as np
import pandas as pd
from math import log
import pprint
import pdb
import json

# 书上题目5.1
def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    feature_dict = {
        u"年龄":         [u"青年", u"中年", u"老年"],
        u"有工作":       [u"是", u"否"],
        u"有自己的房子":  [u"是", u"否"],
        u"信贷情况":     [u"非常好", u"好", u"一般"]
    }
    # 返回数据集和每个维度的名称
    return datasets, labels, feature_dict

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


def calc_ent(datasets):
    """计算信息熵 H(D)"""
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p / data_length) * log(p / data_length, 2)
                for p in label_count.values()])
    return ent

def cond_ent(datasets, axis=0):
    """计算条件熵 H(D|A)"""
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    cond_ent = sum(
        [(len(p) / data_length) * calc_ent(p) for p in feature_sets.values()])

    return cond_ent

def info_gain(ent, cond_ent):
    """计算信息增益 g(D,A)=H(D)-H(D|A)"""
    return ent - cond_ent

def info_gain_train(datasets):
    # 特征数量
    count = len(datasets[0]) - 1
    # 计算信息熵
    ent = calc_ent(datasets)
    # ent = entropy(datasets)

    # 选择信息增益最大的特征
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        best_feature.append((c, c_info_gain))
        print('特征({}) - info_gain - {:.3f}'.format(labels[c], c_info_gain))

    # 比较大小
    best_ = max(best_feature, key=lambda x: x[-1])
    print('特征({})的信息增益最大，选择为根节点特征'.format(labels[best_[0]]))

class Node:
    """定义树结点"""
    def __init__(self, root=True, label=None, feature_name=None, feature=None, samples=None, flag=None):
        self.root = root # 是否为根结点
        self.label = label # 叶结点类别标记
        self.feature_name = feature_name # 特征名称
        self.feature = feature # 特征所对应的id
        self.samples = samples # 结点所包含的样本
        self.flag = flag # 结束划分的标记
        self.tree = {}
        self.result = {
            'label:': self.label,
            'feature_name': self.feature_name,
            'samples': self.samples,
            'flag': self.flag,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        """根据最优特征，对决策树进行分裂
        Args:
            val: 特征
            node: 新的子结点；
        """
        self.tree[val] = node

    def predict(self, features):
        """递归遍历，直到叶子结点"""
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)

class DTree:
    """利用ID3算法生成决策树"""

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # 熵
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2)
                    for p in label_count.values()])
        return ent

    # 经验条件熵
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p)
                        for p in feature_sets.values()])
        return cond_ent

    # 信息增益
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data, feature_dict):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        # y_train: 存储类别；features: 存储特征；
        y_train, features = train_data.iloc[:, -1], train_data.columns[:-1]

        # 1. 若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0], samples=train_data.index.tolist(),
                        flag="All the sample's label is same!")

        # 2. 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(
                root=True,
                label=y_train.value_counts().sort_values(ascending=False).index[0],
                samples=train_data.index.tolist(),
                flag="Feature is null!"
            )

        # 3. 计算最大信息增益 同5.1, Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4. Ag的信息增益小于阈值eta, 则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(
                root=True,
                label=y_train.value_counts().sort_values(ascending=False).index[0],
                samples=train_data.index.tolist(),
                flag="Information gain less eta!"
            )

        # 5. 构建Ag子集
        node_tree = Node(
            root=False, feature_name=max_feature_name, feature=max_feature, samples=train_data.index.tolist())

        # 获取特征的不同取值；下面的方式，可能忽略未出现的属性取值
        #  feature_list = train_data[max_feature_name].value_counts().index
        feature_list = feature_dict[max_feature_name]

        # 根据特征的不同取值，将数据划分成多份；
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)

            # 获取默认label
            label = train_data.iloc[:, -1].value_counts()
            default_label = label.index[0]

            # 6. 特征某个取值样本为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
            if len(sub_train_df) == 0:
                sub_tree = Node(root=True, label=default_label, flag="This node sample is null!")
                node_tree.add_node(f, sub_tree)
            else:
                # 7. 递归生成树
                sub_tree = self.train(sub_train_df, feature_dict)
                node_tree.add_node(f, sub_tree)

        return node_tree

    def fit(self, train_data, feature_dict):
        """开始训练"""

        self._tree = self.train(train_data, feature_dict)

        tree_str = str(self._tree)
        tree_json = json.dumps(eval(tree_str), ensure_ascii=False)
        print(tree_json)

        return self._tree

    def predict(self, X_test):
        """开始预测"""
        return self._tree.predict(X_test)


if __name__ == "__main__":
    # pdb.set_trace()

    # >>>>>>> 统计学习方法 >>>>>>>>
    # 创建数据集
    datasets, labels, feature_dict = create_data()
    train_data = pd.DataFrame(datasets, columns=labels)
    print(train_data)

    # 信息增益
    info_gain_train(np.array(datasets))

    # 创建ID3决策树
    dt = DTree()
    tree = dt.fit(train_data, feature_dict)

    # 执行预测
    print("预测结果：", dt.predict(['老年', '否', '否', '一般']))

    # >>>>>>> 西瓜书数据集 >>>>>>>>
    datasets, labels, feature_dict = create_watermelon_data()
    train_data = pd.DataFrame(datasets, columns=labels)
    print(train_data)

    # 信息增益
    info_gain_train(np.array(datasets))

    # 创建ID3决策树
    dt = DTree()
    tree = dt.fit(train_data, feature_dict)



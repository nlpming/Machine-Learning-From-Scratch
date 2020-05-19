#-*- coding:utf-8 -*-
from __future__ import print_function
import sys
import os
import pdb
import json

import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error

# 定义一个简单的树结构
class RTree:
    def __init__(self, data, z, slicedIdx):
        self.data = data
        self.z = z
        self.isLeaf = True # 是否为叶结点，默认都为叶结点，如果生成子树则设为False
        self.slicedIdx = slicedIdx # 节点上只保存数据的序号,不保存数据子集,节约内存
        self.left = None # 左子树
        self.right = None # 右子树
        self.output = np.mean(z[slicedIdx]) # 子树上所有结点值的平均
        self.j = None # 最优切分属性
        self.s = None # 最优切分点

    # 本节点所带的子数据如果大于1个,则生成两个叶子节点,本节点不再是叶子节点
    def grow(self):
        if len(self.slicedIdx)>1:
            # 用于选择最优切分属性j和切分点s
            j, s, err = bestDivi(self.data, self.z, self.slicedIdx)
            leftIdx, rightIdx = [], []

            for i in self.slicedIdx:
                if self.data[i,j] < s:
                    leftIdx.append(i)
                else:
                    rightIdx.append(i)

            self.isLeaf = False
            self.left = RTree(self.data, self.z, leftIdx)
            self.right = RTree(self.data, self.z, rightIdx)
            self.j = j
            self.s = s

    def err(self):
        return np.mean((self.z[self.slicedIdx]-self.output)**2)

def squaErr(data, output, slicedIdx, j, s):
    """计算平方差
    Args:
        data: 整个训练数据；
        output: 当前结点输出值；
        slicedIdx: 当前结点数据索引；
        j: 第j维属性；
        s: 第j维属性，最优切分点；
    """

    # 挑选数据子集
    region1 = []
    region2 = []
    for i in slicedIdx:
        if data[i, j] < s:
            region1.append(i)
        else:
            region2.append(i)

    # 计算子集上的平均输出
    c1 = np.mean(output[region1])
    err1 = np.sum((output[region1]-c1)**2)

    c2 = np.mean(output[region2])
    err2 = np.sum((output[region2]-c2)**2)

    # 返回平方差
    return err1+err2


def bestDivi(data, z, slicedIdx):
    """用于选择最优切分属性j和切分点s"""

    min_j = 0 # 第1维特征
    sortedValue = np.sort(data[slicedIdx][:, min_j])
    min_s = (sortedValue[0]+sortedValue[1])/2 # 第1个切分点
    err = squaErr(data, z, slicedIdx, min_j, min_s) # 计算平方差

    # 遍历属性
    for j in range(data.shape[1]):
        # 产生某个属性值的分割点集合，均值作为切分点
        sortedValue = np.sort(data[slicedIdx][:,j])
        sliceValue = (sortedValue[1:]+sortedValue[:-1])/2
        for s in sliceValue:
            errNew = squaErr(data, z, slicedIdx, j, s)
            if errNew < err:
                err = errNew
                min_j = j
                min_s = s

    return min_j, min_s, err

# 更新树
def updateTree(tree):
    if tree.isLeaf:
        tree.grow()
    else:
        updateTree(tree.left)
        updateTree(tree.right)

# 预测一个数据点的输出
def predict(single_data, init_tree):
    tree = init_tree
    while True:
        if tree.isLeaf:
            return tree.output
        else:
            if single_data[tree.j] < tree.s:
                tree = tree.left
            else:
                tree = tree.right

# pdb.set_trace()

# 利用z=x+y+noise 人为生成一个数据集, 具有3个特征
n_samples = 10
points = np.random.rand(n_samples, 3)
z = points[:,0]+points[:,1]+points[:,2] + 0.2*(np.random.rand(n_samples)-0.5)

# 生成根节点
root = RTree(points, z, range(n_samples))

# 进行五次生长, 观测每次生长过后的拟合效果
for ii in range(10):
    updateTree(root)

    z_predicted = np.array([predict(p, root) for p in points])
    print("第{}次生长，训练集均方误差：{}".format(ii+1, mean_squared_error(z, z_predicted)))


if __name__ == "__main__":
    pass




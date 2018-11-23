# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 22:21:13 2018
@author: wmy
"""

import math

def CalcShannonEnt(dataset,feature):
    #数据的长度
    num=len(dataset)
    #记录类和次数的字典
    labelcounts={}
    for element in dataset:
        #选择数据的特征类别
        currentlabel=element[feature]
        if currentlabel not in labelcounts.keys():
            #创建新的类别
            labelcounts[currentlabel]=0
        #属于的类别的值加一
        labelcounts[currentlabel]+=1
    #shannon熵
    shannonent=0.0
    for key in labelcounts:
        #计算属于每一个类的概率
        prob=float(labelcounts[key])/num
        #H=H-P*log2(p)
        shannonent-=prob*math.log(prob,2)
    return shannonent

dataset=[['green'],['yellow'],['blue'],['blue'],['blue']]

print(CalcShannonEnt(dataset,0))

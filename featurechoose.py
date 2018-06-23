# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 18:05:22 2018

@author: wmy
"""

import math

def CalcShannonEnt(dataset):
    #数据的长度
    num=len(dataset)
    #记录类和次数的字典
    labelcounts={}
    for element in dataset:
        #选择数据的特征类别
        currentlabel=element[-1]
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

def SplitDataSet(dataset,feature,value):
    retdataset=[]
    for element in dataset:
        if element[feature]==value:
            reducedelement=element[:feature]
            reducedelement.extend(element[feature+1:])
            retdataset.append(reducedelement)
    return retdataset

def ChooseBestFeature(dataset):
    #数据集特征的个数，列表最后一个为类别
    numfeatures=len(dataset[0])-1
    #计算初始的shannon熵
    baseentropy=CalcShannonEnt(dataset)
    #信息增益
    bestinfogain=0.0
    #最恰当的特征值初始化为初始值-1(类别)
    bestfeature=-1
    #遍历数据集的特征值
    for i in range(numfeatures):
        #第i个特征的特征值列表
        featurevaluelist=[example[i] for example in dataset]
        #转换为集合数据类型，用于得到唯一元素值
        uniquevals=set(featurevaluelist)
        #新的shannon熵
        newentropy=0.0
        #遍历这个特征的所有元素的值，开始计算其shannon熵
        for value in uniquevals:
            #划分数据集
            subdataset=SplitDataSet(dataset,i,value)
            #概率=划分到特征为i值为value的元素的个数/数据集总个数
            prob=len(subdataset)/float(len(dataset))
            #计算shannon熵
            newentropy+=prob*CalcShannonEnt(subdataset)
        #信息增益=熵的减小量 
        #print(newentropy)
        currentinfogain=baseentropy-newentropy
        #比较信息增益的大小
        if currentinfogain>bestinfogain:
            #当前更大的信息增益赋给最好的信息增益
            bestinfogain=currentinfogain
            #得出最恰当的特征值
            bestfeature=i
    #返回最恰当的特征值
    return bestfeature
        
dataset=[['h','u','p'],['s','u','p'],['m','e','a'],
         ['h','e','a'],['s','e','a']]            
    
print(ChooseBestFeature(dataset)) 
  

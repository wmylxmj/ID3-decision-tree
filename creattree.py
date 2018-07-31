# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:34:31 2018

@author: wmy
"""

from math import log
import operator

def CalcShannonEntropy(dataset):
    #the number of samples
    NumberEntries = len(dataset)
    #creat a dictionary to record the labels
    LabelCounts = {}
    for Sample in dataset:
        CurrentLabel = Sample[-1]
        if CurrentLabel not in LabelCounts.keys():
            #creat a new label
            LabelCounts[CurrentLabel] = 0
        LabelCounts[CurrentLabel] += 1
    #ShannonEntropy is only decided by labels
    ShannonEntropy = 0.0
    for key in LabelCounts:
        Prob = float(LabelCounts[key])/NumberEntries
        ShannonEntropy -= Prob * log(Prob, 2)
    return ShannonEntropy

def SplitDataSet(dataset, feature, value):
    ReturnDataSet = []
    for Sample in dataset:
        if Sample[feature] == value:
            #remove the feature which is selected 
            ReducedSample = Sample[:feature]
            ReducedSample.extend(Sample[feature+1:])
            ReturnDataSet.append(ReducedSample)
    return ReturnDataSet

def BestFeatureSelect(dataset):
    #the number of features
    NumberFeatures = len(dataset[0]) - 1
    #init the shannon entropy
    BaseEntropy = CalcShannonEntropy(dataset)
    BestInfoGain = 0.0
    BestFeature = -1
    for i in range(NumberFeatures):
        #A feature extraction value in all samples
        FeatureList = [Sample[i] for Sample in dataset]
        #remove duplicate elements
        UniqueValues = set(FeatureList)
        NewEntropy = 0.0
        for value in UniqueValues:
            #split the dataset
            SubDataSet = SplitDataSet(dataset, i, value)
            Prob = len(SubDataSet)/float(len(dataset))
            NewEntropy += Prob * CalcShannonEntropy(SubDataSet)
        InfoGain =  BaseEntropy - NewEntropy
        if InfoGain > BestInfoGain:
            BestInfoGain = InfoGain
            BestFeature = i
    return BestFeature
        
def MajorityCnt(classlist):
    ClassCount = {}
    for element in classlist:
        if element not in ClassCount.keys():
            ClassCount[element] = 0
        ClassCount[element] += 1
    SortedClassCount = sorted(ClassCount.items(),\
                              key=operator.itemgetter(1), reverse=True)
    return SortedClassCount[0][0]

def CreatTree(dataset, labels):
    #move all classes to a list
    ClassList = [Sample[-1] for Sample in dataset]
    #stop spliting if the class is the same
    if ClassList.count(ClassList[0]) == len(ClassList):
        return ClassList[0]
    if len(dataset[0]) == 1:
        return MajorityCnt(ClassList)
    #find the best feature to split the dataset
    BestFeature = BestFeatureSelect(dataset)
    BestFeatureLabel = labels[BestFeature]
    #creat the decision tree
    MyTree = {BestFeatureLabel:{}}
    del(labels[BestFeature])
    FeatureValues = [Sample[BestFeature] for Sample in dataset]
    #remove duplicate elements
    UniqueValues = set(FeatureValues)
    for value in UniqueValues:
        SubLabels = labels[:]
        #iterate
        MyTree[BestFeatureLabel][value] = CreatTree(SplitDataSet\
              (dataset, BestFeature, value), SubLabels)
    return MyTree
    
dataset = [[1,1,'yes'],
           [1,1,'yes'],
           [1,0,'no'],
           [0,1,'no'],
           [0,1,'no']]
labels = ['no surfacing','flippers']

print(CreatTree(dataset, labels))

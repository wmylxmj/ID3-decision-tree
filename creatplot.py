# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 17:34:31 2018

@author: wmy
"""

from math import log
import operator
import matplotlib.pyplot as plt

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
    SortedClassCount = sorted(ClassCount.items(), \
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
        MyTree[BestFeatureLabel][value] = CreatTree(SplitDataSet \
              (dataset, BestFeature, value), SubLabels)
    return MyTree
    
dataset = [[1,1,'yes'],
           [1,1,'yes'],
           [1,0,'no'],
           [0,1,'no'],
           [0,1,'no']]
labels = ['no surfacing','flippers']
mytree = CreatTree(dataset, labels)

print(mytree)

def GetLeafsNumber(tree):
    NumberLeafs = 0 
    #find the first key string
    FirstString = list(tree.keys())[0]
    SecondDict = tree[FirstString]
    for key in SecondDict.keys():
        if type(SecondDict[key]).__name__=='dict':
            NumberLeafs += GetLeafsNumber(SecondDict[key])
        else:
            NumberLeafs += 1
    return NumberLeafs
           
def GetTreeDepth(tree):
    MaxDepth = 0
    #find the first key string
    FirstString = list(tree.keys())[0]
    SecondDict = tree[FirstString]
    for key in SecondDict.keys():
        if type(SecondDict[key]).__name__=='dict':
            ThisDepth = 1 + GetTreeDepth(SecondDict[key])
        else:
            ThisDepth = 1
        if ThisDepth > MaxDepth:
            MaxDepth = ThisDepth
    return MaxDepth
        
print(GetLeafsNumber(mytree))
print(GetTreeDepth(mytree))

DecisionNode = dict(boxstyle="sawtooth", fc="0.8")
LeafNode = dict(boxstyle="round4", fc="0.8")
ArrowArgs = dict(arrowstyle="<-")
    
def PlotNode(nodetxt, centerpt, parentpt, nodetype):
    CreatPlot.ax1.annotate(nodetxt, xy=parentpt, \
                            xycoords='axes fraction',
                            xytext=centerpt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodetype, arrowprops=ArrowArgs)

def CreatPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    CreatPlot.ax1 = plt.subplot(111, frameon=False)
    PlotNode('decision node', (0.5, 0.1), (0.1, 0.5), DecisionNode)
    PlotNode('leaf node', (0.8, 0.1), (0.3, 0.8), LeafNode)
    plt.show()
    
CreatPlot()
        

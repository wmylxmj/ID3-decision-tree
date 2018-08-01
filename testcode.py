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
        #compare the infogain
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
    #copy the labels
    Labels = labels[:]
    #move all classes to a list
    ClassList = [Sample[-1] for Sample in dataset]
    #stop spliting if the class is the same
    if ClassList.count(ClassList[0]) == len(ClassList):
        return ClassList[0]
    if len(dataset[0]) == 1:
        return MajorityCnt(ClassList)
    #find the best feature to split the dataset
    BestFeature = BestFeatureSelect(dataset)
    BestFeatureLabel = Labels[BestFeature]
    #creat the decision tree
    MyTree = {BestFeatureLabel:{}}
    del(Labels[BestFeature])
    FeatureValues = [Sample[BestFeature] for Sample in dataset]
    #remove duplicate elements
    UniqueValues = set(FeatureValues)
    for value in UniqueValues:
        SubLabels = Labels[:]
        #iterate
        MyTree[BestFeatureLabel][value] = CreatTree(SplitDataSet \
              (dataset, BestFeature, value), SubLabels)
    return MyTree

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

DecisionNode = dict(boxstyle="sawtooth", fc="0.8")
LeafNode = dict(boxstyle="round4", fc="0.8")
ArrowArgs = dict(arrowstyle="<-")
    
def PlotNode(nodetxt, centerpt, parentpt, nodetype):
    CreatPlot.ax1.annotate(nodetxt, xy=parentpt, \
                            xycoords='axes fraction',
                            xytext=centerpt, textcoords='axes fraction', \
                            va="center", ha="center", bbox=nodetype, arrowprops=ArrowArgs)

def CreatPlot(intree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    Axprops = dict(xticks=[], yticks=[])
    CreatPlot.ax1 = plt.subplot(111, frameon=False, **Axprops)
    PlotTree.TotalW = float(GetLeafsNumber(intree))
    PlotTree.TotalD = float(GetTreeDepth(intree))
    PlotTree.xoff = -0.5/PlotTree.TotalW
    PlotTree.yoff = 1.0
    PlotTree(intree, (0.5, 1.0), '')
    plt.show()
        
def PlotMidText(cntrpt, parentpt, txtstring):
    xMid = (parentpt[0]-cntrpt[0])/2.0 + cntrpt[0]
    yMid = (parentpt[1]-cntrpt[1])/2.0 + cntrpt[1]
    CreatPlot.ax1.text(xMid, yMid, txtstring)
    
def PlotTree(tree, parentpt, nodetxt):
    NumberLeafs = GetLeafsNumber(tree)
    '''TreeDepth = GetTreeDepth(tree)'''
    FirstString = list(tree.keys())[0]
    CntrPlot = (PlotTree.xoff + (1.0 + float(NumberLeafs))/2.0/PlotTree.TotalW, \
                PlotTree.yoff)
    PlotMidText(CntrPlot, parentpt, nodetxt)
    PlotNode(FirstString, CntrPlot, parentpt, DecisionNode)
    SecondDict = tree[FirstString]
    PlotTree.yoff = PlotTree.yoff - 1.0/PlotTree.TotalD
    for key in SecondDict.keys():
        if type(SecondDict[key]).__name__=='dict':
            PlotTree(SecondDict[key], CntrPlot, str(key))
        else:
            PlotTree.xoff = PlotTree.xoff + 1.0/PlotTree.TotalW
            PlotNode(SecondDict[key], (PlotTree.xoff, PlotTree.yoff), \
                     CntrPlot, LeafNode)
            PlotMidText((PlotTree.xoff, PlotTree.yoff), CntrPlot, str(key))
    PlotTree.yoff = PlotTree.yoff + 1.0/PlotTree.TotalD
  
def ID3DecisionTreeClassify(tree, featlabels, testvector):
    FirstString = list(tree.keys())[0]
    SecondDict = tree[FirstString]
    FeatureIndex = featlabels.index(FirstString)
    for key in SecondDict.keys():
        if testvector[FeatureIndex] == key:
            if type(SecondDict[key]).__name__=='dict':
                ClassLabel = ID3DecisionTreeClassify(SecondDict[key], \
                                                     featlabels, testvector)
            else:
                ClassLabel = SecondDict[key]
    return ClassLabel

dataset = [['yes','yes','no','no','yes','no','mammalia'],#人类
           ['no','no','half','no','yes','no','reptilia'],#海龟
           ['yes','no','no','yes','yes','no','birds'],#鸽子
           ['yes','yes','yes','no','no','no','mammalia'],#鲸
           ['no','no','no','no','yes','no','reptilia'],#蜥蜴
           ['no','no','yes','no','no','no','fish']]#海马

labels = ['constant temperature','viviparity','aquatic','fly','have legs','winter sleep']

mytree = CreatTree(dataset, labels)

CreatPlot(mytree)

print(dataset)
print(labels)
print(mytree)

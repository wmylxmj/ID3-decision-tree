# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 16:47:50 2018

@author: wmy
"""

def SplitDataSet(dataset,feature,value):
    retdataset=[]
    for element in dataset:
        if element[feature]==value:
            reducedelement=element[:feature]
            reducedelement.extend(element[feature+1:])
            retdataset.append(reducedelement)
    return retdataset

dataset=[[1,0,'green'],[1,1,'blue'],[0,1,'yellow'],[0,0,'red'],
         [0,0,'green'],[0,1,'blue'],[1,1,'yellow'],[0,1,'red']]

print(SplitDataSet(dataset,0,0))

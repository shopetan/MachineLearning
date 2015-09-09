#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re,numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score


# GlobalVariable
trainingData = []
trainingLabel = []
testData = []
testLabel = []

def importCSV(csvSrc):
    dataFrame = pd.read_csv(csvSrc)
    return dataFrame

def dataFrameToMatrix(dataFrame):
    dataFrame = dataFrame.astype('float64')
    dataFrame = dataFrame.sort(columns='annu')
    numpyMatrix = dataFrame.as_matrix()
    return numpyMatrix
    
#サンプルデータを一括してトレーニングデータとテストデータに分けてセットしたい場合に使う
#これは単純に学習させて精度を見たい時に用いるので、実際に予測をしたい場合は
# setTrainingData()とsetPredictDataを個別にセットする必要がある。
def setSampleData():
    global trainingData
    global trainingLabel
    global testData
    global testLabel
    num = 0
    
    for row in numpyMatrix:
        node = []
        label = row[2]
        node.append(row[0])
        node.append(row[1])
        if num % 2 == 0:
            trainingData.append(node)
            trainingLabel.append(label)
        else:
            testData.append(node)
            testLabel.append(label)
        num += 1

#トレーニングデータをセット
# TODO:教師データのチューニング方法
def setTrainingData():
    global trainingData
    global trainingLabel
    num = 0
    
    for row in numpyMatrix:
        node = []
        label = row[2]
        node.append(row[0])
        node.append(row[1])
        if num % 2 == 0:
            trainingData.append(node)
            trainingLabel.append(label)
        num += 1

#予測するためのデータをセット
def setPredictData():
    global testData
    global testLabel
    num = 0
    
    for row in numpyMatrix:
        node = []
        label = row[2]
        node.append(row[0])
        node.append(row[1])
        testData.append(node)
        num += 1

# svmにfitする形にmodelをセットする
def setLibSvmModel():
    model = svm.libsvm.fit( np.array( trainingData ), np.float64( np.array( trainingLabel ) ), kernel='linear' )
    return model

#modelを基に予測結果を出力する関数
def setSvmPredictLabelResult(model):
    predictLabel = svm.libsvm.predict( np.array( testData ), *model,  **{'kernel' : 'linear'} )    
    return predictLabel

# 詳細なテスト結果を表示する
# テストデータの答えがわかっている時に使える(setSampleData()を使っている時に利用できる)
def outputPredictResult():
    for i in range( 0, len(resultPredictLabel) ) :
        str = "ok" if( int( testLabel[i] ) == int( resultPredictLabel[i] ) ) else "miss"
        print "testLabel[%d] = %d ,learningLabel[%d] = %d" % (i,int(testLabel[i]),i,int(resultPredictLabel[i]))


csvSrc = '../src/test.csv'
dataFrame = importCSV(csvSrc)
numpyMatrix = dataFrameToMatrix(dataFrame)
setSampleData()
model = setLibSvmModel()
resultPredictLabel = setSvmPredictLabelResult(model)
outputPredictResult()
print "accuracy_score = %lf" % accuracy_score(testLabel, resultPredictLabel)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import itertools as it
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def importCSV(csvSrcPath):
    dataFrame = pd.read_csv(csvSrcPath)
    return dataFrame


def dataFrameToMatrix(dataFrame):
    dataFrame = dataFrame.astype('float64')
    matrix = dataFrame.as_matrix()
    return matrix


def toDataElement(row):
    return [row[0], row[2], row[3], row[4]]


def toLabelElement(row):
    return row[1]


def svmOneAgainstOne(trainingData,trainingLabel,testData):
    C = 1.
    kernel = 'rbf'
    gamma = 0.01

    classifier = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier.fit(trainingData, trainingLabel)
    resultPredictLabel = classifier.predict(testData)
    return resultPredictLabel


def svmOneAgainstRest(trainingData,trainingLabel,testData):
    C = 1.
    kernel = 'rbf'
    gamma = 0.01

    classifier = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier.fit(trainingData, trainingLabel)
    resultPredictLabel = classifier.predict(testData)
    return resultPredictLabel


def svmGridSearch(trainingData,trainingLabel):
    tuned_parameters = [{'kernel': ['rbf'], 
                         'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 
                         'C': [1, 10, 100, 1000]}]
    
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(trainingData, trainingLabel)
    print(clf.best_estimator_)


def getAccuracy(trainingMatrix,testMatrix):
    trainingData = [toDataElement(row)
                    for row in it.islice(trainingMatrix, 0, None, 2)]
    trainingLabel = [toLabelElement(row)
                     for row in it.islice(trainingMatrix, 0, None, 2)]
    testData = [toDataElement(row)
                for row in it.islice(testMatrix, 1, None, 2)]
    testLabel = [toLabelElement(row)
                 for row in it.islice(testMatrix, 1, None, 2)]
    resultPredictLabel = svmOneAgainstOne(trainingData,trainingLabel,testData)
    
    print "accuracy_score = %lf" % accuracy_score(testLabel, resultPredictLabel)


def getPredictFile(trainingMatrix,testMatrix):    
    trainingData = [toDataElement(row)
                    for row in trainingMatrix]
    trainingLabel = [toLabelElement(row)
                     for row in trainingMatrix]
    testData = [toDataElement(row)
                for row in testMatrix]
    resultPredictLabel = svmOneAgainstOne(trainingData,trainingLabel,testData)
    exportToCSV(testDataFrame, resultPredictLabel)


# 詳細なテスト結果を表示する
# テストデータの答えがわかっている時に使える(setSampleData()を使っている時に利用できる)


def outputPredictResult():
    for i in range(0, len(resultPredictLabel)):
        str = "ok" if(int(testLabel[i]) == int(
            resultPredictLabel[i])) else "miss"
        print "%s:testLabel[%d] = %d ,predictLabel[%d] = %d" % (str, i, int(testLabel[i]), i, int(resultPredictLabel[i]))

trainingCsvSrcPath = '../src/training_5column_apper.csv'
trainingDataFrame = importCSV(trainingCsvSrcPath)
trainingMatrix = dataFrameToMatrix(trainingDataFrame)

testCsvSrcPath = '../src/training_5column_apper.csv'
testDataFrame = importCSV(testCsvSrcPath)
testMatrix = dataFrameToMatrix(testDataFrame)

# True: トレーニングデータとテストデータを同じ割合で喰わせて精度比較を行う
# False:トレーニングデータとテストデータは別々で、予測結果をCSVファイルで出力
isAccuracy = True

if isAccuracy:
    getAccuracy(trainingMatrix,testMatrix)

else:
    getPredictFile(trainingMatrix,testMatrix)


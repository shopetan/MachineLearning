#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV


def importCSV(csvSrcPath):
    dataFrame = pd.read_csv(csvSrcPath)
    return dataFrame


def dataFrameToMatrix(dataFrame):
    dataFrame = dataFrame.astype('float64')
    matrix = dataFrame.as_matrix()
    return matrix


# 詳細なテスト結果を表示する
# テストデータの答えがわかっている時に使える(setSampleData()を使っている時に利用できる)


def outputPredictResult():
    for i in range(0, len(resultPredictLabel)):
        str = "ok" if(int(testLabel[i]) == int(
            resultPredictLabel[i])) else "miss"
        print "%s:testLabel[%d] = %d ,predictLabel[%d] = %d" % (str, i, int(testLabel[i]), i, int(resultPredictLabel[i]))

C = 1.
kernel = 'rbf'
gamma = 0.01

trainingCsvSrcPath = '../src/result.csv'
trainingDataFrame = importCSV(trainingCsvSrcPath)
trainingMatrix = dataFrameToMatrix(trainingDataFrame)

testCsvSrcPath = '../src/result.csv'
testDataFrame = importCSV(testCsvSrcPath)
testMatrix = dataFrameToMatrix(testDataFrame)

# True: トレーニングデータとテストデータを同じ割合で喰わせて精度比較を行う
# False:トレーニングデータとテストデータは別々で、予測結果をCSVファイルで出力
CheckaccuracyFlag = True

if CheckaccuracyFlag == True:
    trainingData = [[row[0], row[1]]
                    for i, row in enumerate(trainingMatrix)if i % 2 == 0]
    trainingLabel = [row[2] for i, row in enumerate(trainingMatrix)if i % 2 == 0]
    testData = [[row[0], row[1]]
                for i, row in enumerate(testMatrix)if i % 2 != 0]
    testLabel = [row[2] for i, row in enumerate(testMatrix) if i % 2 != 0]

    classifier = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier.fit(trainingData, trainingLabel)
    resultPredictLabel = classifier.predict(testData)

    print "accuracy_score = %lf" % accuracy_score(testLabel, resultPredictLabel)

else:
    trainingData = [[row[0], row[1]]
                    for i, row in enumerate(trainingMatrix)]
    trainingLabel = [row[2] for i, row in enumerate(trainingMatrix)]
    testData = [[row[1], row[2]]
                for i, row in enumerate(testMatrix)]
    classifier = SVC(C=C, kernel=kernel, gamma=gamma)
    classifier.fit(trainingData, trainingLabel)
    resultPredictLabel = classifier.predict(testData)

    exportToCSV(testDataFrame,resultPredictLabel)




"""
tuned_parameters = [{'kernel': ['rbf'], 
                     'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 
                     'C': [1, 10, 100, 1000]}]

clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1)
clf.fit(trainingData, trainingLabel)
"""


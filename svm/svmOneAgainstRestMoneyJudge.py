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


def importCSV(csvSrcPath):
    dataFrame = pd.read_csv(csvSrcPath)
    return dataFrame


def dataFrameToMatrix(dataFrame, sortColumn):
    dataFrame = dataFrame.astype('float64')
    dataFrame = dataFrame.sort(columns=sortColumn)
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
sortColumn = 'annualpay'

trainingCsvSrcPath = '../src/result.csv'
trainingDataFrame = importCSV(trainingCsvSrcPath)
trainingMatrix = dataFrameToMatrix(trainingDataFrame, sortColumn)

trainingData = [[row[0], row[1]]
                for i, row in enumerate(trainingMatrix) if i % 2 == 0]
trainingLabel = [row[2] for i, row in enumerate(trainingMatrix) if i % 2 == 0]
testData = [[row[0], row[1]]
            for i, row in enumerate(trainingMatrix) if i % 2 != 0]
testLabel = [row[2] for i, row in enumerate(trainingMatrix) if i % 2 != 0]

estimator = SVC(C=C, kernel=kernel, gamma=gamma)
classifier = OneVsRestClassifier(estimator)
classifier.fit(trainingData, trainingLabel)
resultPredictLabel = classifier.predict(testData)

print "accuracy_score = %lf" % accuracy_score(testLabel, resultPredictLabel)

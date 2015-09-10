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


def exportToCSV(DataFrame, label):
    df = pd.Series(label, name="annualpay")
    DataFrame = DataFrame.T.append(df).T
    DataFrame.to_csv("predict_svm.csv")


def toTrainingDataElement(row):
    return [row[0], row[2], row[3], row[4],row[5]]


def toTrainingLabelElement(row):
    return row[1]


def toTestDataElement(row):
    return [row[0], row[2], row[3], row[4],row[5]]


def toTestLabelElement(row):
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
    trainingData = [toTrainingDataElement(row)
                    for row in it.islice(trainingMatrix, 0, None, 2)]
    trainingLabel = [toTrainingLabelElement(row)
                     for row in it.islice(trainingMatrix, 0, None, 2)]
    testData = [toTestDataElement(row)
                for row in it.islice(testMatrix, 1, None, 2)]
    testLabel = [toTestLabelElement(row)
                 for row in it.islice(testMatrix, 1, None, 2)]
    resultPredictLabel = randomForest(trainingData,trainingLabel,testData)
    
    print "accuracy_score = %lf" % accuracy_score(testLabel, resultPredictLabel)


def getPredictFile(trainingMatrix,testMatrix):    
    trainingData = [toTrainingDataElement(row)
                    for row in trainingMatrix]
    trainingLabel = [toTrainingLabelElement(row)
                     for row in trainingMatrix]
    testData = [toTestDataElement(row)
                for row in testMatrix]
    resultPredictLabel = randomForest(trainingData,trainingLabel,testData)
    exportToCSV(testDataFrame, resultPredictLabel)


# 詳細なテスト結果を表示する
# テストデータの答えがわかっている時に使える(setSampleData()を使っている時に利用できる)


def outputPredictResult():
    for i in range(0, len(resultPredictLabel)):
        str = "ok" if(int(testLabel[i]) == int(
            resultPredictLabel[i])) else "miss"
        print "%s:testLabel[%d] = %d ,predictLabel[%d] = %d" % (str, i, int(testLabel[i]), i, int(resultPredictLabel[i]))

trainingCsvSrcPath = '../csv/training_6column.csv'
trainingDataFrame = importCSV(trainingCsvSrcPath)
trainingMatrix = dataFrameToMatrix(trainingDataFrame)

testCsvSrcPath = '../csv/training_6column.csv'
testDataFrame = importCSV(testCsvSrcPath)
testMatrix = dataFrameToMatrix(testDataFrame)

# True: トレーニングデータとテストデータを同じ割合で喰わせて精度比較を行う
# False:トレーニングデータとテストデータは別々で、予測結果をCSVファイルで出力
isAccuracy = True

if isAccuracy:
    getAccuracy(trainingMatrix,testMatrix)

else:
    getPredictFile(trainingMatrix,testMatrix)


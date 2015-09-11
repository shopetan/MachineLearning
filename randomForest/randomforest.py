#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import itertools as it
import dataprocessing.dataprocessing as dp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV


def toTrainingDataElement(row):
    return [row[0], row[2], row[3], row[4],row[5]]


def toTrainingLabelElement(row):
    return row[1]


def toTestDataElement(row):
    return [row[0], row[1], row[2], row[3],row[4]]


def toTestLabelElement(row):
    return row[1]


def randomForest(trainingData,trainingLabel,testData):
    model = RandomForestClassifier(bootstrap=True, compute_importances=None,
                                   criterion='gini', max_depth=10,
                                   max_features='auto', min_density=None,
                                   min_samples_leaf=1, min_samples_split=100,
                                   n_estimators=30, n_jobs=1, oob_score=False,
                                   random_state=0, verbose=0)
    model.fit(trainingData, trainingLabel)
    resultPredictLabel = model.predict(testData)
    return resultPredictLabel


def randomForestGridSearch(trainingData,trainingLabel):
    parameters = {
        'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
        'random_state'      : [0],
        'n_jobs'            : [1],
        'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],
        'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]
    }
    clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
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


def getPredictFile(trainingMatrix,testMatrix,testDataFrame):    
    trainingData = [toTrainingDataElement(row)
                    for row in trainingMatrix]
    trainingLabel = [toTrainingLabelElement(row)
                     for row in trainingMatrix]
    testData = [toTestDataElement(row)
                for row in testMatrix]
    resultPredictLabel = randomForest(trainingData,trainingLabel,testData)
    dp.exportToCSV(testDataFrame, resultPredictLabel)


# 詳細なテスト結果を表示する
# テストデータの答えがわかっている時に使える(setSampleData()を使っている時に利用できる)


def outputPredictResult(testLabel,resultPredictLabel):
    for i in range(0, len(resultPredictLabel)):
        str = "ok" if(int(testLabel[i]) == int(
            resultPredictLabel[i])) else "miss"
        print "%s:testLabel[%d] = %d ,predictLabel[%d] = %d" % (str, i, int(testLabel[i]), i, int(resultPredictLabel[i]))

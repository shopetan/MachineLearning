#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV


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


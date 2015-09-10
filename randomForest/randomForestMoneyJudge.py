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


def dataFrameToMatrix(dataFrame):
    dataFrame = dataFrame.astype('float64')
    matrix = dataFrame.as_matrix()
    return matrix


def exportToCSV(DataFrame,label):
    print DataFrame
    df = pd.Series(label,name="annualpay")
    DataFrame = DataFrame.T.append(df).T
    DataFrame.to_csv("predict_randomForest.csv")

# 詳細なテスト結果を表示する
# テストデータの答えがわかっている時に使える(setSampleData()を使っている時に利用できる)


def outputPredictResult():
    for i in range(0, len(resultPredictLabel)):
        str = "ok" if(int(testLabel[i]) == int(
            resultPredictLabel[i])) else "miss"
        print "%s:testLabel[%d] = %d ,predictLabel[%d] = %d" % (str, i, int(testLabel[i]), i, int(resultPredictLabel[i]))

trainingCsvSrcPath = '../src/result.csv'
trainingDataFrame = importCSV(trainingCsvSrcPath)
trainingMatrix = dataFrameToMatrix(trainingDataFrame)

testCsvSrcPath = '../src/result_test_data.csv'
testDataFrame = importCSV(testCsvSrcPath)
testMatrix = dataFrameToMatrix(testDataFrame)

# True: トレーニングデータとテストデータを同じ割合で喰わせて精度比較を行う
# False:トレーニングデータとテストデータは別々で、予測結果をCSVファイルで出力
CheckAccuracyFlag = False

if CheckAccuracyFlag == True:
    trainingData = [[row[0], row[1]]
                    for i, row in enumerate(trainingMatrix)if i % 2 == 0]
    trainingLabel = [row[2] for i, row in enumerate(trainingMatrix)if i % 2 == 0]
    testData = [[row[0], row[1]]
                for i, row in enumerate(testMatrix)if i % 2 != 0]
    testLabel = [row[2] for i, row in enumerate(testMatrix) if i % 2 != 0]

    model = RandomForestClassifier(bootstrap=True, compute_importances=None,
                                   criterion='gini', max_depth=10, 
                                   max_features='auto',min_density=None, 
                                   min_samples_leaf=1, min_samples_split=100,
                                   n_estimators=30, n_jobs=1, oob_score=False, 
                                   random_state=0,verbose=0)
    model.fit(trainingData, trainingLabel)
    resultPredictLabel = model.predict(testData)
    print "accuracy_score = %lf" % accuracy_score(testLabel, resultPredictLabel)

else:
    trainingData = [[row[0], row[1]]
                    for i, row in enumerate(trainingMatrix)]
    trainingLabel = [row[2] for i, row in enumerate(trainingMatrix)]
    testData = [[row[0], row[1]]
                for i, row in enumerate(testMatrix)]
    model = RandomForestClassifier(bootstrap=True, compute_importances=None,
                                   criterion='gini', max_depth=10, 
                                   max_features='auto',min_density=None, 
                                   min_samples_leaf=1, min_samples_split=100,
                                   n_estimators=30, n_jobs=1, oob_score=False, 
                                   random_state=0,verbose=0)
    model.fit(trainingData, trainingLabel)
    resultPredictLabel = model.predict(testData)

    exportToCSV(testDataFrame,resultPredictLabel)


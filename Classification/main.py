#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataprocessing.dataprocessing as dp
import randomForest.randomforest as rd


trainingCsvSrcPath = 'csv/training_6column.csv'
trainingDataFrame = dp.importCSV(trainingCsvSrcPath)
trainingMatrix = dp.dataFrameToMatrix(trainingDataFrame)

testCsvSrcPath = 'csv/test_data_6column.csv'
testDataFrame = dp.importCSV(testCsvSrcPath)
testMatrix = dp.dataFrameToMatrix(testDataFrame)


# True: トレーニングデータとテストデータを同じ割合で喰わせて精度比較を行う
# False:トレーニングデータとテストデータは別々で、予測結果をCSVファイルで出力
isAccuracy = False

if isAccuracy:
    rd.getAccuracy(trainingMatrix,testMatrix)

else:
    rd.getPredictFile(trainingMatrix,testMatrix,testDataFrame)

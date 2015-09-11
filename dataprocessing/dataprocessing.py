#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


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
    DataFrame.to_csv("predict_randomForest.csv")

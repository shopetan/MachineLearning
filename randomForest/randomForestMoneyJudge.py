#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

training_data = []
training_label = []
predict_data = []
predict_label = []

num = 0
df = pd.read_csv('test.csv')
df = df.astype('float64')
df = df.sort(columns='annu')

print df

numpyMatrix = df.as_matrix()

for row in numpyMatrix:
    node = []
    label = row[2]
    node.append(row[0])
    node.append(row[1])
    if num % 2 == 0:
        training_data.append(node)
        training_label.append(label)
    else:
        predict_data.append(node)
        predict_label.append(label)
    num += 1

predict_data = training_data
model = RandomForestClassifier()
model.fit(training_data, training_label)
output = model.predict(predict_data)

hit = 0
total = len(output)

for i in range( 0, len(output) ) :
    str = "ok" if( int( predict_label[i] ) == int( output[i] ) ) else "miss"
    if str == "ok":
        hit += 1
    print "predict_label[%d] = %d ,output[%d] = %d" % (i,int(predict_label[i]),i,int(output[i]))

accuracy = float(hit) / total
print "accuracy = %lf" % accuracy

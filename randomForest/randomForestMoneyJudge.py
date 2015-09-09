#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score


training_data = []
training_label = []
predict_data = []
predict_label = []

num = 0
df = pd.read_csv('../src/result.csv')
df = df.astype('float64')
df = df.sort(columns='annualpay')

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
    try:
        str = "ok" if( int( predict_label[i] ) == int( output[i] ) ) else "miss"
        if str == "ok":
            hit += 1
            print "predict_label[%d] = %d ,output[%d] = %d" % (i,int(predict_label[i]),i,int(output[i]))
    except IndexError:
        hit -= 1
        total -= 1
        print "例外発生: %d" % i

print accuracy_score(predict_label, output)
        
accuracy = float(hit) / total
print "hit = %ls" % hit 
print "total = %ls" % total 
print "accuracy = %lf" % accuracy

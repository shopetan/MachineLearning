#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re,numpy as np
import pandas as pd
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

training_data = []
training_label = []
predict_data = []
predict_label = []

C = 1.
kernel = 'rbf'
gamma  = 0.01

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

print num
estimator = SVC(C=C, kernel=kernel, gamma=gamma)
classifier = OneVsRestClassifier(estimator)
classifier.fit(training_data, training_label)
output = classifier.predict(predict_data)

#model = svm.libsvm.fit( np.array( training_data ), np.float64( np.array( training_label ) ), kernel='linear' )
#output = svm.libsvm.predict( np.array( predict_data ), *model,  **{'kernel' : 'linear'} )

for i in range( 0, len(output) ) :
    str = "ok" if( int( predict_label[i] ) == int( output[i] ) ) else "miss"
    print "predict_label[%d] = %d ,output[%d] = %d" % (i,int(predict_label[i]),i,int(output[i]))
    
print accuracy_score(predict_label, output)


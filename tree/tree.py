import itertools
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import tree

def main():
    iris = datasets.load_iris()
    features = iris.data
    feature_names = iris.feature_names
    targets = iris.target
    
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(features, targets)
    
    tree.export_graphviz(classifier, out_file='tree.dot') 
    
if __name__ == '__main__':
    main()

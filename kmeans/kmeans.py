#encoding: utf-8

import itertools
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets
import sklearn.decomposition
from sklearn.cluster import KMeans

def main():
    iris = datasets.load_iris()
    features = iris.data
    feature_names = iris.feature_names
    targets = iris.target

    #PCA
    pca = sklearn.decomposition.PCA(2)    
    pca = pca.fit_transform(features);
    
    kmeans_model = KMeans(n_clusters=3, random_state=10).fit(pca)
    labels = kmeans_model.labels_

    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('IrisDataをPCA(2)で圧縮しk-meansでクラスタリング'.decode("utf-8") , size=16)
    ax.set_xlabel('IrisDataをPCA(2)で圧縮したx座標'.decode("utf-8") , size=14)
    ax.set_ylabel('IrisDataをPCA(2)で圧縮したy座標'.decode("utf-8") , size=14)

    for label, pca in zip(labels, pca):
        if   label == 0:
            plt.scatter(pca[0], pca[1],c='r')
        elif label == 1:
            plt.scatter(pca[0], pca[1],c='g')
        elif label == 2:
            plt.scatter(pca[0], pca[1],c='b')

    plt.show()

if __name__ == '__main__':
    main()

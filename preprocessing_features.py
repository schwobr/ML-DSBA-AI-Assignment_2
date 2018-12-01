import numpy as np
import pandas as pd
from sklearn import preprocessing


def deleteUseless(X, features):
    i = getColumn(features, "PassengerId")
    if i != -1:
        X = np.delete(X, i, 1)
        features = np.delete(features, i)
    i = getColumn(features, "Ticket")
    if i != -1:
        X = np.delete(X, i, 1)
        features = np.delete(features, i)
    i = getColumn(features, "Cabin")
    if i != -1:
        X = np.delete(X, i, 1)
        features = np.delete(features, i)
    return (X, features)


def getColumn(features, feature):
    for i in range(len(features)):
        if features[i] == feature:
            return i
    return -1


def changeGender(X, features):
    j = getColumn(features, "Sex")
    if j != -1:
        for i in range(X.shape[0]):
            X[i, j] = 0 if X[i, j] == "male" else 1


def ChangeEmbarked(X, features):
    j = getColumn(features, "Embarked")
    if j != -1:
        for i in range(X.shape[0]):
            X[i, j] = 0 if X[i, j] == "C" else 1 if X[i, j] == "Q" else 2


def ChangeName(X, features):
    j = getColumn(features, "Name")
    if j != -1:
        titles = np.copy(X[:, j])
        for i in range(titles.shape[0]):
            split = titles[i].split(' ')
            for s in split:
                if s[-1] == '.':
                    titles[i] = s
                    break
        names = np.copy(titles)
        titles = np.unique(titles)
        X[:, j] = names
        for i in range(X.shape[0]):
            title = X[i, j]
            for k in range(len(titles)):
                if title == titles[k]:
                    X[i, j] = k
                    break
        features[j] = "Title"


def missingAges(X, features):
    j = getColumn(features, "Age")
    mean = np.mean(np.array([float(x) for x in X[:, j] if x != '']))
    mean = round(mean, 1)
    for i in range(X.shape[0]):
        if X[i, j] == '':
            X[i, j] = mean


def preprocess(X, features):
    X, features = deleteUseless(X, features)
    changeGender(X, features)
    ChangeEmbarked(X, features)
    ChangeName(X, features)
    missingAges(X, features)
    return (preprocessing.scale(np.array(X, dtype='float64')), features)

import numpy as np
import pandas as pd
from sklearn import preprocessing


def change_gender(data):
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1


def change_embarked(data):
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 2
    data.loc[data['Embarked'].isnull(), 'Embarked'] = data['Embarked'].value_counts().index[0]


def change_name(data):
    l = [s for l in data['Name'].str.split(' ') for s in l if ('.' in s and 'L.' not in s)]
    data['Name'] = l
    l = list(set(l))
    n = data.index.values[0]
    for i in range(n, data['Name'].size + n):
        for j in range(len(l)):
            if l[j] == data.at[i, 'Name']:
                data.at[i, 'Name'] = j


def missing_ages(data):
    mean = round(data['Age'].mean(), 1)
    data.loc[data['Age'].isnull(), 'Age'] = mean


def fare_NaN(data):
    if data.isna().any()['Fare']:
        data.loc[data['Fare'].isnull(), 'Fare'] = round(data['Fare'].mean(), 1)

def age_classes(data):
    data_bis = data.copy()
    data_bis.loc[data['Age'] > 80, 'Age'] = 0
    data_bis.loc[data['Age'] <= 80, 'Age'] = 1
    data_bis.loc[data['Age'] <= 60, 'Age'] = 2
    data_bis.loc[data['Age'] <= 45, 'Age'] = 3
    data_bis.loc[data['Age'] <= 30, 'Age'] = 4
    data_bis.loc[data['Age'] <= 15, 'Age'] = 5
    data_bis.loc[data['Age'] <= 5, 'Age'] = 6
    data['Age']=data_bis['Age']
    
    
def preprocess(data, change_ages = False):
    data.drop(['Ticket', 'Cabin', 'Fare', 'Embarked'], axis=1, inplace=True)
    change_gender(data)
    #change_embarked(data)
    change_name(data)
    missing_ages(data)
    if change_ages:
        age_classes(data)
    #fare_NaN(data)
    return preprocessing.scale(data.values)

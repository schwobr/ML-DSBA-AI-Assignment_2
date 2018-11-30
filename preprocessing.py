import numpy as np
import pandas as pd
from sklearn import preprocessing


def changeGender(data):
    data.loc[data['Sex'] == 'male', 'Sex'] = 0
    data.loc[data['Sex'] == 'female', 'Sex'] = 1

def ChangeEmbarked (data):
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 2
    data.loc[data['Embarked'].isnull(), 'Embarked'] = data['Embarked'].value_counts().index[0]

def ChangeName (data):
    l=[s for l in data['Name'].str.split(' ') for s in l if ('.' in s and 'L.' not in s)]
    data['Name']=l
    for i in range(1,data['Name'].size+1):
        for j in range(len(l)):        
            if(l[j]==data.at[i,'Name']):
                data.at[i,'Name']=j

def missingAges(data):
    mean = round(data['Age'].mean(),1)
    data.loc[data['Age'].isnull(), 'Age'] = mean

def preprocess(data):
    data.drop(['Ticket','Cabin'], axis=1, inplace=True)
    changeGender(data)
    ChangeEmbarked(data)
    ChangeName(data)
    missingAges(data)
    print(data.head())
    return (preprocessing.scale(data.values))

#%%
import numpy as np
from sklearn import model_selection
import csv 
from classify import classify
import preprocessing as prep
import matplotlib.pyplot as plt

#%%
# Load data
csv_file_object = csv.reader(open('Data\\train.csv', 'r')) # Load in the csv file
header = next(csv_file_object)				  # Skip the fist line as it is a header
data=[] 											  # Create a variable to hold the data

#%%
for row in csv_file_object: # Skip through each row in the csv file,
    data.append(row[0:]) 	# adding each row to the data variable
X = np.array(data) 		    # Then convert from a list to an array.
y = X[:,1].astype(int) # Save labels to y 

#%%
X = np.delete(X,1,1) # Remove survival column from matrix X
header = np.delete(header, 1)
# Initialize cross validation
kf = model_selection.KFold(n_splits=10)

#%% basic classifier
totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances  
for trainIndex, testIndex in kf.split(X):
    trainSet = X[trainIndex]
    testSet = X[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
	
    predictedLabels = classify(trainSet, trainLabels, testSet)

    correct = 0	
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print ('Accuracy: ' + str(float(correct)/(testLabels.size)))
    totalCorrect += correct
    totalInstances += testLabels.size
print ('Total Accuracy: ' + str(totalCorrect/float(totalInstances)))

#%% Preprocessing 
X, header = prep.preprocess(X, header)

#%% Test with decision tree
from sklearn.tree import DecisionTreeClassifier

def decisionTree(D):
    totalInstances = 0 # Variable that will store the total intances that will be tested  
    totalCorrect = 0 # Variable that will store the correctly predicted intances  
    clf = DecisionTreeClassifier(max_depth=D)
    for trainIndex, testIndex in kf.split(X):
        trainSet = X[trainIndex]
        testSet = X[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
        
        clf.fit(trainSet, trainLabels)
        predictedLabels = clf.predict(testSet)

        correct = 0	
        for i in range(testSet.shape[0]):
            if predictedLabels[i] == testLabels[i]:
                correct += 1
            
        totalCorrect += correct
        totalInstances += testLabels.size
        accuracy = totalCorrect/float(totalInstances)
    print ('Total Accuracy: ' + str(accuracy))
    return accuracy

#%% Testing Decision Tree for different depths (best result with D=5)
Ds = range(2,15)
accuracys = []
for D in Ds:
    print(D)
    accuracys.append(decisionTree(D))

plt.plot(Ds, accuracys, label = "accuracy % D")
plt.show()

#%% Adaboost with Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def adaBoost(D):
    totalInstances = 0 # Variable that will store the total intances that will be tested  
    totalCorrect = 0 # Variable that will store the correctly predicted intances  
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=D))
    for trainIndex, testIndex in kf.split(X):
        trainSet = X[trainIndex]
        testSet = X[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
        
        clf.fit(trainSet, trainLabels)
        predictedLabels = clf.predict(testSet)

        correct = 0	
        for i in range(testSet.shape[0]):
            if predictedLabels[i] == testLabels[i]:
                correct += 1
            
        totalCorrect += correct
        totalInstances += testLabels.size
        accuracy = totalCorrect/float(totalInstances)
    print ('Total Accuracy: ' + str(accuracy))
    return accuracy

#%% Adaboost Test for Different values of D (best with D=2)
Ds = range(2,15)
accuracys = []
for D in Ds:
    print(D)
    accuracys.append(adaBoost(D))

plt.plot(Ds, accuracys, label = "accuracy % D")
plt.show()

#%% Neural Network 
from sklearn.neural_network import MLPClassifier

def NN(hl_sizes=(100,), activation='relu', solver='sgd', lr=0.01, lr_evol='constant', max_iter=200, tol=0.001, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5):
    totalInstances = 0 # Variable that will store the total intances that will be tested  
    totalCorrect = 0 # Variable that will store the correctly predicted intances  
    clf = MLPClassifier(hidden_layer_sizes=hl_sizes, activation=activation, solver=solver , learning_rate_init=lr, learning_rate=lr_evol, max_iter=max_iter, tol= tol, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change)
    for trainIndex, testIndex in kf.split(X):
        trainSet = X[trainIndex]
        testSet = X[testIndex]
        trainLabels = y[trainIndex]
        testLabels = y[testIndex]
        
        clf.fit(trainSet, trainLabels)
        predictedLabels = clf.predict(testSet)

        correct = 0	
        for i in range(testSet.shape[0]):
            if predictedLabels[i] == testLabels[i]:
                correct += 1
            
        totalCorrect += correct
        totalInstances += testLabels.size
        accuracy = totalCorrect/float(totalInstances)
    print ('Total Accuracy: ' + str(accuracy))
    return accuracy

#%% NN Test with sgd, different constant lr, 1 hidden layer of varying size 
lrs = [(2**n)*0.0001 for n in range(11)]
sizes = [(20+10*n,) for n in range(20)]
accuracies=np.zeros((len(lrs), len(sizes)))

for i in range(len(lrs)):
    for j in range(len(sizes)):
        accuracies[i,j]=NN(hl_sizes=sizes[j], lr=lrs[i])

idx = np.argsort(accuracies, axis=0)
plt.figure(1)
plt.plot(sizes, [lrs[i] for i in idx[-1,:]], label="best learning rate for each hidden layer size")
plt.figure(2)
plt.plot(sizes, [accuracies[idx[-1,i],i] for i in range(len(sizes))], label="corresponding accuracies")
plt.show()

#%% NN test for higher hidden layer sizes (from 200 to 400)
lrs = [(2**n)*0.0001 for n in range(11)]
sizes = [(200+10*n,) for n in range(20)]
accuracies=np.zeros((len(lrs), len(sizes)))

for i in range(len(lrs)):
    for j in range(len(sizes)):
        accuracies[i,j]=NN(hl_sizes=sizes[j], lr=lrs[i])

idx = np.argsort(accuracies, axis=0)
plt.figure(1)
plt.plot(sizes, [lrs[i] for i in idx[-1,:]], label="best learning rate for each hidden layer size")
plt.figure(2)
plt.plot(sizes, [accuracies[idx[-1,i],i] for i in range(len(sizes))], label="corresponding accuracies")
plt.show()
import numpy as np

def classify(trainSet, trainLabels, testSet):
	
	predictedLabels = np.zeros(testSet.shape[0])
	
	for i in range(testSet.shape[0]):
		if testSet[i,3] == 'female':
			predictedLabels[i] = 1

	return predictedLabels
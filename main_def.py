# %%
import numpy as np
from sklearn import model_selection
import csv
from classify import classify
import preprocessing as prep
import matplotlib.pyplot as plt


def load_data(file="Data/train.csv"):
    """
    read the data from a file and return x, y and x headers. Using Data/train.csv by default
    :param file:
    :return: x: input data
             y: label
             x_header: label of columns x
    """
    # Load data
    csv_file_object = csv.reader(open(file, 'r'))  # Load in the csv file
    x_header = next(csv_file_object)  # Skip the fist line as it is a header
    data = []  # Create a variable to hold the data

    # %%
    for row in csv_file_object:  # Skip through each row in the csv file,
        data.append(row[0:])  # adding each row to the data variable
    x = np.array(data)  # Then convert from a list to an array.
    y = x[:, 1].astype(int)  # Save labels to y

    # %%
    x = np.delete(x, 1, 1)  # Remove survival column from matrix X
    x_header = np.delete(x_header, 1)
    return x, y, x_header


class Classifier:
    def __init__(self):
        self.kf = model_selection.KFold(n_splits=10)
        self.x, self.y, self.x_header = load_data()

    def basic_classifier(self):
        """
        basic classifier given as example in the Assigment_2 zip file
        :return:
        """

        total_instances = 0  # Variable that will store the total intances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted intances
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]

            predicted_labels = classify(train_set, train_labels, test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            print('Accuracy: ' + str(float(correct) / test_labels.size))
            total_correct += correct
            total_instances += test_labels.size
        print('Total Accuracy: ' + str(total_correct / float(total_instances)))

    def preprocessing(self):
        self.x, self.x_header = prep.preprocess(self.x, self.x_header)

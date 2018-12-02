# %%
import numpy as np
from sklearn import model_selection
import csv
from classify import classify
import preprocessing as prep
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


class Classifier:
    def __init__(self):
        self.kf = model_selection.KFold(n_splits=10)
        self.x = None
        self.y = None
        self.x_header = None
        self.x_test = None
        self.y_test = None
        self.data = None
        self.data_test = None
        self.clf = None
        self.pca = PCA(n_components=0.85, svd_solver="full")
        self.feat_sel = SelectKBest(mutual_info_classif, k=4)

    def load_data(self, file="Data/train.csv"):
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
        self.x = x
        self.y = y
        self.x_header = x_header

    def load_data_panda(self, file="Data/train.csv"):
        """
        read the data from a file and return it using panda
        :param file: path to csv
        :param display: Bool. False by default. Set to true to print the data
        :return: data
        """
        data = pd.read_csv(file, index_col='PassengerId')  # Load in the csv file
        y = data['Survived']
        self.data = data.drop('Survived', axis=1)
        self.x_header = list(data)
        self.x = data.values
        self.y = y.values

    def load_test(self, file="Data/test.csv"):
        """
        read the test data from a file and return it using panda
        :param file: path to csv
        :param display: Bool. False by default. Set to true to print the data
        :return: data
        """
        self.data_test = pd.read_csv(file, index_col="PassengerId")
        self.x_test = self.data_test.values

    def apply_pca(self):
        self.pca.fit_transform(self.x)

    def apply_feat_sel(self):
        self.feat_sel.fit_transform(self.x, self.y)

    def basic_classifier(self):
        """
        basic classifier given as example in the Assigment_2 zip file
        :return:
        """

        total_instances = 0  # Variable that will store the total instances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted instances
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

    def preprocessing(self, change_ages=False):
        self.x = prep.preprocess(self.data, change_ages)

    def decision_tree(self, D):
        total_instances = 0  # Variable that will store the total instances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted instances
        self.clf = DecisionTreeClassifier(max_depth=D)
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]

            self.clf.fit(train_set, train_labels)
            predicted_labels = self.clf.predict(test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            total_correct += correct
            total_instances += test_labels.size
        accuracy = total_correct / float(total_instances)
        print('Total Accuracy: ' + str(accuracy))
        return accuracy

    def ada_boost(self, D):
        total_instances = 0  # Variable that will store the total instances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted instances
        self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=D))
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]

            self.clf.fit(train_set, train_labels)
            predicted_labels = self.clf.predict(test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            total_correct += correct
            total_instances += test_labels.size
        accuracy = total_correct / float(total_instances)
        print('Total Accuracy: ' + str(accuracy))
        return accuracy

    def NN(self, hl_sizes=(100,), activation='relu', solver='sgd', lr=0.01, lr_evol='constant', max_iter=200, tol=0.001,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=5):
        total_instances = 0  # Variable that will store the total instances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted instances
        self.clf = MLPClassifier(hidden_layer_sizes=hl_sizes, activation=activation, solver=solver,
                                 learning_rate_init=lr,
                                 learning_rate=lr_evol, max_iter=max_iter, tol=tol, early_stopping=early_stopping,
                                 validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change)
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]

            self.clf.fit(train_set, train_labels)
            predicted_labels = self.clf.predict(test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            total_correct += correct
            total_instances += test_labels.size
        accuracy = total_correct / float(total_instances)
        print('Total Accuracy: ' + str(accuracy))
        return accuracy

    def LDA(self):
        totalInstances = 0  # Variable that will store the total intances that will be tested
        totalCorrect = 0  # Variable that will store the correctly predicted intances
        self.clf = LDA(solver='eigen')
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]

            self.clf.fit(train_set, train_labels)
            self.clf.transform(test_set)
            predicted_labels = self.clf.predict(test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            totalCorrect += correct
            totalInstances += test_labels.size
        accuracy = totalCorrect / float(totalInstances)
        print("Total accuracy : ", str(accuracy))
        return accuracy

    def SVM(self):
        total_instances = 0  # Variable that will store the total instances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted instances
        self.clf = svm.SVC(gamma='scale')
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]
            self.clf.fit(train_set, train_labels)
            predicted_labels = self.clf.predict(test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            total_correct += correct
            total_instances += test_labels.size
        accuracy = total_correct / float(total_instances)
        print(accuracy)
        return accuracy

    def KNN(self):
        total_instances = 0  # Variable that will store the total instances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted instances
        self.clf = KNeighborsClassifier(n_neighbors=5)
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]
            self.clf.fit(train_set, train_labels)
            predicted_labels = self.clf.predict(test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            total_correct += correct
            total_instances += test_labels.size
        accuracy = total_correct / float(total_instances)
        print("Total accuracy : ", str(accuracy))
        return accuracy

    def random_forest(self):
        total_instances = 0  # Variable that will store the total instances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted instances
        self.clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=0)
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]
            self.clf.fit(train_set, train_labels)
            predicted_labels = self.clf.predict(test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            total_correct += correct
            total_instances += test_labels.size
        accuracy = total_correct / float(total_instances)
        print("Total accuracy : ", str(accuracy))
        return accuracy

    def quadri_discriminant(self):
        total_instances = 0  # Variable that will store the total instances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted instances
        self.clf = QuadraticDiscriminantAnalysis()
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]
            self.clf.fit(train_set, train_labels)
            predicted_labels = self.clf.predict(test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            total_correct += correct
            total_instances += test_labels.size
        accuracy = total_correct / float(total_instances)
        print("Total accuracy : ", str(accuracy))
        return accuracy

    def gaussian_process(self):
        total_instances = 0  # Variable that will store the total instances that will be tested
        total_correct = 0  # Variable that will store the correctly predicted instances
        kernel = 1.0 * RBF(1.0)
        self.clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
        for trainIndex, testIndex in self.kf.split(self.x):
            train_set = self.x[trainIndex]
            test_set = self.x[testIndex]
            train_labels = self.y[trainIndex]
            test_labels = self.y[testIndex]
            self.clf.fit(train_set, train_labels)
            predicted_labels = self.clf.predict(test_set)

            correct = 0
            for i in range(test_set.shape[0]):
                if predicted_labels[i] == test_labels[i]:
                    correct += 1

            total_correct += correct
            total_instances += test_labels.size
        accuracy = total_correct / float(total_instances)
        print("Total accuracy : ", str(accuracy))
        return accuracy

    def test(self, pca=False, feat_sel=False, change_ages=False):
        self.x_test = prep.preprocess(self.data_test, change_ages)
        if pca:
            self.pca.transform(self.x_test)
        if feat_sel:
            self.feat_sel.transform(self.x_test)
        self.y_test = self.clf.predict(self.x_test)

    def generate_submission(self, submission_file='Data/submission.csv'):
        if self.clf is None:
            raise NameError("clf have to be computed before generating a submission")
        y_df = pd.DataFrame(data=self.y_test, columns=['Survived'], index=self.data_test.index)
        print(y_df.head(20))
        y_df.to_csv(path_or_buf=submission_file)

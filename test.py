from main import *
import matplotlib.pyplot as plt


def test_load_data():
    """
    Test load_data by printing the data read
    """
    classifier = Classifier()
    classifier.load_data()
    print(classifier.x)
    print(classifier.y)
    print(classifier.x_header)


#  test_load_data()


def test_load_data_panda():
    """
    Test load_data_panda by printing the data read
    """
    classifier = Classifier()
    classifier.load_data_panda()
    print(classifier.x)
    print(classifier.y)


#  test_load_data_panda()


def test_basic_classifier():
    """
    Test the basic classifier given
    """
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.basic_classifier()


#  test_basic_classifier()

def test_decision_tree():
    """
    Testing Decision Tree for different depths (best result with D=5)
    :return: show plot
    """
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.preprocessing()
    Ds = range(2, 15)
    accuracys = []
    for D in Ds:
        print(D)
        accuracys.append(classifier.decision_tree(D))

    plt.plot(Ds, accuracys, label="accuracy % D")
    plt.show()


#  test_decision_tree()

def test_ada_boot():
    """
    Adaboost Test for Different values of D (best with D=2)
    :return: show plot
    """
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.preprocessing()
    Ds = range(2, 15)
    accuracys = []
    for D in Ds:
        print(D)
        accuracys.append(classifier.ada_boost(D))

    plt.plot(Ds, accuracys, label="accuracy % D")
    plt.show()


#  test_ada_boot()

def test_NN_1():
    """
    NN Test with sgd, different constant lr, 1 hidden layer of varying size
    :return: show plot
    """
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.preprocessing()
    lrs = [(2 ** n) * 0.0001 for n in range(11)]
    sizes = [(20 + 10 * n,) for n in range(20)]
    accuracies = np.zeros((len(lrs), len(sizes)))

    for i in range(len(lrs)):
        for j in range(len(sizes)):
            accuracies[i, j] = classifier.NN(hl_sizes=sizes[j], lr=lrs[i])

    idx = np.argsort(accuracies, axis=0)
    plt.figure(1)
    plt.plot(sizes, [lrs[i] for i in idx[-1, :]], label="best learning rate for each hidden layer size")
    plt.figure(2)
    plt.plot(sizes, [accuracies[idx[-1, i], i] for i in range(len(sizes))], label="corresponding accuracies")
    plt.show()


#  test_NN_1()


def test_NN_2():
    """
    NN test for higher hidden layer sizes (from 200 to 400)
    :return: show plot
    """
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.preprocessing()
    lrs = [(2 ** n) * 0.0001 for n in range(11)]
    sizes = [(200 + 10 * n,) for n in range(20)]
    accuracies = np.zeros((len(lrs), len(sizes)))

    for i in range(len(lrs)):
        for j in range(len(sizes)):
            accuracies[i, j] = classifier.NN(hl_sizes=sizes[j], lr=lrs[i])

    idx = np.argsort(accuracies, axis=0)
    plt.figure(1)
    plt.plot(sizes, [lrs[i] for i in idx[-1, :]], label="best learning rate for each hidden layer size")
    plt.figure(2)
    plt.plot(sizes, [accuracies[idx[-1, i], i] for i in range(len(sizes))], label="corresponding accuracies")
    plt.show()


#  test_NN_2()

def test_LDA():
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.preprocessing()
    classifier.LDA()


#  test_LDA()


def test_SVM():
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.load_test()
    classifier.preprocessing(change_ages=True)
    classifier.apply_pca()
    classifier.SVM()
    classifier.test()
    classifier.generate_submission(submission_file="Data/submission_svm_nsplits-5.csv")


#  test_SVM()


def test_KNN():
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.load_test()
    classifier.preprocessing(change_ages=True)
    classifier.apply_pca()
    classifier.KNN()
    classifier.test()
    classifier.generate_submission(submission_file="Data/submission_knn.csv")


#  test_KNN()

def test_random_forest():
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.load_test()
    classifier.preprocessing(change_ages=True)
    classifier.random_forest()
    classifier.test()
    classifier.generate_submission(submission_file="Data/submission_random_forest_2.csv")


#   test_random_forest()

def test_quadri_discriminant():
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.load_test()
    classifier.preprocessing(change_ages=True)
    classifier.quadri_discriminant()
    classifier.test()
    classifier.generate_submission(submission_file="Data/submission_quadri_discriminant.csv")


#  test_quadri_discriminant()

def test_gaussian_process():
    classifier = Classifier()
    classifier.load_data_panda()
    classifier.load_test()
    classifier.preprocessing(change_ages=True)
    classifier.gaussian_process()
    classifier.test()
    classifier.generate_submission(submission_file="Data/submission_gaussian_process_nsplit.csv")


#  test_gaussian_process()

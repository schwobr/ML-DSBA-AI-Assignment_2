from main_def import *


def test_load_data():
    """
    Test load_data by printing the data read
    """
    print(load_data())


#  test_load_data()


def test_basic_classifier():
    """
    Test the basic classifier given
    """
    classifier = Classifier()
    classifier.basic_classifier()


#  test_basic_classifier()

def test_decision_tree():
    """
    Testing Decision Tree for different depths (best result with D=5)
    :return: show plot
    """
    classifier = Classifier()
    classifier.preprocessing_features()
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
    classifier.preprocessing_features()
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
    try:
        classifier = Classifier()
        classifier.preprocessing_features()
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
    except KeyboardInterrupt as e:
        print(e)


#  test_NN_1()


def test_NN_2():
    """
    NN test for higher hidden layer sizes (from 200 to 400)
    :return: show plot
    """
    try:
        classifier = Classifier()
        classifier.preprocessing_features()
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
    except KeyboardInterrupt as e:
        print(e)

#  test_NN_2()

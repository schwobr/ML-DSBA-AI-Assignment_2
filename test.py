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


test_decision_tree()

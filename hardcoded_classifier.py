# CS450 Prove01
# James Palmer

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class HardcodedClassifier:
    def __init__(self):
        pass
    
    def fit(self, data, target):
        return HardcodedClassifier()
    
    def predict(self, data_test):
        return np.zeros((data_test.shape[0]), dtype = int)
    
    def score(self, x_test, y_test, sample_weight=None):
        return accuracy_score(y_test, self.predict(x_test), sample_weight = sample_weight)


def main():
    iris = datasets.load_iris()
    
    iris_data = iris.data
    iris_target = iris.target
    
    data_train, data_test, target_train, target_test = \
        train_test_split(iris_data, iris_target, test_size = .3)

    model_iris_gaussian_nb(data_test, data_train, target_test, target_train)
    model_iris_hardcoded(data_test, data_train, target_test, target_train)


def model_iris_hardcoded(data_test, data_train, target_test, target_train):
    hc_classifier = HardcodedClassifier()
    hc_classifier.fit(data_train, target_train)
    hcc_score = hc_classifier.score(data_test, target_test)
    print("(Hardcoded) Predicted accuracy: {:.02f}%".format(hcc_score * 100))


def model_iris_gaussian_nb(data_test, data_train, target_test, target_train):
    gnb_classifier = GaussianNB()
    gnb_classifier.fit(data_train, target_train)
    gnb_score = gnb_classifier.score(data_test, target_test)
    print("(GaussianNB) Predicted accuracy: {:.02f}%".format(gnb_score * 100))


if __name__ == '__main__':
    main()

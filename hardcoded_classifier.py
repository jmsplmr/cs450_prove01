import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    iris = datasets.load_iris()
    
    iris_data = iris.data
    iris_target = iris.target
    # iris_target_names = iris.target_names
    
    data_train, data_test, target_train, target_test = \
        train_test_split(iris_data, iris_target, test_size = .3)
    
    classifier = GaussianNB()
    model = classifier.fit(data_train, target_train)
    targets_predicted = model.predict(data_test)
    
    print(targets_predicted)
    
    number_correct = np.sum(targets_predicted == target_test)
    total_tests = len(target_test)
    print(number_correct / total_tests)


if __name__ == '__main__':
    main()

"""
James Palmer
CS450
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, k, data=None, target=None):
        if target is None:
            target = []
        if data is None:
            data = []
        self.k = k
        self.data = data
        self.target = target
    
    def fit(self, data, target):
        self.data = data
        self.target = target
        return self
    
    def predict(self, test_data):
        n_inputs = np.shape(test_data)[0]
        closest = np.zeros(n_inputs)
        
        for n in range(n_inputs):
            distances = np.sum((self.data - test_data[n, :]) ** 2, axis = 1)
            indices = np.argsort(distances, axis = 0)
            classes = np.unique(self.target[indices[:self.k]])
            
            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(self.k):
                    counts[self.target[indices[i]]] += 1
                closest[n] = np.max(counts)
        return closest
    
    def score(self, x_test, y_test, sample_weight=None):
        return accuracy_score(y_test, self.predict(x_test), sample_weight = sample_weight)


def main():
    k = 5
    iris_data_set = load_iris()
    iris_data = iris_data_set.data
    iris_target = iris_data_set.target
    x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_target, test_size = 0.2)
    
    sklearn_nkk = KNeighborsClassifier(n_neighbors = k)
    sklearn_nkk_model = sklearn_nkk.fit(x_train, y_train)
    sklearn_nkk_model.predict(x_test)
    print('> SKLearn nKK: {}'.format(100 * sklearn_nkk_model.score(x_test, y_test)))
    
    my_knn = KNNClassifier(k)
    my_knn_model = my_knn.fit(x_train, y_train)
    # my_knn_model.predict(x_test)
    print('> Implementation of nKK: {}'.format(100 * my_knn_model.score(x_test, y_test)))


if __name__ == '__main__':
    main()

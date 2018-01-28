import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


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
            distances = np.sum((self.data - test_data[n, :]) ** 2, axis=1)
            indices = np.argsort(distances, axis=0)
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
        return accuracy_score(y_test, self.predict(x_test), sample_weight=sample_weight)


def main():
    car_data, car_test = car_data_pre_process()
    pima_data, pima_test = pima_data_pre_process()
    mpg_data, mpg_test = mpg_data_pre_process()
    
    n_neighbors = 5
    test_nearest_neighbors(car_data, car_test, n_neighbors, 'Car:')
    test_nearest_neighbors(pima_data, pima_test, n_neighbors, 'Pima:')
    perform_and_score_classifier('MPG:', KNeighborsRegressor(n_neighbors=n_neighbors,
                                                             weights='distance',
                                                             n_jobs=-1, ), mpg_data, mpg_test)


def test_nearest_neighbors(data, test, n_neighbors, test_set_name):
    # perform_and_score_classifier(test_set_name, KNNClassifier(k=n_neighbors), data, data)
    perform_and_score_classifier(test_set_name, KNeighborsClassifier(n_neighbors=n_neighbors,
                                                                     weights='distance',
                                                                     n_jobs=-1, ), data, test)


def perform_and_score_classifier(name, classifier, data, test):
    print(name)
    scores = cross_val_score(classifier, data, test, cv=10)
    output_accuracy(scores)


def output_accuracy(scores):
    print('Accuracy: {0:.2f}% (+/- {1:.2f})'.format(scores.mean() * 100,
                                                    scores.std() * 2))


def car_data_pre_process():
    uci_car_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car"
                               ".data", header=None)
    
    # Set column names on data frame
    columns = "buying maint doors persons lug_boot safety target".split()
    uci_car_data.columns = columns
    
    # make data numerical
    column_numeration = {"buying": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                         "maint": {"vhigh": 4, "high": 3, "med": 2, "low": 1},
                         "doors": {"2": 1, "3": 2, "4": 3, "5more": 4},
                         "persons": {"2": 1, "4": 2, "more": 3},
                         "lug_boot": {"small": 1, "med": 2, "big": 3},
                         "safety": {"low": 1, "med": 2, "high": 3},
                         "target": {"unacc": 1, "acc": 2, "good": 3, "vgood": 4}}
    uci_car_data.replace(column_numeration, inplace=True)
    
    # split data and targets into separate data frames
    uci_car_targets = uci_car_data.iloc[:, 6:]
    uci_car_data = uci_car_data.iloc[:, :6]
    
    # turn the data and target data frames into lists
    uci_car_data_array = uci_car_data.as_matrix()
    uci_car_targets_array = uci_car_targets.as_matrix()
    uci_car_targets_array = uci_car_targets_array.flatten()
    
    return uci_car_data_array, uci_car_targets_array


def pima_data_pre_process():
    # get data from the UCI database
    pima_data = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima"
            "-indians-diabetes.data")
    
    # give columns names
    columns = "n_preg plasma bp tricep_fold insulin bmi pedigree age target".split()
    pima_data.columns = columns
    
    # mark zero values as NaN
    pima_data[["n_preg", "plasma", "bp", "tricep_fold", "insulin"]] = pima_data[
        ["n_preg", "plasma", "bp", "tricep_fold", "insulin"]].replace(0, np.NaN)
    
    # drop rows with NaN
    pima_data.dropna(inplace=True)
    
    # Split the dataframe into data and targets
    pima_data_targets = pima_data.iloc[:, 8:]
    pima_data = pima_data.iloc[:, :8]
    
    pima_data_array = pima_data.as_matrix()
    pima_data_targets_array = pima_data_targets.as_matrix()
    pima_data_targets_array = pima_data_targets_array.flatten()
    
    return pima_data_array, pima_data_targets_array


def mpg_data_pre_process():
    # get space delimited data set from url
    mpg_data = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
            delim_whitespace=True, na_values='?')
    columns = "mpg cyl disp hp weight accel year origin model".split()
    mpg_data.columns = columns
    
    # replace and remove missing values and associated rows
    mpg_data.dropna(inplace=True)
    mpg_data.drop(columns='model', inplace=True)
    
    # split dataframe into data (minus the model) and targets
    mpg_targets = mpg_data.iloc[:, :1]
    mpg_data = mpg_data.iloc[:, 1:8]
    
    mpg_data_array = mpg_data.as_matrix()
    mpg_targets_array = mpg_targets.as_matrix()
    mpg_targets_array = mpg_targets_array.flatten()
    
    return mpg_data_array, mpg_targets_array


if __name__ == '__main__':
    main()

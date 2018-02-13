from random import randrange

import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.tree.tree import DecisionTreeClassifier


def calc_entropy(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    
    entropy = 0.0
    for group in groups:
        size = float(len(group))

        if size == 0:
            continue
        score = 0.0

        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p

        entropy += (1.0 - score) * (size / n_instances)
    return entropy


def test_split(index, value, data_set):
    left, right = list(), list()
    for row in data_set:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def do_split(data_set):
    class_values = list(set(row[-1] for row in data_set))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(data_set[0]) - 1):
        for row in data_set:
            groups = test_split(index, row[index], data_set)
            entropy = calc_entropy(groups, class_values)
            if entropy < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], entropy, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def create_leaf(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    
    if not left or not right:
        node['left'] = node['right'] = create_leaf(left + right)
        return
    
    if depth >= max_depth:
        node['left'], node['right'] = create_leaf(left), create_leaf(right)
        return
    
    if len(left) <= min_size:
        node['left'] = create_leaf(left)
    else:
        node['left'] = do_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    
    if len(right) <= min_size:
        node['right'] = create_leaf(right)
    else:
        node['right'] = do_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = do_split(train)
    split(root, max_depth, min_size, 1)
    print(root)
    return root


def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % (depth * ' ', (node['index'] + 1), node['value']))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % (depth * ' ', node))


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def decision_tree(train, test, max_depth=5, min_size=10):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions


def cross_validation_split(data_set, n_folds):
    data_set_split = list()
    data_set_copy = list(data_set)
    fold_size = int(len(data_set) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(data_set_copy))
            fold.append(data_set_copy.pop(index))
        data_set_split.append(fold)
    return data_set_split


def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def k_cross_validate(data_set, algorithm, n_folds, *args):
    folds = cross_validation_split(data_set, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        score = accuracy(actual, predicted)
        scores.append(score)
    return scores


def main():
    iris_data, iris_target = iris_data_set_processing()
    lens_data, lens_target = lenses_data_set_processing()
    vote_data, vote_targets = voting_data_set_processing()
    credit_data, credit_target = credit_data_set_processing()
    chess_data, chess_target = chess_data_set_processing()
    
    compare_sklearn_dt(chess_data, chess_target, credit_data, credit_target, iris_data, iris_target,
                       lens_data, lens_target, vote_data, vote_targets)
    
    compare_my_dt(chess_data=chess_data_set_processing(False),
                  credit_data=credit_data_set_processing(False),
                  iris_data=iris_data_set_processing(False),
                  lens_data=lenses_data_set_processing(False),
                  vote_data=voting_data_set_processing(False))


def compare_my_dt(chess_data, credit_data, iris_data, lens_data, vote_data):
    iris_scores = k_cross_validate(iris_data, decision_tree, n_folds=10)
    print('(MY-IRIS) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(iris_scores.mean() * 100,
                                                              iris_scores.std() * 2))
    votes_scores = k_cross_validate(lens_data, decision_tree, n_folds=10)
    print('(MY-LENS) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(votes_scores.mean() * 100,
                                                              votes_scores.std() * 2))
    votes_scores = k_cross_validate(vote_data, decision_tree, n_folds=10)
    print('(MY-VOTES) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(votes_scores.mean() * 100,
                                                               votes_scores.std() * 2))
    credit_scores = k_cross_validate(credit_data, decision_tree, n_folds=10)
    print('(MY-CREDIT) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(credit_scores.mean() * 100,
                                                                credit_scores.std() * 2))
    chess_scores = k_cross_validate(chess_data, decision_tree, n_folds=10)
    print('(MY-CHESS) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(chess_scores.mean() * 100,
                                                               chess_scores.std() * 2))


def compare_sklearn_dt(chess_data, chess_target, credit_data, credit_target, iris_data, iris_target,
                       lens_data, lens_target, vote_data, vote_targets):
    sk_dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
    
    iris_scores = cross_val_score(sk_dt, iris_data, iris_target, cv=10)
    print('(SK-IRIS) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(iris_scores.mean() * 100,
                                                              iris_scores.std() * 2))
    votes_scores = cross_val_score(sk_dt, lens_data, lens_target, cv=10)
    print('(SK-LENS) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(votes_scores.mean() * 100,
                                                              votes_scores.std() * 2))
    votes_scores = cross_val_score(sk_dt, vote_data, vote_targets, cv=10)
    print('(SK-VOTES) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(votes_scores.mean() * 100,
                                                               votes_scores.std() * 2))
    credit_scores = cross_val_score(sk_dt, credit_data, credit_target, cv=10)
    print('(SK-CREDIT) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(credit_scores.mean() * 100,
                                                                credit_scores.std() * 2))
    chess_scores = cross_val_score(sk_dt, chess_data, chess_target, cv=10)
    print('(SK-CHESS) Accuracy: {0:.2f}% (+/- {1:.2f})'.format(chess_scores.mean() * 100,
                                                               chess_scores.std() * 2))


def iris_data_set_processing(sk_learn=True):
    iris_data_set = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris'
                                '/iris.data', header=None)
    columns = 'sepal_l sepal_w petal_l petal_w class'.split()
    iris_data_set.columns = columns
    
    column_numeration = {'class': {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}}
    iris_data_set.replace(column_numeration, inplace=True)
    
    if sk_learn:
        iris_targets = iris_data_set.iloc[:, 4:5].as_matrix().flatten()
        iris_data = iris_data_set.iloc[:, 0:4].as_matrix()
        
        return iris_data, iris_targets
    else:
        return iris_data_set.as_matrix()


def lenses_data_set_processing(sk_learn=True):
    lens_data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data',
            header=None, delim_whitespace=True)
    columns = 'record age script astigmatic tears class'.split()
    lens_data_set.columns = columns
    
    lens_data_set.drop(columns='record', inplace=True)
    
    if sk_learn:
        lens_targets = lens_data_set.iloc[:, 4:5].as_matrix().flatten()
        lens_data = lens_data_set.iloc[:, 0:4].as_matrix()
        
        return lens_data, lens_targets
    else:
        return lens_data_set.as_matrix()


def voting_data_set_processing(sk_learn=True):
    vote_data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes'
            '-84.data', header=None)
    columns = 'class-name handicapped-infants water-project-cost-sharing ' \
              'adoption-of-the-budget-resolution physician-fee-freeze el-salvador-aid ' \
              'religious-groups-in-schools anti-satellite-test-ban aid-to-nicaraguan-contras ' \
              'mx-missile immigration synfuels-corporation-cutback education-spending ' \
              'superfund-right-to-sue crime duty-free-exports ' \
              'export-administration-act-south-africa'.split()
    vote_data_set.columns = columns
    
    reorder_columns = 'handicapped-infants water-project-cost-sharing ' \
                      'adoption-of-the-budget-resolution physician-fee-freeze el-salvador-aid ' \
                      'religious-groups-in-schools anti-satellite-test-ban ' \
                      'aid-to-nicaraguan-contras ' \
                      'mx-missile immigration synfuels-corporation-cutback education-spending ' \
                      'superfund-right-to-sue crime duty-free-exports ' \
                      'export-administration-act-south-africa class-name'.split()
    vote_data_set = vote_data_set[reorder_columns]
    
    vote_data_set = vote_data_set.replace("?", np.NaN)
    vote_data_set.dropna(inplace=True)
    
    column_numeration = {'class-name': {'democrat': 1, 'republican': 0},
                         'handicapped-infants': {'y': 1, 'n': 0},
                         'water-project-cost-sharing': {'y': 1, 'n': 0},
                         'adoption-of-the-budget-resolution': {'y': 1, 'n': 0},
                         'physician-fee-freeze': {'y': 1, 'n': 0},
                         'el-salvador-aid': {'y': 1, 'n': 0},
                         'religious-groups-in-schools': {'y': 1, 'n': 0},
                         'anti-satellite-test-ban': {'y': 1, 'n': 0},
                         'aid-to-nicaraguan-contras': {'y': 1, 'n': 0},
                         'mx-missile': {'y': 1, 'n': 0},
                         'immigration': {'y': 1, 'n': 0},
                         'synfuels-corporation-cutback': {'y': 1, 'n': 0},
                         'education-spending': {'y': 1, 'n': 0},
                         'superfund-right-to-sue': {'y': 1, 'n': 0},
                         'crime': {'y': 1, 'n': 0},
                         'duty-free-exports': {'y': 1, 'n': 0},
                         'export-administration-act-south-africa': {'y': 1, 'n': 0}}
    vote_data_set.replace(column_numeration, inplace=True)
    
    if sk_learn:
        vote_targets = vote_data_set.iloc[:, 16:17].as_matrix().flatten()
        vote_data = vote_data_set.iloc[:, 0:16].as_matrix()
        
        return vote_data, vote_targets
    else:
        return vote_data_set.iloc[:, 0:18].as_matrix()


def credit_data_set_processing(sk_learn=True):
    credit_data_set = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'
                                  '/credit-screening/crx.data', header=None)
    columns = 'a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16'.split()
    credit_data_set.columns = columns
    
    reorder_columns = 'a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a1'.split()
    credit_data_set = credit_data_set[reorder_columns]
    
    credit_data_set = credit_data_set.replace("?", np.NaN)
    credit_data_set.dropna(inplace=True)
    
    column_numeration = {'a1': {'b': 1, 'a': 0},
                         'a4': {'u': 0, 'y': 1, 'l': 2, 't': 3},
                         'a5': {'g': 0, 'p': 1, 'gg': 2},
                         'a6': {'c': 0, 'd': 1, 'cc': 2, 'i': 3, 'j': 4, 'k': 5, 'm': 6, 'r': 7,
                                'q': 8, 'w': 9, 'x': 10, 'e': 11, 'aa': 12, 'ff': 13},
                         'a7': {'v': 0, 'h': 1, 'bb': 2, 'j': 3, 'n': 4, 'z': 5, 'dd': 6, 'ff': 7,
                                'o': 8},
                         'a9': {'t': 1, 'f': 0},
                         'a10': {'t': 1, 'f': 0},
                         'a12': {'t': 1, 'f': 0},
                         'a13': {'g': 0, 'p': 1, 's': 2},
                         'a16': {'+': 1, '-': 0}}
    credit_data_set.replace(column_numeration, inplace=True)
    
    if sk_learn:
        credit_targets = credit_data_set.iloc[:, 15:16].as_matrix().flatten()
        credit_data = credit_data_set.iloc[:, 0:15].as_matrix()
        
        return credit_data, credit_targets
    else:
        return credit_data_set.iloc[:, 0:17].as_matrix()


def chess_data_set_processing(sk_learn=True):
    chess_data_set = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/chess'
                                 '/king-rook-vs-king/krkopt.data', header=None)
    columns = 'wkf wkr wrf wrr bkf bkr optimal-depth'.split()
    chess_data_set.columns = columns
    
    column_enumeration = {'wkf': {'a': 0, 'b': 1, 'c': 2, 'd': 3},
                          'wrf': {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7},
                          'bkf': {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7},
                          'bkr': {},
                          'optimal-depth': {'draw': -1, 'zero': 0, 'one': 1, 'two': 2, 'three': 3,
                                            'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                                            'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
                                            'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
                                            'sixteen': 16}}
    chess_data_set.replace(column_enumeration, inplace=True)
    
    if sk_learn:
        chess_targets = chess_data_set.iloc[:, 6:7].as_matrix().flatten()
        chess_data = chess_data_set.iloc[:, 0:6].as_matrix()
        
        return chess_data, chess_targets
    else:
        return chess_data_set.iloc[:, 0:7].as_matrix()


if __name__ == '__main__':
    main()

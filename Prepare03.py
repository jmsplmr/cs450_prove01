import numpy as np
import pandas as pd

headers = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-num',
           "Marital status", "Occupation", "Relationship", "race",
           "Sex", "Capital-gain", "Capital-loss", "hours-per-week", "native-country", "Class"]

census_data_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult'
                             '.data', header = None, names = headers, na_values = '?')
census_data_df.dropna(axis = 'index', how = 'any', inplace = True)

print(census_data_df["Sex"].value_counts())

census_data_df["Sex"] = np.where(census_data_df["Sex"].str.contains("female"), 1, 0)
print(census_data_df["Sex"].value_counts())

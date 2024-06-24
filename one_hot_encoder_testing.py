#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 02:32:10 2024

@author: sergiybegun
"""

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector


OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

record = { 

 'Name' : ['Ankit', 'Swapnil', 'Aishwarya', 
          'Priyanka', 'Shivangi', 'Shaurya' ],

 'Age' : [22, 20, None, 19, 18, 22], 

 'Stream' : ['Math', 'Commerce', 'Science', 
            'Math', 'Math', 'Science'], 

 'Percentage' : [90, 90, 96, 75, None, 80] }

dataframe = pd.DataFrame(record, columns = ['Name', 'Age', 'Stream', 'Percentage'])

print(dataframe)
print(dataframe.index)

numerical_transformer = SimpleImputer(strategy="mean")
categorical_transformer = Pipeline(steps=[
    ("onehot", OH_encoder)
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, make_column_selector(dtype_include=np.number)),
        ("cat", categorical_transformer, make_column_selector(dtype_include=object))
    ] ).set_output(transform="pandas")


results = preprocessor.fit_transform(dataframe)

print(results)
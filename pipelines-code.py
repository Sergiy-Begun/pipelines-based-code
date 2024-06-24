#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:31:16 2024

@author: sergiybegun
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import make_column_selector
import copy
import numpy as np
from datetime import datetime

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


"""
see other possible useful variants at https://github.com/scikit-learn/scikit-learn/blob/2621573e60c295a435c62137c65ae787bf438e61/sklearn/base.py

and there https://github.com/scikit-learn/scikit-learn/blob/2621573e6/sklearn/preprocessing/_encoders.py#L458

>>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> class MyEstimator(BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=X.shape[0], fill_value=self.param)
    >>> estimator = MyEstimator(param=2)
    >>> estimator.get_params()
    {'param': 2}
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([2, 2, 2])
    >>> estimator.set_params(param=3).fit(X, y).predict(X)
    array([3, 3, 3])

>>> import numpy as np
    >>> from sklearn.base import BaseEstimator, TransformerMixin
    >>> class MyTransformer(TransformerMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         return self
    ...     def transform(self, X):
    ...         return np.full(shape=len(X), fill_value=self.param)
    >>> transformer = MyTransformer()
    >>> X = [[1, 2], [2, 3], [3, 4]]
    >>> transformer.fit_transform(X)
    array([1, 1, 1])

>>> import numpy as np
    >>> from sklearn.base import BaseEstimator, RegressorMixin
    >>> # Mixin classes should always be on the left-hand side for a correct MRO
    >>> class MyEstimator(RegressorMixin, BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=X.shape[0], fill_value=self.param)
    >>> estimator = MyEstimator(param=0)
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([-1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([0, 0, 0])
    >>> estimator.score(X, y)
    0.0

"""

# definition of custom functions and classes

class feature_extraction_tool(BaseEstimator, TransformerMixin):

    def __init__(self, *, features_list = [], columns_to_remove_after = []):
        self.features_list = features_list
        self.columns_to_remove_after = columns_to_remove_after
        # the current year for the age parameter determination
        self.curr_year = float(datetime.now().year)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        
        return self

    def transform(self, X):
        # inserting the new columns for feature extraction from data
        for feature_item in self.features_list:
            X.insert(X.shape[1], str(feature_item), "0")
        
        cheap_margin = 20000.0
        average_class_margin = 70000.0
        luxury_margin = 750000.0
        X.price = X.price.astype('float64')
        X.milage = X.milage.astype('float64')
        
        # filling of the initial data for keys based on actual data and formation of the first version of keys in the dictionary variable
    
        for current_row in range(0,X.shape[0]):
            
            # price categories
            if (((X.at[current_row, "price"]) > 0.0) and ((X.at[current_row, "price"]) < cheap_margin)):
                X.at[current_row, "Cheap_Cars"] = 1
            
            if (((X.at[current_row, "price"]) >= cheap_margin) and ((X.at[current_row, "price"]) < average_class_margin)):
                X.at[current_row, "Average_Class_Cars"] = 1
            
            if (((X.at[current_row, "price"]) >= average_class_margin) and ((X.at[current_row, "price"]) < luxury_margin)):
                X.at[current_row, "Luxury_Cars"] = 1
            
            if ((X.at[current_row, "price"]) >= luxury_margin):
                X.at[current_row, "Super_Cars"] = 1
            
            # Age
            
            if ((str(X.at[current_row, "model_year"]).isdecimal()) and (float(X.at[current_row, "model_year"]) >  0.0)):
                X.at[current_row, "Age"] = self.curr_year - float(X.at[current_row, "model_year"])
            else:
                X.at[current_row, "Age"] =  np.nan # temporary recognizable value
                            
            # Brand
            
            if ((len(str(X.at[current_row, "brand"])) > 0) and (str(X.at[current_row, "brand"]) != "–") and (pd.isna(X.at[current_row, "brand"]) == False)):
                cur_brand_data = str(X.at[current_row, "brand"]).replace(" ","_").replace("/","_").replace(".","_").replace("-","_").replace(",","_")
                X.at[current_row, "Brand"] = cur_brand_data
            else:
                X.at[current_row, "Brand"] = np.nan

            # Fuel_Type
            
            if ((len(str(X.at[current_row, "fuel_type"])) > 0) and (str(X.at[current_row, "fuel_type"]) != "–") and (str(X.at[current_row, "fuel_type"]) != "not supported") and (pd.isna(X.at[current_row, "fuel_type"]) == False)):
                cur_fuel_type = str(X.at[current_row, "fuel_type"]).replace(" ","_").replace("/","_").replace(".","_").replace("-","_").replace(",","_")
                X.at[current_row, "Fuel_Type"] = cur_fuel_type
                
            elif ((pd.isna(X.at[current_row, "fuel_type"]) == True) and (len(str(X.at[current_row, "engine"])) >  0) and ((str(X.at[current_row, "engine"])).find("Electric") >= 0)):
                cur_fuel_type = "Electricity"
                X.at[current_row, "Fuel_Type"] = cur_fuel_type
                
            elif ((len(str(X.at[current_row, "fuel_type"])) > 0) and (str(X.at[current_row, "fuel_type"]) == "not supported") and (len(str(X.at[current_row, "engine"])) >  0) and ((str(X.at[current_row, "engine"])).find("Hydrogen") >= 0)):
                cur_fuel_type = "Hydrogen"
                X.at[current_row, "Fuel_Type"] = cur_fuel_type
                
            else:
                X.at[current_row, "Fuel_Type"] = np.nan
            
            # Model_Main & Model_Extra
            
            if ((len(str(X.at[current_row, "model"])) > 0) and (str(X.at[current_row, "model"]) != "–") and (pd.isna(X.at[current_row, "model"]) == False)):
                cur_model_main = str(X.at[current_row, "model"]).split()[0].replace(" ","_").replace("/","_").replace(".","_").replace("-","_").replace(",","_")
                cur_model_extra = str(X.at[current_row, "model"]).replace(" ","_").replace("/","_").replace(".","_").replace("-","_").replace(",","_")
                X.at[current_row, "Model_Main"] = cur_model_main
                X.at[current_row, "Model_Extra"] = cur_model_extra
            else:
                X.at[current_row, "Model_Main"] = np.nan
                X.at[current_row, "Model_Extra"] = np.nan
            
            # Speed_Num & Transm_Type
            
            if ((len(str(X.at[current_row, "transmission"])) > 0) and (str(X.at[current_row, "transmission"]) != "–") and (pd.isna(X.at[current_row, "transmission"]) == False)):
                if (str(X.at[current_row, "transmission"]).split()[0].find("Speed") >= 0):
                    cur_speed_num = str(X.at[current_row, "transmission"]).split()[0].replace(" ","").replace("/","").replace(".","").replace("-","").replace(",","").replace("Speed","").replace("Single","1")
                    X.at[current_row, "Speed_Num"] = cur_speed_num
                else:
                    X.at[current_row, "Speed_Num"] = np.nan
                
                cur_transm_type = str(X.at[current_row, "transmission"]).replace("A/T","Automatic").replace("M/T","Mechanical").replace(" ","_").replace("/","_").replace(".","_").replace("-","_").replace(",","_")
                X.at[current_row, "Transm_Type"] = cur_transm_type
            else:
                X.at[current_row, "Speed_Num"] = np.nan
                X.at[current_row, "Transm_Type"] = np.nan
            
            # Engine_Type & Engine_Cyl & Engine_L & Engine_HP
            
            if ((len(str(X.at[current_row, "engine"])) > 0) and (str(X.at[current_row, "engine"]) != "–") and (pd.isna(X.at[current_row, "engine"]) == False)):
                if (str(X.at[current_row, "engine"]).split()[0].find("HP") >= 0):
                    X.at[current_row, "Engine_HP"] = float(str(X.at[current_row, "engine"]).split()[0].replace("HP",""))
                else:
                    X.at[current_row, "Engine_HP"] = np.nan # temporary recognizable value
                
                Engine_L_is_set = False
                Engine_Cyl_is_set = False
                cur_Engine_Type_content = ""
                cur_engine_characteristics_words = str(X.at[current_row, "engine"]).split()
                cur_eng_number_of_words_in_char = len(cur_engine_characteristics_words)
                for cur_range_num in range(0,cur_eng_number_of_words_in_char):
                    engine_char = str(cur_engine_characteristics_words[cur_range_num])
                    if ((engine_char.find("L") > 0) and (Engine_L_is_set == False)):
                        X.at[current_row, "Engine_L"] = float(str(engine_char).replace("L",""))
                        Engine_L_is_set = True
                    if ((engine_char.find("L") == 0) and (Engine_L_is_set == False) and (cur_range_num > 0)):
                        X.at[current_row, "Engine_L"] = float(str(str(cur_engine_characteristics_words[cur_range_num - 1])))
                        Engine_L_is_set = True
                    if ((engine_char.find("Liter") >= 0) and (Engine_L_is_set == False) and (cur_range_num > 0)):
                        X.at[current_row, "Engine_L"] = float(str(cur_engine_characteristics_words[cur_range_num - 1]))
                        Engine_L_is_set = True
                    if ((cur_range_num == (cur_eng_number_of_words_in_char - 1)) and (Engine_L_is_set == False)):
                        X.at[current_row, "Engine_L"] = np.nan # temporary recognizable value
                        Engine_L_is_set = True
                    
                    if ((engine_char.find("I") == 0) and (Engine_Cyl_is_set == False) and (engine_char.find("VTEC") < 0) and (engine_char.find("ntercooled") < 0) and (engine_char.find("II") < 0)):
                        cur_engine_cyl = str(engine_char).replace("I","").replace(" ","").replace("/","_").replace(".","_").replace("-","").replace(",","_")
                        X.at[current_row, "Engine_Cyl"] = cur_engine_cyl
                        Engine_Cyl_is_set = True
                    if ((engine_char.find("V") == 0) and (Engine_Cyl_is_set == False) and ((engine_char.find("arioCam") < 0))):
                        cur_engine_cyl = str(engine_char).replace("V","").replace(" ","").replace("/","_").replace(".","_").replace("-","").replace(",","_")
                        X.at[current_row, "Engine_Cyl"] = cur_engine_cyl
                        Engine_Cyl_is_set = True
                    if ((engine_char.find("Cylinder") >= 0) and (Engine_Cyl_is_set == False) and (cur_range_num > 0)):
                        cur_engine_cyl = str(cur_engine_characteristics_words[cur_range_num - 1]).replace(" ","").replace("/","_").replace(".","_").replace("-","_").replace(",","_")
                        X.at[current_row, "Engine_Cyl"] = cur_engine_cyl
                        Engine_Cyl_is_set = True
                    if ((cur_range_num == (cur_eng_number_of_words_in_char - 1)) and (Engine_Cyl_is_set == False)):
                        X.at[current_row, "Engine_Cyl"] = np.nan
                        Engine_Cyl_is_set = True
                    
                    if ((engine_char.find("I") != 0) and (engine_char.find("V") != 0) and (engine_char.find("Cylinder") < 0) and (engine_char.find("L") < 0) and (engine_char.find("Liter") < 0) and (engine_char.find("HP") < 0) and (engine_char.isdecimal() == False)):
                        cur_Engine_Type_content += engine_char + "_"
                
                cur_engine_type = str(cur_Engine_Type_content).replace(" ","_").replace("/","_").replace(".","_").replace("-","_").replace(",","_")
                X.at[current_row, "Engine_Type"] = cur_engine_type
            else:
                X.at[current_row, "Engine_HP"] = np.nan # temporary recognizable value
                X.at[current_row, "Engine_L"] = np.nan # temporary recognizable value
                X.at[current_row, "Engine_Cyl"] = np.nan
                X.at[current_row, "Engine_Type"] = np.nan

        # removing unnecessary columns after feature extraction is finished to avoid data duplication on categorical columns digitalization stage
        X = X.drop(self.columns_to_remove_after, axis=1)
        return pd.DataFrame(X)

class existing_numerical_data_cleaning_tool(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.is_fitted_ = True

        return self

    def transform(self, X):

        for current_row in range(0,X.shape[0]):
                            
            # milage
            
            if ((str(X.at[current_row, "milage"]).isdecimal()) and (float(X.at[current_row, "milage"]) >=  0.0)):
                X.at[current_row, "milage"] = float(X.at[current_row, "milage"])
            elif ((str(X.at[current_row, "milage"]).find("mi.") > 0) and (len(str(X.at[current_row, "milage"])) >  0)):
                X.at[current_row, "milage"] = float((str(X.at[current_row, "milage"]).replace(",","")).split()[0])
            else:
                X.at[current_row, "milage"] = np.nan
            
            # price
            
            if ((str(X.at[current_row, "price"]).isdecimal()) and (float(X.at[current_row, "price"]) >=  0.0)):
                X.at[current_row, "price"] = float(X.at[current_row, "price"])
            elif ((str(X.at[current_row, "price"]).find("$") >= 0) and (len(str(X.at[current_row, "price"])) >  0)):
                X.at[current_row, "price"] = float((str(X.at[current_row, "price"]).replace(",","").replace("$","")))
            else:
                X.at[current_row, "price"] = np.nan
        
        return pd.DataFrame(X)

def test_training_equivalentaizer(X, main_data_frame):
    # this function should be called before models training
    # X would be the DataFrame with test values, main_data_frame is the DataFrame with training data
    # the result would be the modified DataFrames with equivalent set of columns (the missing columns in some of DataFrames are filled with zeros)
    
    X_to_return = copy.deepcopy(X)
    
    main_to_return = copy.deepcopy(main_data_frame)

    for col_name in X.columns:
        if col_name not in main_data_frame.columns:
            main_to_return.insert(main_to_return.shape[1], col_name, "0")
            
    for col_name in main_data_frame.columns:
        if col_name not in X.columns:
            X_to_return.insert(X.shape[1], col_name, "0")
    
    return [pd.DataFrame(X_to_return), pd.DataFrame(main_to_return)]

class setting_data_types(BaseEstimator, TransformerMixin):
    # this function should be called before models training
    # X would be the DataFrame with test values, main_data_frame is the DataFrame with training data
    # the result would be the modified DataFrames with equivalent set of columns (the missing columns in some of DataFrames are filled with zeros)

    def __init__(self, *, categorical_columns=[], numerical_columns=[]):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        for col_name in X.columns:
            if col_name in self.categorical_columns:
                X[[col_name,]] = X[[col_name,]].astype('category')
            if col_name in self.numerical_columns:
                X[[col_name,]] = X[[col_name,]].astype('float64')
        return pd.DataFrame(X)


main_file_path = ".../study-in-kaggle/KaggleX-Skill-Assessment-Challenge/"

test_file_path = main_file_path + "test.csv"

train_file_path = main_file_path + "train.csv"

extra_data_file_path = main_file_path + "used_cars.csv"

extra_data_modified_file_path = main_file_path + "extra_data_modified.csv"

train_modified_file_path = main_file_path + "train_modified.csv"

test_modified_file_path = main_file_path + "test_modified.csv"

# main data will be the concatination of the train data and extra data to enhance the precision

main_data_file_path = main_file_path + "main_data.csv"

# X_full data file path

X_full_file_path = main_file_path + "X_full.csv"

# output file for predicted prices for test data

output_data_file_path_1 = main_file_path + "output_begun_sergiy_XGBoost_23_06_2024_WS.csv"

output_data_file_path_2 = main_file_path + "output_begun_sergiy_XGBoost_full_range_23_06_2024_WS.csv"


# read the data from files

train_df = pd.read_csv(train_file_path)

test_df = pd.read_csv(test_file_path)

extra_data_df = pd.read_csv(extra_data_file_path)

# we use temp data frames for data frames modification purposes before the concatination

temp_data_frame_extra = copy.deepcopy(extra_data_df.dropna(axis=0, how="all"))

temp_data_frame_train = copy.deepcopy(
    train_df.drop(["id"], axis=1).dropna(axis=0, how="all"))

temp_data_frame_extra.to_csv(extra_data_modified_file_path)

temp_data_frame_train.to_csv(train_modified_file_path)

# we need to modify test data for prediction too

test_modified_df = copy.deepcopy(test_df.dropna(axis=0, how="all"))

# formation of the first version of combined data frame

main_data_df = pd.concat(
    [temp_data_frame_train, temp_data_frame_extra], ignore_index=True)

contaminated_numerical_cols = ["price", "milage"]

# filling gaps in dataset
numerical_transformer = SimpleImputer(strategy="most_frequent").set_output(transform="pandas") # "most_frequent", "median," or "mean"

# categorical data processing
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(transform="pandas")

# Preprocessing for contaminated numerical data
data_transformer = Pipeline(steps=[
    ("existing_numerical_data_cleaning", existing_numerical_data_cleaning_tool()),
    ("feature_extraction", feature_extraction_tool()),
    ("imputer_mixed", numerical_transformer), # new numerical columns are appeared as a result of feature_extraction_tool, but the option "most_frequent" solves this problem
    ("setting_datatypes",setting_data_types()),
])

# Bundle preprocessing for numerical and categorical data
categorical_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, make_column_selector(dtype_include=np.number)),
        ("cat", categorical_transformer, make_column_selector(dtype_include="category"))
    ] ).set_output(transform="pandas")

features_dictionary = ["Engine_HP", "Engine_L", "Age", "Engine_Cyl", "Engine_Type", "Speed_Num", "Transm_Type", "Model_Main", "Model_Extra", "Fuel_Type", "Brand", "Cheap_Cars", "Average_Class_Cars", "Luxury_Cars", "Super_Cars"]

list_of_categorical_columns_to_remove_after_feature_extraction_finished = ["brand", "model", "model_year", "fuel_type", "engine", "transmission", "ext_col", "int_col"]

list_of_final_categorical_columns_before_one_hot_imputer = ["Engine_Type", "Transm_Type", "Model_Main", "Model_Extra", "Fuel_Type", "Brand", "accident", "clean_title"]

list_of_final_numerical_columns_before_one_hot_imputer = ["price", "milage", "Engine_HP", "Engine_L", "Age", "Engine_Cyl", "Speed_Num", "Cheap_Cars", "Average_Class_Cars", "Luxury_Cars", "Super_Cars"]

test_id_to_redefine_after_one_hot_encoder = test_modified_df.id

main_data_df = data_transformer.set_params(
    feature_extraction__features_list=features_dictionary,
    feature_extraction__columns_to_remove_after=list_of_categorical_columns_to_remove_after_feature_extraction_finished,
    setting_datatypes__categorical_columns=list_of_final_categorical_columns_before_one_hot_imputer,
    setting_datatypes__numerical_columns=list_of_final_numerical_columns_before_one_hot_imputer
    ).fit_transform(main_data_df)



main_data_df = categorical_preprocessor.fit_transform(main_data_df)

test_modified_df.insert(test_modified_df.shape[1], "price", "0")

test_modified_df = data_transformer.set_params(
    feature_extraction__features_list=features_dictionary,
    feature_extraction__columns_to_remove_after=list_of_categorical_columns_to_remove_after_feature_extraction_finished,
    setting_datatypes__categorical_columns=list_of_final_categorical_columns_before_one_hot_imputer,
    setting_datatypes__numerical_columns=list_of_final_numerical_columns_before_one_hot_imputer
    ).fit_transform(test_modified_df)



test_modified_df = categorical_preprocessor.fit_transform(test_modified_df)

eq_datasets = test_training_equivalentaizer(X=test_modified_df,main_data_frame=main_data_df)

test_modified_df = copy.deepcopy(eq_datasets[0])

test_modified_df.insert(test_modified_df.shape[1], "id", test_id_to_redefine_after_one_hot_encoder)

main_data_df = copy.deepcopy(eq_datasets[1])

y_full = copy.deepcopy(main_data_df.num__price)
X_full = copy.deepcopy(main_data_df.drop(["num__price"], axis=1)).sort_index(axis=1)


y_cheap_full = copy.deepcopy(main_data_df[(main_data_df.num__Cheap_Cars == 1.0)].num__price)
X_cheap_full = copy.deepcopy(main_data_df[(main_data_df.num__Cheap_Cars == 1.0)].drop(["num__price"], axis=1)).sort_index(axis=1)


y_average_class_full = copy.deepcopy(main_data_df[(main_data_df.num__Average_Class_Cars == 1.0)].num__price)
X_average_class_full = copy.deepcopy(main_data_df[(main_data_df.num__Average_Class_Cars == 1.0)].drop(["num__price"], axis=1)).sort_index(axis=1)


y_luxury_full = copy.deepcopy(main_data_df[(main_data_df.num__Luxury_Cars == 1.0)].num__price)
X_luxury_full = copy.deepcopy(main_data_df[(main_data_df.num__Luxury_Cars == 1.0)].drop(["num__price"], axis=1)).sort_index(axis=1)


y_super_car_full = copy.deepcopy(main_data_df[(main_data_df.num__Super_Cars == 1.0)].num__price)
X_super_car_full = copy.deepcopy(main_data_df[(main_data_df.num__Super_Cars == 1.0)].drop(["num__price"], axis=1)).sort_index(axis=1)


y_Cheap_Cars_classifier_full = copy.deepcopy(main_data_df.num__Cheap_Cars)
X_Cheap_Cars_classifier_full = copy.deepcopy(main_data_df.drop(["num__price", "num__Cheap_Cars", "num__Average_Class_Cars", "num__Luxury_Cars", "num__Super_Cars"], axis=1)).sort_index(axis=1)

y_Average_Class_Cars_classifier_full = copy.deepcopy(main_data_df.num__Average_Class_Cars)
X_Average_Class_Cars_classifier_full = copy.deepcopy(main_data_df.drop(["num__price", "num__Cheap_Cars", "num__Average_Class_Cars", "num__Luxury_Cars", "num__Super_Cars"], axis=1)).sort_index(axis=1)

y_Luxury_Cars_classifier_full = copy.deepcopy(main_data_df.num__Luxury_Cars)
X_Luxury_Cars_Cars_classifier_full = copy.deepcopy(main_data_df.drop(["num__price", "num__Cheap_Cars", "num__Average_Class_Cars", "num__Luxury_Cars", "num__Super_Cars"], axis=1)).sort_index(axis=1)

y_Super_Cars_classifier_full = copy.deepcopy(main_data_df.num__Super_Cars)
X_Super_Cars_Cars_classifier_full = copy.deepcopy(main_data_df.drop(["num__price", "num__Cheap_Cars", "num__Average_Class_Cars", "num__Luxury_Cars", "num__Super_Cars"], axis=1)).sort_index(axis=1)


y_Cheap_Cars_classifier_test = copy.deepcopy(test_modified_df.num__Cheap_Cars)
X_Cheap_Cars_classifier_test = copy.deepcopy(test_modified_df.drop(["id", "num__price", "num__Cheap_Cars", "num__Average_Class_Cars", "num__Luxury_Cars", "num__Super_Cars"], axis=1)).sort_index(axis=1)

y_Average_Class_Cars_classifier_test = copy.deepcopy(test_modified_df.num__Average_Class_Cars)
X_Average_Class_Cars_classifier_test = copy.deepcopy(test_modified_df.drop(["id","num__price", "num__Cheap_Cars", "num__Average_Class_Cars", "num__Luxury_Cars", "num__Super_Cars"], axis=1)).sort_index(axis=1)

y_Luxury_Cars_classifier_test = copy.deepcopy(test_modified_df.num__Luxury_Cars)
X_Luxury_Cars_Cars_classifier_test = copy.deepcopy(test_modified_df.drop(["id", "num__price", "num__Cheap_Cars", "num__Average_Class_Cars", "num__Luxury_Cars", "num__Super_Cars"], axis=1)).sort_index(axis=1)

y_Super_Cars_classifier_test = copy.deepcopy(test_modified_df.num__Super_Cars)
X_Super_Cars_Cars_classifier_test = copy.deepcopy(test_modified_df.drop(["id", "num__price", "num__Cheap_Cars", "num__Average_Class_Cars", "num__Luxury_Cars", "num__Super_Cars"], axis=1)).sort_index(axis=1)

# Break off validation set from training data
# X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, train_size=0.8, test_size=0.2, random_state=0)

# Define model

# Bundle preprocessing and modeling code in a pipeline
cheap_pipeline_based_data_fitting = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("model", XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8, enable_categorical=True))
    ])

# Bundle preprocessing and modeling code in a pipeline
average_class_pipeline_based_data_fitting = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("model", XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8, enable_categorical=True))
    ])

# Bundle preprocessing and modeling code in a pipeline
luxury_pipeline_based_data_fitting = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("model", XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8, enable_categorical=True))
    ])

# Bundle preprocessing and modeling code in a pipeline
super_car_pipeline_based_data_fitting = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("model", XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8, enable_categorical=True))
    ])

# Bundle preprocessing and modeling code in a pipeline
full_range_pipeline_based_data_fitting = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("model", XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8, enable_categorical=True))
    ])

print("cheap_full_fit_stage")
cheap_pipeline_based_data_fitting.fit(X_cheap_full,y_cheap_full)

print("average_class_full_fit_stage")
average_class_pipeline_based_data_fitting.fit(X_average_class_full,y_average_class_full)

print("luxury_full_fit_stage")
luxury_pipeline_based_data_fitting.fit(X_luxury_full,y_luxury_full)

print("super_car_full_fit_stage")
super_car_pipeline_based_data_fitting.fit(X_super_car_full,y_super_car_full)

print("full_range_full_fit_stage")
full_range_pipeline_based_data_fitting.fit(X_full,y_full)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8)

# Bundle preprocessing and modeling code in a pipeline
pipeline_based_data_fitting = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ("model", model)
    ])
    

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8, enable_categorical=True)
print("cheap_classifier_fit_stage")
pipeline_based_data_fitting.fit(X_Cheap_Cars_classifier_full,y_Cheap_Cars_classifier_full)
print("cheap_classifier_prediction_stage")
cheap_predictions_test = pipeline_based_data_fitting.predict(X_Cheap_Cars_classifier_test)

test_modified_df.num__Cheap_Cars = copy.deepcopy(cheap_predictions_test)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8, enable_categorical=True)
print("average_class_classifier_fit_stage")
pipeline_based_data_fitting.fit(X_Average_Class_Cars_classifier_full,y_Average_Class_Cars_classifier_full)
print("average_class_classifier_prediction_stage")
average_class_predictions_test = pipeline_based_data_fitting.predict(X_Average_Class_Cars_classifier_test)

test_modified_df.num__Average_Class_Cars = copy.deepcopy(average_class_predictions_test)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8, enable_categorical=True)
print("luxury_classifier_fit_stage")
pipeline_based_data_fitting.fit(X_Luxury_Cars_Cars_classifier_full,y_Luxury_Cars_classifier_full)
print("luxury_classifier_prediction_stage")
luxury_predictions_test = pipeline_based_data_fitting.predict(X_Luxury_Cars_Cars_classifier_test)

test_modified_df.num__Luxury_Cars = copy.deepcopy(luxury_predictions_test)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=8, enable_categorical=True)
print("super_car_classifier_fit_stage")
pipeline_based_data_fitting.fit(X_Super_Cars_Cars_classifier_full,y_Super_Cars_classifier_full)
print("super_car_classifier_prediction_stage")
super_car_predictions_test = pipeline_based_data_fitting.predict(X_Super_Cars_Cars_classifier_test)

test_modified_df.num__Super_Cars = copy.deepcopy(super_car_predictions_test)


y_test = copy.deepcopy(test_modified_df.num__price)
X_test = copy.deepcopy(test_modified_df.drop(["id","num__price"], axis=1)).sort_index(axis=1)


test_modified_df.insert(test_modified_df.shape[1], "num__price_2", "0")


for row in range(0,test_modified_df.shape[0]):
    print("cur_row = ", row)
    cur_test_id = test_modified_df.at[row, "id"]
    cur_test_x = copy.deepcopy(test_modified_df[(test_modified_df.id == cur_test_id)].drop(["id","num__price", "num__price_2"], axis=1)).sort_index(axis=1)
    max_model_indication = test_modified_df.at[row, "num__Cheap_Cars"]
    fitting_model = cheap_pipeline_based_data_fitting
    if test_modified_df.at[row, "num__Average_Class_Cars"] > max_model_indication:
        max_model_indication = test_modified_df.at[row, "num__Average_Class_Cars"]
        fitting_model = average_class_pipeline_based_data_fitting
    if test_modified_df.at[row, "num__Luxury_Cars"] > max_model_indication:
        max_model_indication = test_modified_df.at[row, "num__Luxury_Cars"]
        fitting_model = luxury_pipeline_based_data_fitting
    if test_modified_df.at[row, "num__Super_Cars"] > max_model_indication:
        max_model_indication = test_modified_df.at[row, "num__Super_Cars"]
        fitting_model = super_car_pipeline_based_data_fitting
    test_modified_df.at[row, "num__price"] = fitting_model.predict(cur_test_x)
    test_modified_df.at[row, "num__price_2"] = full_range_pipeline_based_data_fitting.predict(cur_test_x)
    





output_1 = pd.DataFrame({'id': test_modified_df.id,'price': test_modified_df.num__price})
output_1.to_csv(output_data_file_path_1, index=False)

output_2 = pd.DataFrame({'id': test_modified_df.id,'price': test_modified_df.num__price_2})
output_2.to_csv(output_data_file_path_2, index=False)


# saving modified combined data frame and test data frame to the csv files

# X_full.to_csv(X_full_file_path)

main_data_df.to_csv(main_data_file_path)

test_modified_df.to_csv(test_modified_file_path)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import pickle
# for Q-Q plots
import columns 
import model_best_hyperparameters
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.simplefilter('ignore')

ds = pd.read_csv('D:/3Kurs/1Sem/SS/model_bank_marketing/data/variant_6.csv')

ds = ds.drop(columns=["Unnamed: 0"],axis=1)

# Detect and remove outliers
def find_skewed_boundaries(df, variable, distance):

    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return upper_boundary, lower_boundary

outliers_iqr_column = dict()
for column in columns.outliers_iqr_column: 
    # outliers_iqr_column[column] = ds[column]
    upper_boundary, lower_boundary = find_skewed_boundaries(ds, column, 4.99)
    outliers = ds.loc[(ds[column] > upper_boundary) | (ds[column] < lower_boundary), column]
    outliers_iqr_column[column] = outliers  

# Missing data imputation
# Counts columns
def impute_na(df, variable, value):
    return df[variable].fillna(value)

median_impute_values = dict()#---------------------------------------------
for column in columns.median_impute_columns:
    median_impute_values[column] = ds[column].median()
    ds[column] = impute_na(ds, column, median_impute_values[column])

percentile_impute_columns = dict()#--------------------------------------
for column in columns.percentile_impute_columns:
    percentile = ds[column].quantile(0.01)  
    ds[column] = ds[column].fillna(percentile) 
    # percentile_impute_columns[column] = ds[column] 

    
any_impute_column = dict()#--------------------------------------
for column in columns.any_impute_column:
    any_impute_column = 0
    ds[column]=impute_na(ds, column, any_impute_column)

# Catecorical columns    
unknown_impute_columns = dict()#--------------------------------------
for column in columns.unknown_impute_columns:
    unknown_impute_columns = 'unknown'
    ds[column].fillna(unknown_impute_columns, inplace=True)
    
most_common_impute_columns = dict()#--------------------------------------
for column in columns.most_common_impute_columns:
    most_common_impute_columns = ds[column].mode()[0]
    # most_common_impute_columns[column] = ds[column]
    ds[column].fillna(most_common_impute_columns, inplace=True)
    
# create_catecory_impute_columns = dict()
# for column in columns.create_catecory_impute_columns:
#     create_catecory_impute_columns[column] = ds[column]
#     ds[column].fillna('unknown', inplace=True)
    
random_impute_columns = dict() #--------------------------------------
for column in columns.random_impute_columns:
    random_impute_columns[column] = ds[column]
    random_sample_train = ds[column].dropna().sample(
        ds[column].isnull().sum(), random_state=0
    )
    random_sample_train.index = ds[ds[column].isnull()].index
    ds[column].fillna(random_sample_train, inplace=True) 


# categorical_cols = columns.ohe_encode_columns
# ohe_encode_columns = OneHotEncoder(sparse=False)
# for column in columns.ohe_encode_columns:
#     ds[column] = ds[column].astype('str')
#     ohe_encode_columns.fit(ds[categorical_cols])
#     ohe_output = ohe_encode_columns.transform(ds[categorical_cols])
#     ohe_output = pd.DataFrame(ohe_output)
#     ohe_output.columns = ohe_encode_columns.get_feature_names_out(categorical_cols)
    
#     ds = ds.drop(columns=categorical_cols)
#     ds = pd.concat([ds, ohe_output], axis=1)
    
#     ds[column].fillna('unknown', inplace=True)
#     ds[column] = ds[column].astype('category')
#     ohe_encode_columns[column] = dict(zip(ds[column], ds[column].cat.codes))  # Створюємо кодування
#     ds[column] = ds[column].cat.codes  

# ohe = OneHotEncoder(sparse_output=False) #---------------------------------
# ohe_encode_columns = dict()
# for column in columns.ohe_encode_columns:
#     ohe_encode_columns[column] = ds[column].astype('str')
# ohe.fit(ohe_encode_columns[column])
# ohe_output = ohe.transform(ohe_encode_columns[column])
# ohe_output = pd.DataFrame(ohe_output, index=ohe_encode_columns.index)
# ohe_output.columns = ohe.get_feature_names_out(column)
# ohe_output = ohe_output.drop(["default_no", "y_no", "loan_imputed_no"], axis=1, errors='ignore')
ohe_encode_columns = dict()
for column in columns.ohe_encode_columns:
    ds[column] = ds [column].astype('category')
    ohe_encode_columns[column] = dict(zip(ds[column], ds[column].cat.codes))
    ds[column] = ds[column].cat.codes

    
    
ordered_integer_encoding_columns = dict()
for column in columns.ordered_integer_encoding_columns:
    ordered_integer_encoding_columns[column] = ds[column].astype('category')
    ds.groupby([column])['y'].mean().sort_values()
    ordered_labels = ds.groupby([column])['y'].mean().sort_values().index
    ordinal_mapping = {k: i for i, k in enumerate(ordered_labels, 0)}
    ds[column] = ds[column].map(ordinal_mapping)

freqeuncy_encoding_columns = dict()
for column in columns.freqeuncy_encoding_columns:
    freqeuncy_encoding_columns[column] = ds[column].astype('category')
    frequency_map = ds[column].value_counts(normalize=True).to_dict()
    ds[column] = ds[column].map(frequency_map)

param_dict = {'outliers_iqr_column':outliers_iqr_column,
            'median_impute_values':median_impute_values,
            'percentile_impute_columns':percentile_impute_columns,
            'any_impute_column':any_impute_column,
            'unknown_impute_columns':unknown_impute_columns,
            'most_common_impute_columns':most_common_impute_columns,
            # 'create_catecory_impute_columns':create_catecory_impute_columns,
            'random_impute_columns':random_impute_columns,
            'ohe_encode_columns':ohe_encode_columns,
            'ordered_integer_encoding_columns':ordered_integer_encoding_columns,
            'freqeuncy_encoding_columns':freqeuncy_encoding_columns
            }
with open('D:/3Kurs/1Sem/SS/model_bank_marketing/pipeline/param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
X = ds[columns.X_column]
y = ds[columns.y_column]

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9)

gb = GradientBoostingClassifier(**model_best_hyperparameters.params)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

y_pred_df = pd.DataFrame(y_pred, columns=['predictions'])
y_pred_df.to_csv('D:/3Kurs/1Sem/SS/model_bank_marketing/data/predictions.csv', index=False)

print('test set metrics: \n', metrics.classification_report(y_test, y_pred))

filename = 'D:/3Kurs/1Sem/SS/model_bank_marketing/models/finalized_model.sav'
pickle.dump(gb, open(filename, 'wb'))
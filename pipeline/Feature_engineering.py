import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from statsmodels.imputation import mice
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import pickle
# for Q-Q plots
import scipy.stats as stats
import columns 
import warnings
warnings.simplefilter('ignore')

# ds = pd.read_csv('D:/3Kurs/1Sem/SS/model_bank_marketing/data/variant_6.csv')
def preprocess_train_data(ds: pd.DataFrame) -> pd.DataFrame:
    # feature engineering
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


    percentile_impute_columns = dict()  # Ensure this is handled correctly
    for column in columns.percentile_impute_columns:
        percentile_impute_columns[column] = ds[column].quantile(0.01)
        # percentile = ds[column].quantile(0.01)  
        # percentile_impute_columns[column] = percentile  # Save the percentile value in param_dict
        ds[column] = impute_na(ds, column, percentile_impute_columns[column])

        
    any_impute_column = dict()#--------------------------------------
    for column in columns.any_impute_column:
        any_impute_column = 0
        ds[column]=impute_na(ds, column, any_impute_column)

    # Catecorical columns    
    unknown_impute_columns = dict()#--------------------------------------
    for column in columns.unknown_impute_columns:
        unknown_impute_columns = 'unknown'
        ds[column].fillna(unknown_impute_columns, inplace=True)
        
    most_common_impute_columns = dict()#-----------------------------------
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
    
    #--------------------------------------------------------------------------------------------------
    ohe_encode_columns = dict()
    for column in columns.ohe_encode_columns:
        ds[column] = ds [column].astype('category')
        ohe_encode_columns[column] = dict(zip(ds[column], ds[column].cat.codes))
        ds[column] = ds[column].cat.codes
    #--------------------------------------------------------------------------------------------------
    
    # categorical_cols = ds.select_dtypes(include='object').columns

    # one_hot_cols = [col for col in categorical_cols if ds[col].nunique() > 2]

    # from sklearn.preprocessing import OneHotEncoder
    # ohe = OneHotEncoder(sparse_output=False)
    
    # for col in one_hot_cols:
    #     ds[col] = ds[col].astype('str')

    # ohe.fit(ds[one_hot_cols])
    # ohe_output = ohe.transform(ds[one_hot_cols])
    # ohe_output = pd.DataFrame(ohe_output)
    # ohe_output.columns = ohe.get_feature_names_out(one_hot_cols)
        
    
    # integer_encoding = dict()
    # for column in columns.integer_encoding:
    #     ordinal_mapping = {
    #         k: i
    #         for i, k in enumerate(ds[column].unique(), 0)
    #     }
    #     integer_encoding = ordinal_mapping
    #     # print(ordinal_mapping)
    #     # print(integer_encoding)
    #     ds[column] = ds[column].map(ordinal_mapping)

    ordered_integer_encoding = dict()
    for column in columns.ordered_integer_encoding:
        # print(columns.ordered_integer_encoding_columns)
        # ds.groupby([column])['y'].mean().sort_values()
        ordered_labels = ds.groupby([column])['y'].mean().sort_values().index
        ordinal_mapping = {k: i for i, k in enumerate(ordered_labels, 0)}
        ordered_integer_encoding[column] = ordinal_mapping
        # print[ordered_integer_encoding]
        ds[column] = ds[column].map(ordinal_mapping)

    # freqeuncy_encoding_columns = dict()
    # for column in columns.freqeuncy_encoding_columns:
    #     freqeuncy_encoding_columns[column] = ds[column].astype('category')
    #     frequency_map = ds[column].value_counts(normalize=True).to_dict()
    #     ds[column] = ds[column].map(frequency_map)
        
    # print(ds.head())

    param_dict = {'outliers_iqr_column':outliers_iqr_column,
                'median_impute_values':median_impute_values,
                'percentile_impute_columns':percentile_impute_columns,
                'any_impute_column':any_impute_column,
                'unknown_impute_columns':unknown_impute_columns,
                'most_common_impute_columns':most_common_impute_columns,
                # 'create_catecory_impute_columns':create_catecory_impute_columns,
                'random_impute_columns':random_impute_columns,
                'ohe_encode_columns':ohe_encode_columns,
                'ordered_integer_encoding':ordered_integer_encoding
                # 'integer_encoding' :integer_encoding,
                # 'freqeuncy_encoding_columns':freqeuncy_encoding_columns
                }
    with open('D:/3Kurs/1Sem/SS/model_bank_marketing/pipeline/param_dict.pickle', 'wb') as handle:
        pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    ds.to_csv('D:/3Kurs/1Sem/SS/model_bank_marketing/data/train_look.csv', index=False)

    return ds

def preprocess_testing_data(ds: pd.DataFrame) -> pd.DataFrame:
    # dropping unnecessary columns
    ds = ds.drop(columns=["Unnamed: 0"], axis=1)

    # Detect and remove outliers
    def find_skewed_boundaries(df, variable, distance):
        IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
        lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
        upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
        return upper_boundary, lower_boundary

    outliers_iqr_column = {}
    for column in columns.outliers_iqr_column:
        upper_boundary, lower_boundary = find_skewed_boundaries(ds, column, 4.99)
        outliers = ds.loc[(ds[column] > upper_boundary) | (ds[column] < lower_boundary), column]
        outliers_iqr_column[column] = outliers

    # Missing data imputation
    with open('D:/3Kurs/1Sem/SS/model_bank_marketing/pipeline/param_dict.pickle', 'rb') as handle:
        param_dict = pickle.load(handle)
        
        
    # if 'duration' not in param_dict['percentile_impute_columns']:
    #     print("*****\n\n*****Error: 'duration' not found in percentile_impute_columns******\n\n******")
    
    def impute_na(df, variable, value):
        return df[variable].fillna(value)    

    for column in columns.median_impute_columns:
        median_value = param_dict['median_impute_values'][column]
        ds[column] = ds[column].fillna(median_value)

    for column in columns.percentile_impute_columns:
        # percentile = ds[column].quantile(0.01)
        # ds[column] = impute_na(ds, column, percentile[column])
        percentile = ds[column].quantile(0.01)
        ds[column] = ds[column].fillna(percentile)
        percentile_value = param_dict['percentile_impute_columns']
        ds[column] = ds[column].fillna(percentile_value)


    for column in columns.any_impute_column:
        ds[column] = ds[column].fillna(0)

    for column in columns.unknown_impute_columns:
        ds[column].fillna('unknown', inplace=True)

    for column in columns.most_common_impute_columns:
        mode_value = ds[column].mode()[0]
        ds[column] = ds[column].fillna(mode_value)

    # for column in columns.create_catecory_impute_columns:
    #     ds[column].fillna('unknown', inplace=True)

    for column in columns.random_impute_columns:
        random_sample_train = param_dict['random_impute_columns'][column]
        ds[column].fillna(random_sample_train, inplace=True)
        
    # print(ds.head())

    # One-hot encoding categorical columns
    # categorical_cols = columns.ohe_encode_columns
    
    
    # for column in columns.ohe_encode_columns:
    #     # Перетворюємо стовпець у тип 'category'
    #     ds[column] = ds[column].astype('category')

    #     if column in param_dict['ohe_encode_columns']:
    #         encoding_map = param_dict['ohe_encode_columns'][column]
    #         ds[column] = ds[column].replace(encoding_map)
    #     else:
    #         encoding_map = dict(zip(ds[column], ds[column].cat.codes))
    #         ds[column] = ds[column].cat.codes
    #         param_dict['ohe_encode_columns'][column] = encoding_map


    # for col in columns.ohe_encode_columns:
    #     ds[col] = ds[col].astype('str')
    #     ohe = param_dict[col]
    #     ohe_output = ohe.transform(ds[columns.ohe_encode_columns])
    #     ohe_output = pd.DataFrame(ohe_output)
    #     ohe_output.columns = ohe.get_feature_names_out(columns.ohe_encode_columns)

    # ds = ds.drop(columns=columns.ohe_encode_columns)
    # ds = pd.concat([ds, ohe_output], axis=1)

    # One-Hot Encoding
    for column in columns.ohe_encode_columns:
        ds[column] = ds[column].astype('category')
        if column in param_dict['ohe_encode_columns']:
            # Використовуємо мапінг з тренування
            encoding_map = param_dict['ohe_encode_columns'][column]
            ds[column] = ds[column].map(encoding_map)
        else:
            # Якщо нові категорії, які не зустрічались у тренувальних даних
            encoding_map = dict(zip(ds[column], ds[column].cat.codes))
            ds[column] = ds[column].cat.codes

    # # Integer Encoding
    # integer_encoding = param_dict['integer_encoding']
    # # integer_encoding = {'married': 1, 'single': 2, 'divorced': 3}
    # print(integer_encoding)
    # for column in columns.integer_encoding:
    #     ds[column] = ds[column].map(integer_encoding)

    # Ordered Integer Encoding
    ordered_integer_encoding = param_dict['ordered_integer_encoding']
    for column in columns.ordered_integer_encoding:
        # print(ordered_integer_encoding[column])
        ds[column] = ds[column].map(ordered_integer_encoding[column])


    # Frequency Encoding
    # freqeuncy_encoding_columns = param_dict['freqeuncy_encoding_columns']
    # for column in columns.freqeuncy_encoding_columns:
    #     frequency_map = param_dict['freqeuncy_encoding_columns'][column]
    #     ds[column] = ds[column].map(frequency_map)


    return ds

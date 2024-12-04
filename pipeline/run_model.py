from train import train_model
from test import test_model
import pandas as pd
from Feature_engineering import preprocess_train_data
from Feature_engineering import preprocess_testing_data



print('Gradient_Boosting_Classifier:\n')

ds = pd.read_csv('D:/3Kurs/1Sem/SS/model_bank_marketing/data/test.csv')
ds2 = pd.read_csv('D:/3Kurs/1Sem/SS/model_bank_marketing/data/train.csv')

# print("Unique categories in train 'marital':", ds2['marital'].unique())
# print("Unique categories in test 'marital':", ds['marital'].unique())


# # ds2 = preprocess_train_data(ds2)
# # print('training\n', ds2.head())
# # ds = preprocess_testing_data(ds)
# # print('testing\n',ds.head())

# # train_model(file_name="train.csv",model_name='Gradient_Boosting_Classifier')
# ds2 = preprocess_train_data(ds2)
# print('training\n', ds2.head())  
# r= 0
# missing = list()
# for x in ds2.columns:
#     if ds2[x].isnull().sum() != 0:
#         print(x, ds2[x].isnull().sum(), "\t",round((ds2[x].isnull().sum()*100)/ds2.shape[0],2),"%")
#         missing.append(x)
#         r+=1

# print("\ncount of missing colomn:", r)

# non_numeric_columns = ds2.select_dtypes(exclude=['number'])

# print("All categorical column")
# for idx, column in enumerate(non_numeric_columns.columns, start=1):
#     print(f"{idx}. {column}")
    
# ds = preprocess_testing_data(ds)
# print('testing\n',ds.head())
# r= 0
# missing = list()
# for x in ds.columns:
#     if ds[x].isnull().sum() != 0:
#         print(x, ds[x].isnull().sum(), "\t",round((ds[x].isnull().sum()*100)/ds.shape[0],2),"%")
#         missing.append(x)
#         r+=1

# print("\ncount of missing colomn:", r)

# non_numeric_columns = ds.select_dtypes(exclude=['number'])

# print("All categorical column")
# for idx, column in enumerate(non_numeric_columns.columns, start=1):
#     print(f"{idx}. {column}")

train_model(file_name="train.csv",model_name='Gradient_Boosting_Classifier')
test_model(file_name='test.csv',model_name='Gradient_Boosting_Classifier')
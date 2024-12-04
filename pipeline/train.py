import pandas as pd
import pickle 
from Feature_engineering import preprocess_train_data
import columns
from sklearn.ensemble import GradientBoostingClassifier
import model_best_hyperparameters


def train_model(file_name: str = 'train.csv', model_name: str = 'desicion_tree'):
    # loading data
    ds = pd.read_csv('D:/3Kurs/1Sem/SS/model_bank_marketing/data/' + file_name)
    
    ds = preprocess_train_data(ds)
    
    X = ds[columns.X_column]
    y = ds[columns.y_column]
    
    gb = GradientBoostingClassifier(**model_best_hyperparameters.params)
    gb.fit(X, y)
    
    with open(f'D:/3Kurs/1Sem/SS/model_bank_marketing/models/Gradient_Boosting_Classifier.pickle', 'wb') as handle:
        pickle.dump(gb, handle, protocol=pickle.HIGHEST_PROTOCOL)
   

import pandas as pd
import pickle
from Feature_engineering import preprocess_testing_data
from sklearn import metrics

import columns


def test_model(file_name: str = 'test.csv', model_name: str = 'Gradient_Boosting_Classifier'):
    
       # loading data
    ds = pd.read_csv('D:/3Kurs/1Sem/SS/model_bank_marketing/data/' + file_name)
    
    ds = preprocess_testing_data(ds)
    
    X = ds[columns.X_column]
    y = ds[columns.y_column]
    
    with open(f'D:/3Kurs/1Sem/SS/model_bank_marketing/models/{model_name}.pickle', 'rb') as f:
        model = pickle.load(f)
        
    predictions = model.predict(X)
    pd.DataFrame(predictions).to_csv('D:/3Kurs/1Sem/SS/model_bank_marketing/data/predictions.csv', index=False)
    
    # printing accuracy of predictions
    print('test set metrics: \n', metrics.classification_report(y, predictions))

    MAPE = metrics.mean_absolute_percentage_error(y,predictions)
    print(f'\nmean_absolute_percentage_error: {MAPE}')

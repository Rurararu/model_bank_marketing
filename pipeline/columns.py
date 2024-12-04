# аномалії
outliers_iqr_column = ['balance',
                      'campaign',
                      'pdays']

# потім код зі справленнями рядочками, які мають більше ніж 5 пропущених значень

# Дропнути колонку Unnamed: 0

# завповнення відсутніх значень
median_impute_columns = ['age', 'previous']

percentile_impute_columns = ['duration']

any_impute_column = ['campaign']

unknown_impute_columns = ['job', 'education', 'contact', 'housing']

most_common_impute_columns = ['marital','default', 'y']

# create_catecory_impute_columns = ['housing']

random_impute_columns = ['loan']

# ohe_encode_columns = ['education', 'default',
#                       'housing', 'contact', 
#                       'poutcome', 'y', 'loan',
#                       'marital', 'job','month']

# кодування категоіальних ознак
ohe_encode_columns = ['education', 'default',
                      'housing', 'contact', 
                      'poutcome', 'y', 'loan']

ohe_encode_more = ['education', 'housing', 
                      'contact', 'poutcome']

ohe_encode_doble = ['default', 'y', 'loan']


ordered_integer_encoding_columns = [ 'marital','job']
ordered_integer_encoding = [ 'job', 'month', 'marital']


integer_encoding = ['job', 'month']

# freqeuncy_encoding_columns = ['month']


y_column = ['y']

X_column = ['age', 'job', 'marital', 
            'education', 'default', 'balance', 
            'housing', 'loan', 'contact', 'day',
            'month', 'duration', 'campaign',
            'pdays', 'previous', 'poutcome']

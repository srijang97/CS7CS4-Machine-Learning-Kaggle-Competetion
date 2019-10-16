
# coding: utf-8

# In[3]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import stats as ss

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import category_encoders as ce
from math import sqrt
import os
import copy

for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Data Processing




def fill_missing_values(df):
# Function for imputation of missing values:

#   This function replaces missing values in all columns by mean or mode of the column or 'Not Specified'/'unknown'

#       input: df (pd.DataFrame)- merged data, train and test
#       returns: df(pd.DataFrame) - data with missing values replaced

    #Dictionary of type (column name, replacing value for missing values)
    values = {'Year of Record': df['Year of Record'].median(), 'Age': df['Age'].median(), 'Gender': 'unknown',
              'Hair Color': 'unknown', 'Wears Glasses': df['Wears Glasses'].mode().values[0], 'Profession': 'Not Specified',
              'University Degree': 'Not specified', 'Income in EUR': df['Income in EUR'].median()}
    
    df = df.fillna(value=values)
    
    return df



def cat_encoding(imputed_df, cat_cols, num_cols, enc_type):
    
# Function for encoding the categorical columns.

#   This function encodes the categorical columns according to the encoding type specified

#       input: df (pd.DataFrame)- merged data with missing values removed
#              cat_cols (list) - list of categorical columns
#              num_cols (list) - list of numerical columns
#              enc_type (string) - Type of encoding: 'binary' (Binary Encoding), 'hash' (Feature Hashing),
#                                  'weighted' for One-Hot and/ or Target Encoding
#       returns: df(pd.DataFrame)- data with categorical columns encoded
    
    if enc_type=='binary':
        encoder = ce.BinaryEncoder(cols= cat_cols)
        encoded_df = encoder.fit_transform(imputed_df)
        return encoded_df
        
    elif enc_type=='hash':
        encoder = FeatureHasher(n_features=500, input_type='string')
        encoded_df = pd.DataFrame(encoder.fit_transform(imputed_df[cat_cols].values).todense())
        encoded_df = pd.concat([encoded_df, imputed_df[num_cols]], axis=1, sort=False)
        return encoded_df
    
    
    if enc_type=='target':
        
        avg_income = imputed_df['Income in EUR'].mean()
        
        for col in cat_cols['target']:
            
            #Index of column values that are in test set but not the train set
            col_index = imputed_df[imputed_df['train']==1].index.union(imputed_df[imputed_df[col].isin(list(set(raw_test_df[col]) - set(raw_train_df[col])))].index)
            
            df_x = imputed_df.loc[col_index, [col, 'Income in EUR']].astype({"Income in EUR": 'int64', col: 'category'}).groupby(by=[col]).mean()            
            
            imputed_df[col] = imputed_df[col].apply(lambda x: (df_x.loc[str(x)][0]))   
        
        if len(cat_cols['onehot'])>0:
            encoded_df = pd.get_dummies(imputed_df[cat_cols['onehot']])
            encoded_df = pd.concat([encoded_df, imputed_df[num_cols+ cat_cols['target']]], axis=1, sort=False)
            
            return encoded_df
        
        else:
            return imputed_df[num_cols+cat_cols['target']]
    

def scaling(data, scaler_type):
# Function for scaling the data.

#   This function scales the columns according to the method specified

#       input: df (pd.DataFrame)- merged data with categorical values encoded
#              scaler_type (string) - Type of scaling: 'standard': Standard Scaling, 'minmax' : MinMax Scaling
#       returns: df(pd.DataFrame) - data with columns scaled

    if scaler_type=='standard':
        scaler = StandardScaler()
    elif scaler_type=='minmax':
        scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    return data


def preprocess_data(raw_train_df, raw_test_df, y, validation_split, validation_ratio, cat_cols, num_cols, encoder, scaler):
# Function for preprocessing the data. 

#   This function makes calls to the fill_missing_values(), cat_encoding() and scaling() functions.
#   It also splits the data into validation and training sets. 
#       input: raw_train_df (pd.DataFrame)- raw train data
#              raw_test_df (pd.DataFrame)- raw test data
#              y (pd.Series) - training labels
#              validation_split (Boolean): True if training and validation split required
#              validation_ratio (float): Number between 0 and 1 for the proportion of training data in the validation set.
#              cat_cols (list) - list of categorical columns
#              num_cols (list) - list of numerical columns
#              encoder (string) - Type of encoding: 'binary' (Binary Encoding), 'hash' (Feature Hashing),
#                                  'weighted' for One-Hot and/ or Target Encoding
#              scaler_type (string) - Type of scaling: 'standard': Standard Scaling, 'minmax' : MinMax Scaling,
#                                                       'none': no scaling
#       returns: Numpy Arrays: X, y (training features, training labels) and X_test (testing features) if no validation split
#                           OR X, X_train, X_valid, y_train, y_valid, X_test if validation split  
    

    train_df = raw_train_df.copy().drop(labels = ['Instance'], axis=1)
    test_df = raw_test_df.copy().drop(labels = ['Instance'], axis=1)
    train_df['train']= 1
    test_df['train']=0
    
    #Concatenate datasets, fill missing values and perform categorical encoding
    merged_df = pd.concat([train_df, test_df], sort=False)
    
    merged_df = fill_missing_values(merged_df)
      
    merged_df = cat_encoding(merged_df, cat_cols, num_cols, encoder)
    
    #Separate datasets into two numpy arrays.
    train_df = merged_df[merged_df['train']==1]
    test_df = merged_df[merged_df['train']==0]
    
    X = train_df.drop(labels=['train'], axis=1).values
    X_test = test_df.drop(labels=['train'], axis=1).values
    
    if scaler!='none':
        scale_len = len(num_cols) + len(cat_cols['weighted'])
        X[:, -5] = np.log(1 + X[:, -5])
        X[:, -scale_len:] = scaling(X[:, -scale_len:], scaler)
        X_test[:, -scale_len:] = scaling(X_test[:, -scale_len:], scaler)
    
    if validation_split:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=validation_ratio, random_state=0)
        return X, X_train, X_valid, y_train, y_valid, X_test
    else:
        return X, y, X_test
    

def xgb_cross_validation(params):
    kfold = KFold(5, random_state=42, shuffle=False)
    
    data_dmatrix = xgb.DMatrix(data=X, label=y)
    
    cvresults = xgb.cv(params, data_dmatrix, num_boost_round=5000, nfold=5,folds=kfold, metrics='rmse', 
                       seed=42, early_stopping_rounds = 50)
    return cvresults

def fit_model(method, model, params, X, y, X_train, y_train, X_valid, y_valid, X_test):
# Function for fitting regression models to the data. 

#   This function fits Linear Regression, Rudge Regression, Random Forest Regression and XGBoost regression models according
#   to specification. It also performs normal validation or K-Fold Cross validation if needed, otherwise fits the model on 
#   the whole dataset.


#       input: method (String)- 'kfold' for K-Fold Cross Validation, 'validation' for normal validation,
#                                'testing' for fitting model on whole training data.
#              model (String)- The model to apply: 'linear' (linear regression), 'ridge' (ridge regression), 
#                              'randomforest' (Random Forest regression), 'xgb' (XGBoost regression)                              
#              params (python dict) - parameters for the model
#              X, y  (np.array): Full training data and labels
#              X_train, y_train (np.array): Training data and label after validation split
#              X_valid, y_valid - Validation data after validation split
#              X_test - Whole testing data

#       returns: y_pred (np.array) - Predictions if method = 'testing'
#                otherwise prints rmse score for validation and mean and standard deviation of rmse scores for K-Fold

    if model=='linear':
        regressor = LinearRegression()
    elif model=='ridge':
        regressor = Ridge(alpha=params['alpha'])
    elif model=='randomforest':
        regressor = RandomForestRegressor(n_estimators=params['n_estimators'], max_features='auto')
    elif model=='xgb':
        regressor = xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = params['learning_rate'], 
                                              max_depth = params['max_depth'], n_estimators=params['n_estimators'],
                                              min_child_weight=params['min_child_weight'], 
                                              colsample_bytree=params['colsample_bytree'], subsample=params['subsample'], 
                                              gamma=params['gamma'], alpha=params['alpha'], tree_method = 'gpu_hist')
    
    if method=='kfold':
        kfold = KFold(5, random_state=42, shuffle=False)
        rmse_scores = []
        for train_index, test_index in kfold.split(X):
            X_train, y_train, X_valid, y_valid = X[train_index], y[train_index], X[test_index], y[test_index]
        
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_valid)
            rmse_scores.append(sqrt(mean_squared_error(y_valid, y_pred)))
        print("Mean: ", np.mean(rmse_scores), " Std: ", np.std(rmse_scores))            
            
    elif method=='validation':
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_valid)
        print('RMSE Score: ', sqrt(mean_squared_error(y_valid, y_pred)))  
    
    elif method=='testing':
        regressor.fit(X, y)
        y_pred = regressor.predict(X_test)
        return y_pred
    
#############################################################################################    

#Import training and testing data from local drive
raw_train_df = pd.read_csv("data/tcd ml 2019-20 income prediction training (with labels).csv")
raw_test_df = pd.read_csv("data/tcd ml 2019-20 income prediction test (without labels).csv")

#Drop rows with negative income in training set
raw_train_df.drop(raw_train_df[raw_train_df['Income in EUR']<0].index, inplace=True)

#Define columns
cat_cols={'onehot':[], 'target': ['Country', 'Profession', 'University Degree', 'Gender']}
num_cols = ['Year of Record', 'Age', 'Size of City','Body Height [cm]', 'train']

X_test_instances = raw_test_df['Instance'].values
y = raw_train_df['Income in EUR'].values

#Get Preprocessed Data
X, X_train, X_valid, y_train, y_valid, X_test = preprocess_data(raw_train_df, raw_test_df, y, True, 0.2, cat_cols, num_cols, 'target', 'none')

#Fit the model and get predictions:
xgb_params = {
    'n_estimators': 2365,
    'max_depth': 5,
    'min_child_weight':1,
    'gamma': 0,
    'colsample_bytree': 0.9,
    'subsample': 0.6,
    'gamma': 0.3,
    'learning_rate': 0.01,
    'objective': 'reg:squarederror',
    'alpha': 1
}

y_pred = fit_model('testing', 'xgb', xgb_params, X, y, X_train, y_train, X_valid, y_valid, X_test)

#Save results to local disk

pd.DataFrame({'Instance': X_test_instances, 'Income': y_pred}).to_csv('data/tcd ml 2019-20 income prediction submission file.csv', index=False)



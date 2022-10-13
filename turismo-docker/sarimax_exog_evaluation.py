#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:41:12 2022

@author: sandro
"""
import pandas as pd      
import statsmodels.api as sm
from itertools import product  
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns # For pairplots and heatmaps
import os
import warnings
import numpy as np
# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

def read_dataset(path_name, country_name, start_date, end_date):
    
    dataset = pd.read_excel(path_name, sheet_name=country_name, skiprows=10)
   
    # Set date as index
    datetime = pd.DataFrame({'year':dataset['Año'] , 'month': dataset['Mes'], 'day': 1})
    dataset = dataset.set_index(pd.to_datetime(datetime))
    dataset.index.name = 'fecha'
    
    # Take a range of datetime to work with
    dataset = split_by_datetime(dataset, start_date, end_date)
    dataset = dataset.drop(['Año','Mes'], axis=1)    
    return dataset
    
def split_by_datetime(dataset, start_date, end_date) -> pd.DataFrame:
    """
    Given a pandas dataset and start date/end date, select a portion of this dataset
    
    Date format: 'yyyy-mm-dd'
    
    Returns a portion of the original dataset
    """
    mask = (dataset.index>= start_date) & (dataset.index <= end_date)
    dataset = dataset.loc[mask]
    return dataset

# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    print(train.index[0])
    print(train.index[-1])
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True): 
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    # p_params = [0, 1, 2, 3]
    # d_params = [0, 1, 2]
    # q_params = [0, 1, 2, 3]
    # t_params = ['n','c']
    # P_params = [0, 1, 2, 3]
    # D_params = [0, 1]
    # Q_params = [0, 1, 2, 3 ]
    # m_params = seasonal
    
    p_params = [1]
    d_params = [1]
    q_params = [1]
    t_params = ['n']
    P_params = [1]
    D_params = [1]
    Q_params = [1]
    m_params = seasonal
    
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p,d,q), (P,D,Q,m), t]
                                    models.append(cfg)
    return models

def plot(prediction, dataset, target_col):
    plt.figure(figsize=(25,5))
    plt.rcParams.update({'font.size': 10})
    # Plot time series
    plt.plot(dataset[-80:].index, dataset[-80:][target_column], label='Actual Data', 
             linewidth=3)
    plt.plot(dataset[-36:].index, prediction, color='orange', 
             label='Predictions', linewidth=3)
    
    # Add title and labels
    plt.title('Predicción del número de '+str(target_col))
    plt.xlabel('Fecha')
    plt.ylabel(str(target_col))

    # Add legend
    plt.legend()
    # Auto space
    plt.tight_layout()
    # Display plot
    plt.show() 
    
def compute_metrics(test: ndarray, pred: ndarray) -> dict:
    """
    Provided the forecasted and test data, computes several metrics.

    Returns a dictionary of metrics
    """
    
    mae = mean_absolute_error(test, pred)
    rmse = math.sqrt(mean_squared_error(test, pred))
    r2 = r2_score(test, pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2}

def metrics_time_series_cv(dataset,target_column, order,s_order,trend):
    
    skip_years = 3 # skip the last three years 
    dataset_train = dataset.iloc[:-12*skip_years]
    dataset_test = dataset.iloc[-12*skip_years:]

    # Split dataframe in X_train, X_test, y_train, y_test 
    train = dataset_train[[target_column]]
    train = np.asarray(train, dtype=float)
    test = dataset_test[[target_column]]
    
    # PREDICCIÓN DE TRES AÑOS
    best_model=sm.tsa.statespace.SARIMAX(endog=train,trend=trend, 
                                    order=order, 
                                    seasonal_order=s_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit(method = 'powell', disp=False)

    test_date = str(dataset_test.index[-2]).split('-')
    test_year = test_date[0]
    print("Model evaluation in test data ("+test_year+")" )
    forecast = best_model.forecast(steps=36)
    print("Metrics TEST DATA ("+test_year+")")
    
    metrics = compute_metrics(np.asarray(test[:-12], dtype=float), forecast[:-12])
    print(metrics)
    plot(forecast, dataset_test,target_column)
    
    return forecast, metrics

def get_params(scores):
    s = scores[0][0] # Get the best configuration
    p,d,q = int(s[2]), int(s[5]), int(s[8])
    P,D,Q = int(s[13]), int(s[16]), int(s[19])
    seasonal = 12
    trend = s[28]
    order = (p,d,q)
    s_order = (P,D,Q,seasonal)
    return order, s_order, trend
    
    
if __name__ == '__main__':
    
    # Path desde donde leemos el excel
    path_name = "./data/predicciones_viajeros.xlsx"
    # Path para guardar las predicciones
    path_csv = "./resultados_preds/forecasting.csv"
    # Path para guardar las metricas y las configuraciones 
    path_metrics = "./resultados_preds/metrics.csv"
    country_name = 'España'
    start_date = '2009-01-01'
    end_date = '2019-12-01'
    dataset = read_dataset(path_name, country_name, start_date, end_date)
    
    """  
    Haremos predicciones para cada una de las variables exógenas
    Entrenamos el modelo con los siguiente conjuntos de datos para ofrecer forecasting de 3 años
    train [2009-2014] --> test[2015-2017]
    train [2009-2015] --> test[2016-2018]
    train [2009-2016] --> test[2017-2019] 
    """
    exogs = ['Pib Pc', 'IPC Armonizado','Desempleo Armonizado', 'Asientos ofertados', 'Llegadas a AGP España'] 
    dates = ['2017-12-01'] #,'2018-12-01','2019-12-01'
    list_predictions = []
    list_metrics = []
    for dt in dates:
        list_preds = []
        for target_column in exogs:
            dataset_split = split_by_datetime(dataset, start_date, dt)
            data = dataset_split[target_column]    
            # data split: Escogemos como conjunto de test de los últimos tres años
            n_test = 12*3
            # model configs
            cfg_list = sarima_configs(seasonal=[0, 12])
            # grid search
            scores = grid_search(data, cfg_list, n_test)
            print('done')
            
            # Get metrics for predictions and plot it 
            order, s_order, trend = get_params(scores)
            predictions, metrics = metrics_time_series_cv(dataset_split, target_column, order,s_order,trend)
            # Save predictions
            dc = {str(target_column): predictions[-12:]}    
            df = pd.DataFrame(dc, index= dataset_split.index[-12:]) 
            list_preds.append(df)
            # save metrics
            metrics["Params"] = scores[0][0]
            metrics["exog"] = target_column
            # print(metrics)
            df_metrics = pd.DataFrame(metrics, index=dataset_split.index[-1:])
            list_metrics.append(df_metrics)
        # Necesitamos concatenar los dataframes por fecha (en horizontal)
        df_preds = pd.concat(list_preds,axis=1, join="inner")
        list_predictions.append(df_preds)
        
    # Aqui concatenamos los dataframes por variables (en vertical)
    dataframe = pd.concat(list_predictions)
    dataframe_metrics = pd.concat(list_metrics) 
    # saving the dataframe 
    dataframe.to_csv(path_csv) 
    dataframe_metrics.to_csv(path_metrics)
    print(dataframe.head())
    print("DONE")












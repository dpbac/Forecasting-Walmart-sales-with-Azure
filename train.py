""" In this script we provide code used to prepare data and perform training using Light GBM on the subset 
of the Walmart data that can be found at https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/tree/master/data """"


import argparse
import os
import numpy as np
import pandas as pd
from datetime import timedelta
import lightgbm as lgb
from azureml.core import Run

def replace_nan_events(df):
    """ Replace nan events with "no_event"
    
    Args:
        df: dataframe with data including events
        
    Return:
        df: dataframe without nan in event columns
    """
    # columns about events
    event_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    
    for feature in event_features:
        df[feature].fillna('no_event', inplace = True)
        
    return df

def change_data_type(df):
    """ 
    Transform categorical features into the appropriate type that is expected by LightGBM 
    
    Args: 
    
        df: dataframe data for forecasting
    
    Return:
    
        df: dataframe with data type modified where needed
    
    """
    for c in df.columns:
        col_type = df[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            df[c] = df[c].astype('category')
            
    return df

def create_lag_features(df, forecast_horizon):
    """ 
    Create lags using forecast horizon as basis 
    
    Args:
        df: dataframe with data for forecasting
    
        forecast_horizon (int): Number of time units (e.g. days) to be forecasted
    
    Return:
        df: dataframe updated containing features created
    """
    
    for num in range(3):
        df['lag_t'+str(forecast_horizon+num)] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(forecast_horizon+num))

    return df

def create_df_rolling_stats(df):
    """ 
    Create statistical features based on demand using rolling windows of different sizes.
    
    Args:
        df: dataframe with data for forecasting including variable demand
        
    Return
        df: dataframe updated containing features created
    """
    
    df['rolling_mean_t7']   = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    df['rolling_std_t7']    = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    df['rolling_mean_t30']  = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    df['rolling_std_t30']   = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    df['rolling_mean_t90']  = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    df['rolling_std_t90']  = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).std())
    df['rolling_mean_t180'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    df['rolling_std_t180'] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).std())

    return df

def create_features_price(df):
    """ 
    Create features based on price.
    
    Args:
        df: dataframe with data for forecasting including variable related to price
    
    Return:  
        df: dataframe updated containing features created
        
    """
    df['lag_price_t1'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    df['price_change_t1'] = (df['lag_price_t1'] - df['sell_price']) / (df['lag_price_t1'])

    df['rolling_price_max_t365'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    df['price_change_t365'] = (df['rolling_price_max_t365'] - df['sell_price']) / (df['rolling_price_max_t365'])

    df.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)

    df['rolling_price_std_t7'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    df['rolling_price_std_t30'] = df.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    
    return df

def identify_weekend(day):
    """ Returns 1 if it is a weekend day
    
    Args:
        day (int): day of week 
    Return:
        1: is weekend day
        0: is NOT a weekend day
    """
    if day==5 or day==6:
        return 1
    else:
        return 0

def create_date_features(df):
    """ Create features based on date
    
    Args:
        df : dataframe containg a datetime feature
        
    Return:
        df: dataframe updated containing features created
        
    """
    
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week'] = df['date'].dt.week
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_month_start'] = (df['date'].dt.is_month_start).astype(int)
    df['is_month_end'] = (df['date'].dt.is_month_end).astype(int)
    df['is_weekend'] = df['day_of_week'].apply(identify_weekend)
    
    return df

def create_revenue_features(df):
    """ Create features based on revenue as defined by demand * price
    
    Args:
        df: dataframe containing demand and price of items
    
    Return:
        df: dataframe updated containing features created
    
    """

    df['revenue'] = df['demand'] * df['sell_price']
    df['lag_revenue_t1'] = df.groupby(['id'])['revenue'].transform(lambda x: x.shift(28))
    df['rolling_revenue_std_t28'] = df.groupby(['id'])['lag_revenue_t1'].transform(lambda x: x.rolling(28).std())
    df['rolling_revenue_mean_t28'] = df.groupby(['id'])['lag_revenue_t1'].transform(lambda x: x.rolling(28).mean())

    df.drop(['revenue'],axis=1,inplace=True)
    
    return df

def split_train_test(df,forecast_horizon, gap):
    """ 
    Split time-series data in training and test datasets.
    
    Args:
        df: Dataframe with forecasting data
        forecast_horizon (int): Number of time units (e.g. days) to forecast in the future, or the test period length
        gap (int): Gap (in days) between the training and test data. This is to allow business managers to plan for the 
        forecasted demand.
        
    Return:
        df_train: Train dataframe covering first day of available date until current day
        df_test: Test dataframe covering period of forecast horizon days
        
    """
    
    last_day_train = df['date'].max() - timedelta(days=forecast_horizon) - timedelta(days=gap)
    first_day_test = last_day_train + timedelta(days=1)
    last_day_test = df['date'].max()
    
    print("First day training dataset:{}".format(df['date'].min()))
    print("Last day training dataset:{}".format(last_day_train))
    print("First day test dataset:{}".format(first_day_test))
    print("Last day test dataset:{}".format(last_day_test))

    # split dataset using the above limits
    
    df_train = df[df['date']<=last_day_train]
    df_test = df[df['date']>=first_day_test]
    
    return df_train, df_test

def create_features(df,forecast_horizon):
    """ 
    Args:
        df: dataframe with data for forecasting    
        forecast_horizon (int): Number of time units (e.g. days) to be forecasted
    
    Return:
        df: dataframe updated containing features created
    """
    # clean df and create features
    
    df = replace_nan_events(df)
    df = change_data_type(df)
    df = create_lag_features(df, forecast_horizon)
    df = create_df_rolling_stats(df)
    df = create_features_price(df)
    df = create_date_features(df)
    df = create_revenue_features(df)
    
    # Remove rows with nan
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=str, dest="data_folder", default=".", help="data folder mounting point")
    parser.add_argument("--num-leaves", type=int, dest="num_leaves", default=64, help="# of leaves of the tree")
    parser.add_argument("--min-data-in-leaf", type=int, dest="min_data_in_leaf", default=50, help="minimum # of samples in each leaf")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=0.001, help="learning rate")
    parser.add_argument("--feature-fraction",type=float,dest="feature_fraction",default=1.0,help="ratio of features used in each iteration")
    parser.add_argument("--bagging-fraction",type=float,dest="bagging_fraction",default=1.0,help="ratio of samples used in each iteration")
    parser.add_argument("--bagging-freq", type=int, dest="bagging_freq", default=1, help="bagging frequency")
    parser.add_argument("--max-rounds", type=int, dest="max_rounds", default=400, help="# of boosting iterations")
    args = parser.parse_args()
    args.feature_fraction = round(args.feature_fraction, 2)
    args.bagging_fraction = round(args.bagging_fraction, 2)
    print(args)

    # Start an Azure ML run
    run = Run.get_context()

    # Data paths
    DATA_DIR = args.data_folder
    
    # Data and forecast problem parameters
    time_column_name = 'date'
    forecast_horizon = 28
    gap = 0

    
  # Parameters of GBM model
    params = {
        "objective": "root_mean_squared_error",
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "learning_rate": args.learning_rate,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "num_rounds": args.max_rounds,
        "early_stopping_rounds": 125,
        "num_threads": 16,
    }
    
    print(params)
    
    
    # Train and validate the model using only the first round data
    r = 0
    print("---- Round " + str(r + 1) + " ----")
    
    # Load training data
    default_train_file = os.path.join(DATA_DIR, "train.csv")
    if os.path.isfile(default_train_file):
        df_train = pd.read_csv(default_train_file,parse_dates=[time_column_name])
        print(df_train.head())
    else:
        df_train = pd.read_csv(os.path.join(DATA_DIR, "train_" + str(r + 1) + ".csv"),parse_dates=[time_column_name])
        
    # transform object type to category type to be used by lgbm
    df_train = change_data_type(df_train)
    
    # Split train data into training dataset and validation dataset
    df_train_2, df_val = split_train_test(df_train,forecast_horizon, gap)
    
    # Get features and labels
    X_train=df_train_2.drop(['demand'],axis=1)
    y_train=df_train_2['demand']
    X_val=df_val.drop(['demand'],axis=1)
    y_val=df_val['demand']
    
    X_train.drop(columns='date',inplace=True)
    X_val.drop(columns='date',inplace=True)
    
    d_train = lgb.Dataset(X_train, y_train)
    d_val = lgb.Dataset(X_val, y_val)
    
    print(X_train.info())
    
    # A dictionary to record training results
    evals_result = {}

    # Train LightGBM model
    bst = lgb.train(params, d_train, valid_sets=[d_train, d_val], categorical_feature="auto", evals_result=evals_result)

    # Get final training loss & validation loss 
    print(evals_result["training"].keys())
    train_loss = evals_result["training"]["rmse"][-1]
    val_loss = evals_result["valid_1"]["rmse"][-1]
    print("Final training loss is {}".format(train_loss))
    print("Final test loss is {}".format(val_loss))
    
    y_max = y_val.max()
    y_min = y_val.min()
    y_diff = (y_max - y_min)
    
    # Log the validation loss (NRMSE - normalized root mean squared error)
    run.log("NRMSE", np.float(val_loss/y_diff))
 
 
    #Dump the model using joblib
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=bst, filename='outputs/bst-model.pkl')

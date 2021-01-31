#check the following list
import argparse
import os
import numpy as np
import pandas as pd
from datetime import timedelta
import lightgbm as lgb
from azureml.core import Run


# from sklearn.metrics import mean_squared_error
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from azureml.data.dataset_factory import TabularDatasetFactory


# import math


#Load dataset
# df = pd.read_csv('https://raw.githubusercontent.com/dpbac/Forecasting-Walmart-sales-with-Azure/master/data/walmart_tx_stores_10_items.csv?token=AEBB67J22CSLJ35FSAYTPELACF5W4')
# or the following?
# df = TabularDatasetFactory.from_delimited_files(path='https://raw.githubusercontent.com/dpbac/Forecasting-Walmart-sales-with-Azure/master/data/walmart_tx_stores_10_items.csv?token=AEBB67J22CSLJ35FSAYTPELACF5W4')

# run = Run.get_context()

# Functions

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

# If there is a problem with the change of data type
# from sklearn.preprocessing import LabelEncoder

# def encode_categorical(df):
#     """ Transform categorical features in numerical features.
    
#     Args:
#         df: dataframe containing categorical features
        
#     Return:
#         df: dataframe with numerical features
#     """
    
#     cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 
#        'event_name_2', 'event_type_2']

#     for feature in cat:
#         encoder = LabelEncoder()
#         df[feature] = encoder.fit_transform(df[feature])
        
#     return df

# create features

# demand based features

def create_lag_features(df, forecasting_horizon):
    """ 
    Create lags using forecast horizon as basis 
    
    Args:
        df: dataframe with data for forecasting
    
        forecasting_horizon (int): Number of time units (e.g. days) to be forecasted
    
    Return:
        df: dataframe updated containing features created
    """
    
    for num in range(3):
        df['lag_t'+str(forecasting_horizon+num)] = df.groupby(['id'])['demand'].transform(lambda x: x.shift(forecasting_horizon+num))

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

# price based features

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

# date based features

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

# revenue based features

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

# Split time series data

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
    
    last_day_train = df['date'].max() - timedelta(days=forecasting_horizon) - timedelta(days=gap)
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

def main():
    
    
    # Parse input arguments
    parser = argparse.ArgumentParser()
#     parser.add_argument("--data-folder", type=str, dest="data_folder", default=".", help="data folder mounting point")
    parser.add_argument("--num-leaves", type=int, dest="num_leaves", default=64, help="# of leaves of the tree")
    parser.add_argument("--min-data-in-leaf", type=int, dest="min_data_in_leaf", default=50, help="minimum # of samples in each leaf")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate", default=0.001, help="learning rate")
    parser.add_argument("--feature-fraction",type=float,dest="feature_fraction",default=1.0,help="ratio of features used in each iteration")
    parser.add_argument("--bagging-fraction",type=float,dest="bagging_fraction",default=1.0,help="ratio of samples used in each iteration")
    parser.add_argument("--bagging-freq", type=int, dest="bagging_freq", default=1, help="bagging frequency")
    parser.add_argument("--max-rounds", type=int, dest="max_rounds", default=400, help="# of boosting iterations")
    
#     parser.add_argument("--max-lag", type=int, dest="max_lag", default=10, help="max lag of unit sales")
#     parser.add_argument("--window-size", type=int, dest="window_size", default=10, help="window size of moving average of unit sales")
    
    args = parser.parse_args()
    args.feature_fraction = round(args.feature_fraction, 2)
    args.bagging_fraction = round(args.bagging_fraction, 2)
    print(args)

    # Start an Azure ML run
    run = Run.get_context()

#     # Data paths
#     DATA_DIR = args.data_folder
#     TRAIN_DIR = os.path.join(DATA_DIR, "train")

    # Data and forecast problem parameters
    time_column_name = 'date'
    forecasting_horizon = 28
    gap = 0

    # Parameters of GBM model
    params = {
        "objective": "mean_absolute_error",
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
    
    # Train and validate the model using only the first round data
#     r = 0
#     print("---- Round " + str(r + 1) + " ----")
#     # Load training data
#     default_train_file = os.path.join(TRAIN_DIR, "train.csv")
#     if os.path.isfile(default_train_file):
#         train_df = pd.read_csv(default_train_file)
#     else:
#         train_df = pd.read_csv(os.path.join(TRAIN_DIR, "train_" + str(r + 1) + ".csv"))
        
    # Load data
    data = pd.read_csv("../data/walmart/walmart_tx_stores_10_items.csv",parse_dates=[time_column_name])
        
        
    # clean data and create features
    
    data = replace_nan_events(data)
    # data = encode_categorical(data)
    data = change_data_type(data)
    data = create_lag_features(data, forecasting_horizon)
    data = create_df_rolling_stats(data)
    data = create_features_price(data)
    data = create_date_features(data)
    data = create_revenue_features(data)
    
    # Remove rows with nan
    data.dropna(inplace=True)
    
    # Create a training/testing split
    df_train, df_test = split_train_test(data,forecast_horizon, gap)
    
    X_train=df_train.drop(['demand'],axis=1)
    y_train=df_train['demand']
    X_test=df_test.drop(['demand'],axis=1)
    y_test=df_test['demand']
    
     train_fea, valid_fea, train_label, valid_label = train_test_split(
        train_fea.drop("move", axis=1, inplace=False), train_fea["move"], test_size=0.05, random_state=1
    )
    
    d_train = lgb.Dataset(X_train, y_train)
    d_test = lgb.Dataset(X_test, y_test)
    
    # A dictionary to record training results
    evals_result = {}
    
    # Train LightGBM model
    bst = lgb.train(
        params, d_train, valid_sets=[d_train, d_test], categorical_feature="auto", evals_result=evals_result)
    
    # Get final training loss & validation loss
    train_loss = evals_result["training"]["mean_absolute_error"][-1]
    valid_loss = evals_result["test"]["mean_absolute_error"][-1]
    print("Final training loss is {}".format(train_loss))
    print("Final validation loss is {}".format(valid_loss))

    # Log the validation loss (MAE)
    run.log("MAE", np.float(valid_loss) * 100)

    # Files saved in the "./outputs" folder are automatically uploaded into run history
    os.makedirs("./outputs/model", exist_ok=True)
    bst.save_model("./outputs/model/best-model.txt")
    

## DO THIS PART BASED OM MY PROBLEM

# def main():
#     # Add arguments to script
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
#     parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

#     args = parser.parse_args()

#     run.log("Regularization Strength:", np.float(args.C))
#     run.log("Max iterations:", np.int(args.max_iter))
    
#     x, y = clean_data(ds)

#     # I'm using stratify since the data is a bit imbalanced 
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123, stratify = y)


#     model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

#     accuracy = model.score(x_test, y_test)
    
#     os.makedirs('outputs',exist_ok = True)
    
#     joblib.dump(model,'outputs/model.joblib')
    
    
#     run.log("Accuracy", np.float(accuracy))
    
    
    
    

if __name__ == '__main__':
    main()
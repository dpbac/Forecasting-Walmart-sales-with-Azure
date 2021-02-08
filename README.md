<img align="left" width="100" height="75" src="https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/microsoft-azure-640x401.png">
<img align="right" width="100" height="70" src="https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/Walmart1_Logo-scaled.jpg">


# Forecasting Walmart Sales with Azure 

In this repository we present the Capstone project of **Udacity Nanodegree program Machine Learning Engineer with Microsoft Azure**.
In this last project, we create two models to solve a forecasting problem: one using `Automated ML` and one customized model whose hyperparameters are tuned using `HyperDrive`. Then we compare the performance of both the models and deploy the best performing model as a web service.

In particular, we chose a `Light GBM` as our customized model to have hyperparameters optimized by HyperDrive.

## Architecture Diagram

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/architecture_diagram.JPG)


## Dataset

### Overview

The dataset used in this project is a small subset of a much bigger dataset made available at Kaggle's competition [M5 Forecasting - Accuracy Estimate the unit sales of Walmart retail goods](https://www.kaggle.com/c/m5-forecasting-accuracy/overview/description).

The complete dataset covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events.(e.g. Super Bowl, Valentine’s Day, and Orthodox Easter) that typically affect unit sales and could improve forecasting accuracy. 

In order to demonstrate the use of Azure ML in forecasting we used the available data and created a reduced dataset with **10 products of the 3 Texas stores of Walmart**. 

The data used to build our reduced dataset consists of the following datasets:

* **calendar.csv** - Contains information about the dates on which the products are sold.
* **sell_prices.csv** - Contains information about the price of the products sold per store and date.
* **sales_train_evaluation.csv** - Includes sales [d_1 - d_1941] (labels used for the Public leaderboard)

**File 1:** `calendar.csv` 

Contains information about the dates the products are sold.

    * date: The date in a “y-m-d” format.
    * wm_yr_wk: The id of the week the date belongs to.
    * weekday: The type of the day (Saturday, Sunday, …, Friday).
    * wday: The id of the weekday, starting from Saturday.
    * month: The month of the date.
    * year: The year of the date.
    * event_name_1: If the date includes an event, the name of this event.
    * event_type_1: If the date includes an event, the type of this event.
    * event_name_2: If the date includes a second event, the name of this event.
    * event_type_2: If the date includes a second event, the type of this event.
    * snap_CA, snap_TX, and snap_WI: A binary variable (0 or 1) indicating whether the stores of CA, TX or WI allow SNAP  purchases on the examined date. 1 indicates that [SNAP](https://en.wikipedia.org/wiki/Supplemental_Nutrition_Assistance_Program) purchases are allowed.
    
    
**File 2:** `sell_prices.csv`

Contains information about the price of the products sold per store and date.

    * store_id: The id of the store where the product is sold. 
    * item_id: The id of the product.
    * wm_yr_wk: The id of the week.
    * sell_price: The price of the product for the given week/store. The price is provided per week (average across seven days). If not available, this means that the product was not sold during the examined week. Note that although prices are constant at weekly basis, they may change through time (both training and test set).  

**File 3:** `sales_train.csv` 
Contains the historical daily unit sales data per product and store.

    * item_id: The id of the product.
    * dept_id: The id of the department the product belongs to.
    * cat_id: The id of the category the product belongs to.
    * store_id: The id of the store where the product is sold.
    * state_id: The State where the store is located.
    * d_1, d_2, …, d_i, … d_1941: The number of units sold at day i, starting from 2011-01-29. 

Basicaly we removed data from other states, i.e., **CA** and **WI** and we kept only the 10 products of the department **HOBBIE_2**. Details on how the new dataset was created can be seen in notebook [01-walmart_data_preparation](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/01-create_sample_data_walmart.ipynb).

### Task

Our task in this project it to forecast daily sales for the next `28 days` of products in `HOBBIES_2` department for the stores of Walmart in Texas.

All the features listed above considering items of `HOBBIES_2` department of Walmart were used. In addition, these features were used as basis to create other features in order to help improving the performance of the model. 

More details and the code can be found in the script [train.py](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/train.py)

After preprocessing the data (cleaning and creating features) it was split as shown in the image below. The `Training` part was used for training and validation while the `Testing` was used to test the model. For our experiments we used `GAP=0`. Details about `Splitting Time Dataset` can be found in the [Automated ML notebook](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/automl-final-version-080221.ipynb).

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/time_series_split.JPG)

### Access

The original dataset was downloaded from [Kaggle]( https://www.kaggle.com/c/m5-forecasting-accuracy/data). The subset created from this data was made available at 
[GitHub]( https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/tree/master/data). 

The GitHub address of the [raw data](https://raw.githubusercontent.com/dpbac/Forecasting-Walmart-sales-with-Azure/master/data/walmart_tx_stores_10_items_with_day.csv) is used to load the data in the Azure workspace. 

For both `AutoML` and `HyperDrive` approaches the data used is upload to a datastore from where it is accessed by the models. 


## Automated ML

The task for our AutoML model is `forecasting`. Therefore, it is necessary to set `forecasting_parameters`, i.e.,

* `time_column_name` = 'date' (The name of your time column.)
* `forecast_horizon` = 28 (How many days forward we would like to forecast.)
* `time_series_id_column_names` = ['item_id','store_id'] (Names used to uniquely identify the time series in data that has multiple rows with the same timestamp)

In the case of forecasting AutoML tries two types of time-series models:

- Classical statistical models as Arima and Prophet
- Machine Learning regression models

In the case of the statistical ones AutoML loops over all time-series in your dataset and trains one model for each series. This can result in long runtimes to train these models if there are a lot of series in the data. We can mitigate it by making multiple compute cores available. That is the reason why we defined `max_cores_per_iteration=-1`.

Since we have a limited time to run all experiments, we set `experiment_timeout_minutes=30`.

The primary metric was set to `normalized_root_mean_squared_error` considering that this metric is usually used in cases of [forecasting demand]( https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train#primary-metrics-for-time-series-forecasting-scenarios).

We were planning to convert our model to ONNX. However, it is not possible in the case of forecasting.

### Results

The best model obtained by using AutoML was `VotingEmsemble` with `NMRSE=0.1106`.

In the image bellow you can observe the progress of the training runs of different experiments.

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/automl_widget_01.JPG)
![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/automl_widget_02.JPG)

Within the notebook, you can already obtain detail about the best model (See following image). 

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/automl_best_model_notebook.JPG)

Then in Azure ML Studio you can access the top run and also details details of the best model obtained.

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/models_automl.JPG)
![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/best_model_details_01.JPG)
![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/best_model_details_02.JPG)


To try to improve this result we could, for instances:

* Increase `experiment_timeout_minutes` to give more time for AutoML to try other models.
* Make use of [`FeaturizationConfig`](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features), for example, to use other form of imputation of Nan values than the one chosen by AutoML.

## Hyperparameter Tuning

Classical models used for forecasting are statistical models such as `Arima` and `Prophet`. In this experiment I wanted to try a Machine Learning algorithm. I have chosen [`Light GBM (LGBM)`]( https://lightgbm.readthedocs.io/en/latest/index.html) for its great performance on different kind of tasks being, for instance, one of the most used algorithms in [Kaggle]( https://www.kaggle.com/) competitions.

Usually, this implementation gradient boosting algorithm shows better performance than other gradient boost and ensemble algorithms like for example [XGBoost]( https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/).
LGBM presents usually best accuracy and high speed.

The ranges of parameters for the LGBM used were chosen considering the parameters tuning guides for different scenarios provided [here]( https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html).

`Bayesian sampling` method was chosen because tries to intelligently pick the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric. This sampling method does not support `terminantion_policy`. Therefore, `policy=None`.

For Bayesian Sampling it is recommend using a `maximum number of runs` greater than or equal to 20 times the number of hyperparameters being tuned. The recommended value is 140. We set the maximum number of child runs of HyperDrive `max_total_runs` to `20` to reduce the running time.

In order to compare the performance of HyperDrive with the one of AutoML we chose as [objective metric]( https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective) of LGBM `root_mean_squared_root` and we used the fact that `normalized_root_mean_squared_error` is the root_mean_squared_error divided by the range of the data. For more information check this [link]( https://docs.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml#metric-normalization).


### Results

The best LGBM model obtained by using hyperparameter tunning using HyperDrive achieved NRMSE: 0.1430 which is worse than the result 
obtained by the AutoML model.

The image bellow shows the progress of the training runs of different experiments.

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/hd_widget_01.JPG)
![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/hd_widget_02.JPG)
![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/hd_widget_03.JPG)

The hyperparameters of the LGBM model with best result is shown below.

First, you can see already in the notebook which one is the best model, details about it, and its parameters.

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/hd_best_model_notebook.JPG)

But you can also check within ML Studio, the top runs and details about the best model.

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/hd_top_runs.JPG)
![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/hd_best_model_01.JPG)
![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/hd_best_model_02.JPG)

To try to improve this result we could increase `max_total_runs`.


## Model Deployment

### AutoML (Best model compared with HyperDrive model)

`Deployment` is about delivering a trained model into production so that it can be consumed by others.

Comparing the results of AutoML and HyperDrive we saw that AutoML gave us the best model (lower NMRSE). Therefore, this is the model to be deployed.

Details of the deployment of the AutoML model can be seen in section `Model Deployment` of the [AutoML notebook]( https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/automl-final-version-070221.ipynb).

For the deployment we need a function ([score.py]()) which will run the forecast on serialized data. It can be obtained from the `best_run`.

[](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/obtain_score.JPG)

In addition, it is necessary to an `InferenceConfig` and a container instance including in its parameters the necessary numbers of `cpu_cores` and `memory_gb` suitable for the application considered.
Once everything is defined the model can be deployed.

In order to test the deployed model, we have sent a request using the test dataset that consisted of data from `2016-04-25` till `2016-05-22`, i.e., our test dataset. Check [AutoML notebook]( https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/automl-final-version-070221.ipynb) for results.

Here we can see that the model is active.

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/model_deployed_notebook_01.JPG)

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/model_deployed_notebook_02.JPG)

And also in the Azure ML Studio:

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/model_deployed_autoML_studio_01.JPG)
![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/model_deployed_autoML_studio_02.JPG)


### HyperDrive 

Although, this was not the best model we decided to include in the [HyperDrive notebook]() the implementation of the model deployment since it is a bit different then the way of implementing for an AutoML model.

Scoring (score.py) and environment files are provided as part of the AutoML files. For HyperDrive, this is not the same. We must write the scoring file for our HyperDrive experiment. Details on how to do it can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=azcli#define-an-entry-script). Our scoring file for the hyper drive is [here](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/score_hd.py).

More details about the deployment of an HyperDrive Experiment can be seen in section `Model Deployment` of [HyperDrive notebook]( https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/hyperparameter-tuning-final-version-070221.ipynb)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

# Suggestions for future work

- **Use Multi-Round Forecasting**:

I've employed `Single-Round Forecasting`, i.e., I've split the data in train and test once. The use of `Multi-Round Forecasting` means to split the data multiple times in non-overlaping time intervals. This allows us to evaluate the forecasting model on multiple rounds of data, and get a more robust estimate of our model's performance.

- **Use complete dataset**: 

Because of time restriction a subset of the available dataset was used. It would be interesting to use all dataset and the code presented here are ready for it. This would probably improve results because of the greater amount of data made available for training the model.

- **Change parameters of AutoML**

    This could include as mentioned earlier:
    
    * Increase `experiment_timeout_minutes` to give more time for AutoML to try other models.
    * Make use of [`FeaturizationConfig`](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features), for example, to use other form of imputation of Nan values than the one chosen by AutoML.
    
- **Increase `max_total_runs` in HyperDrive**:

Try HyperDrive with higher value of `max_total_runs` to see if the performance increases.

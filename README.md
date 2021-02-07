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

After preprocessing the data (cleaning and creating features) it was split as shown in the image below. The `Training` part was used for training and validation while the `Testing` was used to test the model. For our experiments we used `GAP=0`. Details about `Splitting Time Dataset` can be found in the [Automated ML notebook](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/automl-final-version-070221.ipynb).

![](https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/blob/master/images/time_series_split.JPG)

### Access

The original dataset was downloaded from [Kaggle]( https://www.kaggle.com/c/m5-forecasting-accuracy/data). The subset created from this data was made available at 
[GitHub]( https://github.com/dpbac/Forecasting-Walmart-sales-with-Azure/tree/master/data). 

The GitHub address of the raw data is used to load the data in the Azure workspace. 

For both `AutoML` and `HyperDrive` approaches the data used is upload to a datastore from where it is accessed by the models. 


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

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




## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

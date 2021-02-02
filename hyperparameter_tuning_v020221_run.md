# Hyperparameter Tuning using HyperDrive

TODO: Import Dependencies. In the cell below, import all the dependencies that you will need to complete the project.


```python
import os
import sys
import azureml
import pandas as pd
# import numpy as np
# import logging
import joblib
# import json

from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.estimator import Estimator

from azureml.widgets import RunDetails
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import BayesianParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform, quniform, choice

from azureml.core.model import Model

from azureml.core.webservice import AciWebservice
from azureml.core.model import Model, InferenceConfig


# onnx

from azureml.automl.runtime.onnx_convert import OnnxConverter
from azureml.automl.core.onnx_convert import OnnxConvertConstants
from azureml.train.automl import constants
import onnxruntime
from azureml.automl.runtime.onnx_convert import OnnxInferenceHelper

import warnings
warnings.filterwarnings("ignore")

```

# Initialize workspace and create an Azure ML experiment

To start we need to initialize our workspace and create a Azule ML experiment. It is also to remember that accessing the Azure ML workspace requires authentication with Azure.

Make sure the config file is present at `.\config.json`. This file can be downloaded from home of Azure Machine Learning Studio.


```python
#Define the workspace
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
```

    quick-starts-ws-136926
    aml-quickstarts-136926
    southcentralus
    d7f39349-a66b-446e-aba6-0053c2cf1c11



```python
#Create an experiment
experiment_name = 'hyper-lgbm-walmart-forecasting'
experiment = Experiment(ws, experiment_name)
experiment
```




<table style="width:100%"><tr><th>Name</th><th>Workspace</th><th>Report Page</th><th>Docs Page</th></tr><tr><td>hyper-lgbm-walmart-forecasting</td><td>quick-starts-ws-136926</td><td><a href="https://ml.azure.com/experiments/hyper-lgbm-walmart-forecasting?wsid=/subscriptions/d7f39349-a66b-446e-aba6-0053c2cf1c11/resourcegroups/aml-quickstarts-136926/workspaces/quick-starts-ws-136926" target="_blank" rel="noopener">Link to Azure Machine Learning studio</a></td><td><a href="https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.experiment.Experiment?view=azure-ml-py" target="_blank" rel="noopener">Link to Documentation</a></td></tr></table>




```python
dic_data = {'Workspace name': ws.name,
            'Azure region': ws.location,
            'Subscription id': ws.subscription_id,
            'Resource group': ws.resource_group,
            'Experiment Name': experiment.name}

df_data = pd.DataFrame.from_dict(data = dic_data, orient='index')

df_data.rename(columns={0:''}, inplace = True)
df_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Workspace name</th>
      <td>quick-starts-ws-136926</td>
    </tr>
    <tr>
      <th>Azure region</th>
      <td>southcentralus</td>
    </tr>
    <tr>
      <th>Subscription id</th>
      <td>d7f39349-a66b-446e-aba6-0053c2cf1c11</td>
    </tr>
    <tr>
      <th>Resource group</th>
      <td>aml-quickstarts-136926</td>
    </tr>
    <tr>
      <th>Experiment Name</th>
      <td>hyper-lgbm-walmart-forecasting</td>
    </tr>
  </tbody>
</table>
</div>



# Create or Attach an AmlCompute cluster


```python
# Define CPU cluster name
compute_target_name = "cpu-cluster"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_target_name)
    print("Found existing cpu-cluster. Use it.")
except ComputeTargetException:
    # Specify the configuration for the new cluster
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_DS12_V2",
                                                           min_nodes=1, # when innactive
                                                           max_nodes=4) # when busy
    # Create the cluster with the specified name and configuration
    compute_target = ComputeTarget.create(ws, compute_target_name, compute_config)

compute_target.wait_for_completion(show_output=True)

# For a more detailed view of current AmlCompute status, use get_status()
print(compute_target.get_status().serialize())
```

    Found existing cpu-cluster. Use it.
    
    Running
    {'errors': [], 'creationTime': '2021-02-02T09:05:34.012412+00:00', 'createdBy': {'userObjectId': '0689ef1e-123e-4954-9919-2e6044ec6fd0', 'userTenantId': '660b3398-b80e-49d2-bc5b-ac1dc93b5254', 'userName': None}, 'modifiedTime': '2021-02-02T09:08:50.132901+00:00', 'state': 'Running', 'vmSize': 'STANDARD_DS12_V2'}


# Dataset

TODO: Get data. In the cell below, write code to access the data you will be using in this project. Remember that the dataset needs to be external.

## Overview

The dataset used in this project is a small subset of a much bigger dataset made available at Kaggle's competition [M5 Forecasting - Accuracy Estimate the unit sales of Walmart retail goods](https://www.kaggle.com/c/m5-forecasting-accuracy/overview/description).

The complete dataset covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. **The task is to forecast daily sales for the next 28 days.**

In order to demonstrate the use of Azure ML in forecasting we used the available data consisting of the following files and create a reduced dataset with **10 products of the 3 Texas stores of Walmart**. 

* **calendar.csv** - Contains information about the dates on which the products are sold.
* **sell_prices.csv** - Contains information about the price of the products sold per store and date.
* **sales_train_evaluation.csv** - Includes sales [d_1 - d_1941] (labels used for the Public leaderboard)

Details on how the new dataset was created can be seen in notebook [01-walmart_data_preparation](http://localhost:8888/notebooks/Capstone%20Project/notebooks/01-walmart_data_preparation.ipynb).



```python
time_column_name = 'date'
# data = pd.read_csv("./data/walmart_tx_stores_10_items_with_day.csv",parse_dates=[time_column_name])
data = pd.read_csv("https://raw.githubusercontent.com/dpbac/Forecasting-Walmart-sales-with-Azure/master/data/walmart_tx_stores_10_items_with_day.csv", parse_dates=[time_column_name])
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>item_id</th>
      <th>dept_id</th>
      <th>cat_id</th>
      <th>store_id</th>
      <th>state_id</th>
      <th>day</th>
      <th>demand</th>
      <th>date</th>
      <th>wm_yr_wk</th>
      <th>event_name_1</th>
      <th>event_type_1</th>
      <th>event_name_2</th>
      <th>event_type_2</th>
      <th>snap_TX</th>
      <th>sell_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HOBBIES_2_001_TX_1_evaluation</td>
      <td>HOBBIES_2_001</td>
      <td>HOBBIES_2</td>
      <td>HOBBIES</td>
      <td>TX_1</td>
      <td>TX</td>
      <td>d_1</td>
      <td>0</td>
      <td>2011-01-29</td>
      <td>11101</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>HOBBIES_2_002_TX_1_evaluation</td>
      <td>HOBBIES_2_002</td>
      <td>HOBBIES_2</td>
      <td>HOBBIES</td>
      <td>TX_1</td>
      <td>TX</td>
      <td>d_1</td>
      <td>0</td>
      <td>2011-01-29</td>
      <td>11101</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1.97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HOBBIES_2_003_TX_1_evaluation</td>
      <td>HOBBIES_2_003</td>
      <td>HOBBIES_2</td>
      <td>HOBBIES</td>
      <td>TX_1</td>
      <td>TX</td>
      <td>d_1</td>
      <td>0</td>
      <td>2011-01-29</td>
      <td>11101</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HOBBIES_2_004_TX_1_evaluation</td>
      <td>HOBBIES_2_004</td>
      <td>HOBBIES_2</td>
      <td>HOBBIES</td>
      <td>TX_1</td>
      <td>TX</td>
      <td>d_1</td>
      <td>0</td>
      <td>2011-01-29</td>
      <td>11101</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>nan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HOBBIES_2_005_TX_1_evaluation</td>
      <td>HOBBIES_2_005</td>
      <td>HOBBIES_2</td>
      <td>HOBBIES</td>
      <td>TX_1</td>
      <td>TX</td>
      <td>d_1</td>
      <td>0</td>
      <td>2011-01-29</td>
      <td>11101</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>nan</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 58230 entries, 0 to 58229
    Data columns (total 16 columns):
    id              58230 non-null object
    item_id         58230 non-null object
    dept_id         58230 non-null object
    cat_id          58230 non-null object
    store_id        58230 non-null object
    state_id        58230 non-null object
    day             58230 non-null object
    demand          58230 non-null int64
    date            58230 non-null datetime64[ns]
    wm_yr_wk        58230 non-null int64
    event_name_1    4740 non-null object
    event_type_1    4740 non-null object
    event_name_2    120 non-null object
    event_type_2    120 non-null object
    snap_TX         58230 non-null int64
    sell_price      52938 non-null float64
    dtypes: datetime64[ns](1), float64(1), int64(3), object(11)
    memory usage: 7.1+ MB


## Hyperdrive Configuration

TODO: Explain the model you are using and the reason for chosing the different hyperparameters, termination policy and config settings.

**REVIEW AND EDIT**

Now we are ready to tune hyperparameters of the LightGBM forecast model by launching multiple runs on the cluster. In the following cell, we define the configuration of a HyperDrive job that does a parallel search of the hyperparameter space using a Bayesian sampling method. HyperDrive also supports random sampling of the parameter space.

It is recommended that the maximum number of runs should be greater than or equal to 20 times the number of hyperparameters being tuned, for best results with Bayesian sampling. Specifically, it should be no less than 180 in the following case as we have 9 hyperparameters to tune. Nevertheless, we find that even with a very small amount of runs Bayesian search can achieve decent performance. Thus, the maximum number of child runs of HyperDrive `max_total_runs` is set as `20` to reduce the running time.


```python

# Early Stop Policy


# Specify hyperparameter space
param_sampling = BayesianParameterSampling(
    {
        "--num_leaves": quniform(8, 128, 1),
        "--min_data_in_leaf": quniform(20, 500, 10),
        "--learning_rate": choice(1e-4, 1e-3, 5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 1e-1),
        "--feature_fraction": uniform(0.2, 1),
        "--bagging_fraction": uniform(0.1, 1),
        "--bagging_freq": quniform(1, 20, 1),
        "--max_rounds": quniform(50, 2000, 10),
    }
)

# Create an estimator for use with train.py

est = Estimator(
    source_directory='./', # directory containing experiment configuration files (train.py)
    compute_target=compute_target, # compute target where training will happen
    vm_size="STANDARD_DS12_V2", # VM size of the compute target
    entry_script='train.py')

   
# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.

hyperdrive_config = HyperDriveConfig(
    estimator=est,
    hyperparameter_sampling=param_sampling,
    primary_metric_name='MAE',# mean_absolute_error
    primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
    max_total_runs=140, # Bayesian Sampling we recommend using a maximum number of runs greater than or equal to 20 times the number of hyperparameters being tuned. Recommendend value:140
    max_concurrent_runs=4,
    policy=None, #Bayesian sampling does not support early termination policies.
)

```

    'Estimator' is deprecated. Please use 'ScriptRunConfig' from 'azureml.core.script_run_config' with your own defined environment or an Azure ML curated environment.



```python
# # TODO: Create an early termination policy. This is not required if you are using Bayesian sampling.
# early_termination_policy = <your policy here>

# #TODO: Create the different params that you will be using during training
# param_sampling = <your params here>

# #TODO: Create your estimator and hyperdrive config
# estimator = <your estimator here>

# hyperdrive_run_config = <your config here?
```


```python
# Submit hyperdrive run to the experiment 

hyperdrive_run = experiment.submit(config = hyperdrive_config)
```

    WARNING:root:If 'script' has been provided here and a script file name has been specified in 'run_config', 'script' provided in ScriptRunConfig initialization will take precedence.


## Run Details

OPTIONAL: Write about the different models trained and their performance. Why do you think some models did better than others?

TODO: In the cell below, use the `RunDetails` widget to show the different experiments.


```python
# Show run details with the Jupyter widget

RunDetails(hyperdrive_run).show()
hyperdrive_run.wait_for_completion(show_output=True)
hyperdrive_run.get_metrics()
```


    _HyperDriveWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO'…




    RunId: HD_de8ddeb9-5a7a-400d-a575-d647de6ec74c
    Web View: https://ml.azure.com/experiments/hyper-lgbm-walmart-forecasting/runs/HD_de8ddeb9-5a7a-400d-a575-d647de6ec74c?wsid=/subscriptions/d7f39349-a66b-446e-aba6-0053c2cf1c11/resourcegroups/aml-quickstarts-136926/workspaces/quick-starts-ws-136926
    
    Streaming azureml-logs/hyperdrive.txt
    =====================================
    
    "<START>[2021-02-02T09:19:00.466959][API][INFO]Experiment created<END>\n""<START>[2021-02-02T09:19:01.111832][GENERATOR][INFO]Trying to sample '4' jobs from the hyperparameter space<END>\n""<START>[2021-02-02T09:19:01.659204][GENERATOR][INFO]Successfully sampled '4' jobs, they will soon be submitted to the execution target.<END>\n"<START>[2021-02-02T09:19:02.2262685Z][SCHEDULER][INFO]The execution environment is being prepared. Please be patient as it can take a few minutes.<END>
    
    Execution Summary
    =================
    RunId: HD_de8ddeb9-5a7a-400d-a575-d647de6ec74c
    Web View: https://ml.azure.com/experiments/hyper-lgbm-walmart-forecasting/runs/HD_de8ddeb9-5a7a-400d-a575-d647de6ec74c?wsid=/subscriptions/d7f39349-a66b-446e-aba6-0053c2cf1c11/resourcegroups/aml-quickstarts-136926/workspaces/quick-starts-ws-136926
    





    {}



## Retrieve and Save Best Model

TODO: In the cell below, get the best model from the hyperdrive experiments and display all the properties of the model.


```python
# Get your best run and save the model from that run.

best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()

print('Best Run Id: ', best_run.id)
print('NMAE:', best_run_metrics['normalized_mean_absolute_error'])

best_run
```


```python
parameter_values = best_run.get_details()["runDefinition"]["arguments"]
print(parameter_values)
```


```python
# Save the best model
model = best_run.register_model(
    model_name="hd_lgbm_walmart_forecast", 
    model_path="./outputs/model.pkl",
    description='Best HyperDrive Walmart forecasting model'
)
print("Model successfully saved.")
```

# ONNX model

## Retrieve and save the best ONNX model


```python
#Retrieve and save the best model

best_run, onnx_model = hyperdrive_run.get_output(return_onnx_model=True)
onnx_model_path = "results/best_model.onnx"
OnnxConverter.save_onnx_model(onnx_model, onnx_model_path)
```

## Predict with the ONNX model


```python
if sys.version_info < OnnxConvertConstants.OnnxIncompatiblePythonVersion:
    python_version_compatible = True
else:
    python_version_compatible = False

def get_onnx_res(run):
    res_path = 'onnx_resource.json'
    run.download_file(name=constants.MODEL_RESOURCE_PATH_ONNX, output_file_path=res_path)
    with open(res_path) as f:
        onnx_res = json.load(f)
    return onnx_res

if python_version_compatible:
    test_df = test_data.to_pandas_dataframe()
    mdl_bytes = onnx_mdl.SerializeToString()
    onnx_res = get_onnx_res(best_run)

    onnxrt_helper = OnnxInferenceHelper(mdl_bytes, onnx_res)
    pred_onnx, pred_prob_onnx = onnxrt_helper.predict(test_df)

    print(pred_onnx)
    print(pred_prob_onnx)
else:
    print('Use Python version 3.6 or 3.7 to run the inference helper.')
```

## Model Deployment

**REVIEW AND EDIT**

Remember you have to deploy only one of the two models you trained.. Perform the steps in the rest of this notebook only if you wish to deploy this model.

TODO: In the cell below, register the model, create an inference config and deploy the model as a web service.

Now we are ready to deploy the model as a web service running in Azure Container Instance [ACI](https://azure.microsoft.com/en-us/services/container-instances/). Azure Machine Learning accomplishes this by constructing a Docker image with the scoring logic and model baked in.

### Create score.py

First, we will create a scoring script that will be invoked by the web service call.

* Note that the scoring script must have two required functions, `init()` and `run(input_data)`.
    - In `init()` function, you typically load the model into a global object. This function is executed only once when the Docker container is started.
    - In `run(input_data)` function, the model is used to predict a value based on the input data. The input and output to run typically use JSON as serialization and de-serialization format but you are not limited to that.


```python
%%writefile score.py
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb


def init():
    global bst
    model_root = os.getenv("AZUREML_MODEL_DIR")
    # The name of the folder in which to look for LightGBM model files
    lgbm_model_folder = "model"
    bst = lgb.Booster(
        model_file=os.path.join(model_root, lgbm_model_folder, "best-model.txt")
    )


def run(raw_data):
    columns = bst.feature_name()
    data = np.array(json.loads(raw_data)["data"])
    test_df = pd.DataFrame(data=data, columns=columns)
    # Make prediction
    out = bst.predict(test_df)
    return out.tolist()
```

### Create myenv.yml

We also need to create an environment file so that Azure Machine Learning can install the necessary packages in the Docker image which are required by your scoring script. In this case, we need to specify packages `numpy`, `pandas`, and `lightgbm`.


```python
print(pd.__version__)
print(np.__version__)
print(lgb.__version__)
```


```python
cd = CondaDependencies.create()
cd.add_conda_package("numpy=1.16.2")
cd.add_conda_package("pandas=0.23.4")
cd.add_conda_package("lightgbm=2.3.0")
cd.save_to_file(base_directory="./", conda_file_path="myenv.yml")

print(cd.serialize_to_string())
```

### Deploy to ACI

We are almost ready to deploy. In the next cell, we first create the inference configuration and deployment configuration. Then, we deploy the model to ACI. This cell will run for several minutes.


```python
inference_config = InferenceConfig(environment = best_run.get_environment(), 
                                   entry_script = script_file_name)

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 2, 
                                               auth_enabled=True, 
                                               enable_app_insights=True,
                                               tags = {'type': "automl-forecasting"},
                                               description = "Automl forecasting sample service")

aci_service_name = 'automl-walmart-forecast-01'
print(aci_service_name)
aci_service = Model.deploy(ws, aci_service_name, [model], inference_config, aciconfig)
aci_service.wait_for_deployment(True)
print(aci_service.state)
```


```python
%%time

inference_config = InferenceConfig(runtime="python", 
                                   entry_script="score.py", 
                                   conda_file="myenv.yml")

aciconfig = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=2,
    auth_enabled=True, 
    enable_app_insights=True,
    tags={"name": "Walmart_data", "framework": "LightGBM"},
    description="LightGBM model on Walmart dataset",
)

aci_service_name = 'hyperdrive-walmart-forecast-01'
print(aci_service_name)

aci_service = Model.deploy(workspace=ws, 
                       name=aci_service_name, 
                       models=[model], 
                       inference_config=inference_config, 
                       deployment_config=aciconfig
)

aci_service.wait_for_deployment(True)
print(aci_service.state)
```


```python
print("Scoring web service endpoint: {}".format(service.scoring_uri))
```

### Test the deployed model

Let's test the deployed model. We create a few test data points and send them to the web service hosted in ACI. Note here we are using the run API in the SDK to invoke the service. You can also make raw HTTP calls using any HTTP tool such as curl.

After the invocation, we print the returned predictions each of which represents the forecasted sales of a target store, brand in a given week as specified by `store, brand, week` in `used_columns`.

TODO: In the cell below, send a request to the web service you deployed to test it.

TODO: In the cell below, print the logs of the web service and delete the service


```python

```

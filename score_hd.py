import os
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from azureml.core import Model

# def init():
#     global bst
#     model_root = os.getenv("AZUREML_MODEL_DIR")
#     # The name of the folder in which to look for LightGBM model files
#     lgbm_model_folder = "model"
#     bst = lgb.Booster(
#         model_file=os.path.join(model_root, lgbm_model_folder, "bst-model.pkl")
#     )

# def run(raw_data):
#     columns = bst.feature_name()
#     data = np.array(json.loads(raw_data)["data"])
#     test_df = pd.DataFrame(data=data, columns=columns)
#     # Make prediction
#     out = bst.predict(test_df)
#     return out.tolist()


def init():
    global model
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'bst-model.pkl')
    model = joblib.load(model_path)


def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
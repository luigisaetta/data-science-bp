
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

import json
import os
import pickle

import io
import logging 

# logging configuration - OPTIONAL 
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger_pred = logging.getLogger('model-prediction')
logger_pred.setLevel(logging.INFO)
logger_feat = logging.getLogger('input-features')
logger_feat.setLevel(logging.INFO)

model_name = 'model.pkl'

# to enable/disable detailed logging
DEBUG = True

"""
   Inference script. This script is used for prediction by scoring server when schema is known.
"""

def load_model(model_file_name=model_name):
    """
    Loads model from the serialized format

    Returns
    -------
    model:  a model instance on which predict API can be invoked
    """
    
    model_dir = os.path.dirname(os.path.realpath(__file__))
    contents = os.listdir(model_dir)
    
    # Load the model from the model_dir using the appropriate loader
    
    if model_file_name in contents:
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), model_file_name), "rb") as file:
            model = pickle.load(file) 
            logger_pred.info("Loaded the model !!!")
                
    else:
        raise Exception('{0} is not found in model directory {1}'.format(model_file_name, model_dir))
    
    return model

def pre_inference(data):
    """
    Preprocess data

    Parameters
    ----------
    data: Data format as expected by the predict API of the core estimator.

    Returns
    -------
    data: Data format after any processing.

    """
    logger_pred.info("Preprocessing...")
    
    return data

def post_inference(yhat):
    """
    Post-process the model results

    Parameters
    ----------
    yhat: Data format after calling model.predict.

    Returns
    -------
    yhat: Data format after any processing.

    """
    logger_pred.info("Postprocessing output...")
    
    return yhat

def predict(data, model=load_model()):
    """
    Returns prediction given the model and data to predict

    Parameters
    ----------
    model: Model instance returned by load_model API
    data: Data format as expected by the predict API of the core estimator. For eg. in case of sckit models it could be numpy array/List of list/Pandas DataFrame

    Returns
    -------
    predictions: Output from scoring server
        Format: {'prediction': output from model.predict method}

    """
    
    logger_pred.info("In function predict...")
    
    # some check
    assert model is not None, "Model is not loaded"
    
    x = pd.read_json(io.StringIO(data)).values
    
    if DEBUG:
        logger_feat.info("Logging features")
        logger_feat.info(x)
    
    # preprocess data (for example normalize features)
    x = pre_inference(x)

    logger_pred.info("Invoking model......")
    
    # compute predictions (binary, from model)
    preds = model.predict(x)
    
    # to avoid not JSON serialiable error (np.array is not)
    preds = preds.tolist()
    
    # post inference not needed
    return {'prediction': preds}

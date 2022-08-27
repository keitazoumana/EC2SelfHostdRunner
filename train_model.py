import json
import pandas as pd
import random
from tqdm import tqdm
import os
import yaml

import mlflow

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer,  AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch

if __name__ == "__main__":


    mlflow_config = yaml.safe_load(open("config.yaml"))["credentials"]["mlflow_credentials"]
    metrics_file = "./metrics/metrics.json"

    MLFLOW_TRACKING_URI= mlflow_config['MLFLOW_TRACKING_URI']
    MLFLOW_TRACKING_USERNAME = mlflow_config['MLFLOW_TRACKING_USERNAME']
    MLFLOW_TRACKING_PASSWORD = mlflow_config['MLFLOW_TRACKING_PASSWORD'] 

    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run():

        # Load the metrics file
        with open(metrics_file) as metrics_info:

            metrics_values = json.load(metrics_info)

            # Track them with MLFlow
            mlflow.log_metric("Precision", metrics_values['precision'])
            mlflow.log_metric("Recall", metrics_values['recall'])
            mlflow.log_metric("F1-Score", metrics_values['f1-score'])

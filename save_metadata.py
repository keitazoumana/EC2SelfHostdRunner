from yaml import safe_load 
import os

# Get The Credentials 
credentials = safe_load(open("config.yaml"))["credentials"]

# DVC Credentials
dvc_config = credentials["dvc_config"]
DVC_REMOTE_URL = dvc_config["DAGSHUB_REMOTE_URL"]
USERNAME = dvc_config["USERNAME"]
PASSWORD = dvc_config["PASSWORD"]

# Metadata values
meta_data = credentials["metadata_path"]

METRICS = meta_data["METRICS_PATH"]
MODEL = meta_data["MODEL_PATH"]

# Configure DVC
os.system("dvc remote add origin {DVC_REMOTE_URL}")
os.system("dvc remote modify origin --local auth basic")
os.system("dvc remote modify origin --local user {USERNAME}}")
os.system("dvc remote modify origin --local password {PASSWORD}")

# Add Metadata to DVC
os.system("dvc add {METRICS} {MODEL}")
os.system("dvc push")
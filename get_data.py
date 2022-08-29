# Set Authentication
from yaml import safe_load 
import os

# Get The Credentials 
credentials = safe_load(open("config.yaml"))["credentials"]

# DVC Credentials
dvc_config = credentials["dvc_config"]
DVC_REMOTE_URL = dvc_config["DAGSHUB_REMOTE_URL"]
USERNAME = dvc_config["USERNAME"]
PASSWORD = dvc_config["PASSWORD"]

# Get Raw Data path for pulling
meta_data = credentials["metadata_path"]
RAW_DATA_PATH = meta_data["RAW_DATA_PATH"]

os.system("dvc remote add origin {}".format(DVC_REMOTE_URL))
os.system("dvc remote modify origin --local auth basic")
os.system("dvc remote modify origin --local user {}".format(USERNAME))
os.system("dvc remote modify origin --local password {}".format(PASSWORD))

# Get data from DVC
os.system("dvc pull -r origin {}".format(RAW_DATA_PATH))

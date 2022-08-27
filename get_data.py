import os

# Get data from DVC using the raw data folder
os.system("dvc pull -r origin data/raw")

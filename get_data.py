import os

# Get data from DVC
os.system("dvc pull -r origin data/raw")
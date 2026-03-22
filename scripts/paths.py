from pathlib import Path
import os

# If this script is in /project/scripts/script.py, .parent.parent gets /project/
root = Path(__file__).resolve().parent.parent
#root = "/home/projects/s215045/PredictPhagePPI/"
raw_data_path = os.path.join(root, "raw_data/")
data_prod_path = os.path.join(root, "data_prod/")

path_to_nn_runs = os.path.join(root, "nn_runs/")
import json
import shutil
from pathlib import Path


##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = Path(config["output_folder_path"])
output_model_path = Path(config["output_model_path"])

prod_deployment_path = Path(config["prod_deployment_path"])
prod_deployment_path.mkdir(exist_ok=True)

ingested_filename = "ingestedfiles.csv"
score_filename = "latestscore.txt"
model_filename = "trainedmodel.pkl"

ingested_path = Path(dataset_csv_path, ingested_filename)
score_path = Path(output_model_path, score_filename)
model_path = Path(output_model_path, model_filename)

####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy(ingested_path, prod_deployment_path)
    shutil.copy(score_path, prod_deployment_path)
    shutil.copy(model_path, prod_deployment_path)


if __name__ == "__main__":
    store_model_into_pickle()

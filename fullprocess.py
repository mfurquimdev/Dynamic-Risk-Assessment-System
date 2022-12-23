import json
from pathlib import Path

import apicalls
import deployment
import diagnostics
import ingestion
import reporting
import scoring
import training

##################Check and read new data
# first, read ingestedfiles.txt
# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here

print("Checking for new data")
if not ingestion.is_there_new_data():
    print("No new data found!\nExiting...")
    exit(0)

print("\nThere's new data!\nMergin all CSV into a DataFrame")
ingestion.merge_multiple_dataframe()

##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here

print("\nChecking if model has drifted")
if not scoring.has_model_drifted():
    print("Model has not drifted!\nExiting...")
    exit(0)


##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_directory = config["output_folder_path"]
dataset_filename = config["dataset_filename"]
output_model_dir = Path(config["output_model_path"])

print("Model has drifted!\nTraining a new model with the new data")
training.preprocess_and_train()

print("\nScoring current model")
scoring.score_model(dataset_csv_directory, dataset_filename)

print("\nDeploying new model")
deployment.store_model_into_pickle()


##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
print("\n\nRunning diagnostics and reporting")

print(f"\nRunning diagnostics with data at {Path(dataset_csv_directory,dataset_filename)}")
percentage_missing, dataframe_summary = diagnostics.run_diagnostics(dataset_csv_directory, dataset_filename)
print(f"Percentage missing: {percentage_missing}")
print(f"Dataframe summary: {dataframe_summary}")


print("\nTiming execution of ingestion.py and training.py")
executing_time = diagnostics.execution_time()
print(f"Executing time: {executing_time }")

print("\nGetting package list")
package_report = diagnostics.outdated_packages_list()
print(f"package_report: {package_report}")


print(f"\nPlotting confusion matrix 2 with data at {Path(dataset_csv_directory, dataset_filename)}")
confusion_matrix_filename = "confusionmatrix2.png"
reporting.plot_confusion_matrix(
    data_dir=dataset_csv_directory,
    data_filename=dataset_filename,
    confusion_matrix_filename=confusion_matrix_filename,
)
print(f"Confusion Matrix 2 was saved at {Path(output_model_dir, confusion_matrix_filename)}")


print("\nMaking apicalls")
api_returns_filename = "apireturns2.txt"
apicalls.make_calls(api_returns_filename=api_returns_filename)
print(f"API returns were saved at {Path(output_model_dir, api_returns_filename)}")

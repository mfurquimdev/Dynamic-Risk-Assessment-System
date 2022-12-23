import json
import os
import pickle
import subprocess
import timeit
from pathlib import Path

import numpy as np

from preprocess import preprocess_data

##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)


dataset_csv_path = Path(config["output_folder_path"])
test_data_dir = Path(config["test_data_path"])
prod_deployment_path = Path(config["prod_deployment_path"])
test_filename = config["test_filename"]

##################Function to get model predictions
def model_predictions(X):
    # read the deployed model and a test dataset, calculate predictions
    model_filename = config["model_filename"]
    model = pickle.load(open(Path(prod_deployment_path, model_filename), "rb"))
    y_pred = model.predict(X)

    return y_pred  # return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary(X):
    # calculate summary statistics here
    summary_filename = config["summary_filename"]
    summary_path = Path(dataset_csv_path, summary_filename)

    summary_statistics = []
    for col in X.columns:
        mean = np.mean(X[col])
        median = np.median(X[col])
        stdev = np.std(X[col])
        summary_statistics.append([mean, median, stdev])

    X.describe().to_csv(summary_path)

    return summary_statistics  # return value should be a list containing all summary statistics


def check_missing_data(X):
    nas = list(X.isna().sum())
    na_percents = [nas[i] / len(X.index) for i in range(len(nas))]
    return na_percents


##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    execution_time_list = []

    start_time = timeit.default_timer()
    os.system("python3 ingestion.py")
    execution_time_list.append(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    os.system("python3 training.py")
    execution_time_list.append(timeit.default_timer() - start_time)

    return execution_time_list  # return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list():
    broken = subprocess.check_output(["pip", "check"])
    with open(config["broken_filename"], "wb") as fd:
        fd.write(broken)

    installed = subprocess.check_output(["pip", "list"])
    with open(config["installed_filename"], "wb") as fd:
        fd.write(installed)

    requirements = subprocess.check_output(["pip", "freeze"])
    with open(config["new_requirements_filename"], "wb") as fd:
        fd.write(requirements)

    report = {
        "broken": broken.decode("utf-8"),
        "installed": installed.decode("utf-8"),
        "requirements": requirements.decode("utf-8"),
    }
    return report


def model_summary_statistics():
    X, y = preprocess_data(test_data_dir, test_filename)

    y_pred = model_predictions(X)

    summary = dataframe_summary(X)

    return summary


def check_missing_test_data():
    X, _ = preprocess_data(test_data_dir, test_filename)
    return check_missing_data(X)


def run_diagnostics(directory, filename):
    X, _ = preprocess_data(directory, filename)
    percentage_missing = check_missing_data(X)
    df_summary = dataframe_summary(X)

    return percentage_missing, df_summary


if __name__ == "__main__":
    X, y = preprocess_data(test_data_dir, test_filename)

    y_pred = model_predictions(X)

    summary = dataframe_summary(X)

    percentage_missing = check_missing_data(X)

    executing_time = execution_time()

    packages_report = outdated_packages_list()

import json
import os
import pickle
import subprocess
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)


dataset_csv_path = Path(config["output_folder_path"])
test_data_dir = Path(config["test_data_path"])
prod_deployment_path = Path(config["prod_deployment_path"])

##################Function to get model predictions
def model_predictions(X):
    # read the deployed model and a test dataset, calculate predictions
    model_filename = "trainedmodel.pkl"
    model = pickle.load(open(Path(prod_deployment_path, model_filename), "rb"))
    y_pred = model.predict(X)

    return y_pred  # return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary(X):
    # calculate summary statistics here
    summary_filename = "summary_statistics.csv"
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
    # get a list of
    broken = subprocess.check_output(["pip", "check"])
    with open("broken.txt", "wb") as fd:
        fd.write(broken)

    installed = subprocess.check_output(["pip", "list"])
    with open("installed.txt", "wb") as fd:
        fd.write(installed)

    requirements = subprocess.check_output(["pip", "freeze"])
    with open("new_requirements.txt", "wb") as fd:
        fd.write(requirements)


def preprocess_data(test_data_path):
    X = pd.read_csv(test_data_path)
    y = X.pop("exited")

    X = X.select_dtypes(include=["number"])

    return X, y


if __name__ == "__main__":
    test_data_path = Path(test_data_dir, "testdata.csv")
    X, y = preprocess_data(test_data_path)

    y_pred = model_predictions(X)

    summary = dataframe_summary(X)

    check_missing_data(X)

    dataframe_summary(X)
    print(execution_time())
    outdated_packages_list()

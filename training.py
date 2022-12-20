import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


###################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = Path(config["output_folder_path"])
output_model_path = Path(config["output_model_path"])
output_model_path.mkdir(exist_ok=True)


def preprocess_data():
    train_filename = "finaldata.csv"
    X = pd.read_csv(Path(dataset_csv_path, train_filename))
    y = X.pop("exited")

    X = X.select_dtypes(include=["number"])

    return X, y


#################Function for training the model
def train_model(X, y):

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    model.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    model_filename = "trainedmodel.pkl"
    with open(Path(output_model_path, model_filename), "wb") as fd:
        pickle.dump(model, fd)


if __name__ == "__main__":
    X, y = preprocess_data()
    train_model(X, y)

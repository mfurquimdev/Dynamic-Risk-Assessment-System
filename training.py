import json
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from preprocess import preprocess_data


###################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = Path(config["output_folder_path"])
output_model_path = Path(config["output_model_path"])
output_model_path.mkdir(exist_ok=True)

dataset_filename = config["dataset_filename"]

model_filename = config["model_filename"]

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
    with open(Path(output_model_path, model_filename), "wb") as fd:
        pickle.dump(model, fd)


def preprocess_and_train():
    X, y = preprocess_data(dataset_csv_path, dataset_filename)
    train_model(X, y)


if __name__ == "__main__":
    preprocess_and_train()

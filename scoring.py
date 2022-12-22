import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn import metrics

#################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = Path(config["output_folder_path"])
output_model_path = Path(config["output_model_path"])
test_data_path = Path(config["test_data_path"])


def preprocess_data():
    test_filename = "testdata.csv"
    X = pd.read_csv(Path(test_data_path, test_filename))
    y = X.pop("exited")

    X = X.select_dtypes(include=["number"])

    return X, y


#################Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    X, y = preprocess_data()

    model_filename = "trainedmodel.pkl"
    model = pickle.load(open(Path(output_model_path, model_filename), "rb"))
    y_pred = model.predict(X)
    f1 = metrics.f1_score(y, y_pred)

    score_filename = "latestscore.txt"
    score_path = Path(output_model_path, score_filename)
    with open(score_path, "w") as fd:
        fd.write(str(f1))

    return f1


if __name__ == "__main__":
    score_model()

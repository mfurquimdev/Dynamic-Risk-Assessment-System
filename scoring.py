import json
import pickle
from pathlib import Path

from sklearn import metrics

from preprocess import preprocess_data

#################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_directory = Path(config["output_folder_path"])
output_model_directory = Path(config["output_model_path"])
test_data_directory = Path(config["test_data_path"])
test_filename = Path(config["test_filename"])
model_filename = Path(config["model_filename"])
score_filename = Path(config["score_filename"])
dataset_filename = Path(config["dataset_filename"])

#################Function for model scoring
def score_model():
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    X, y = preprocess_data(test_data_directory, test_filename)

    model = pickle.load(open(Path(output_model_directory, model_filename), "rb"))
    y_pred = model.predict(X)
    f1 = metrics.f1_score(y, y_pred)

    with open(Path(output_model_directory, score_filename), "w") as fd:
        fd.write(str(f1))

    return f1


def has_model_drifted():
    X, y = preprocess_data(dataset_csv_directory, dataset_filename)

    score_path = Path(output_model_directory, score_filename)

    if not score_path.exists():
        return True

    latest_score = float(score_path.read_text(encoding="utf-8"))
    current_score = score_model()

    print(f"Latest score: {latest_score}\nCurrent score: {current_score}")

    return current_score < latest_score


if __name__ == "__main__":
    score_model()

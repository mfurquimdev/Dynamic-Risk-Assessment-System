import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from diagnostics import model_predictions
from diagnostics import preprocess_data

###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_dir = Path(config["output_folder_path"])
test_data_dir = Path(config["test_data_path"])
output_model_dir = Path(config["output_model_path"])


##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace

    test_data_path = Path(test_data_dir, "testdata.csv")
    X, y = preprocess_data(test_data_path)
    y_pred = model_predictions(X)

    model_confusion_matrix = confusion_matrix(y, y_pred)
    display_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=model_confusion_matrix)
    display_confusion_matrix.plot()
    confusion_matrix_filepath = Path(output_model_dir, "confusionmatrix.png")
    plt.savefig(confusion_matrix_filepath)


if __name__ == "__main__":
    score_model()

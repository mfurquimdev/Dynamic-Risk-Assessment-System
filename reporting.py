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
from preprocess import preprocess_data

###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_dir = Path(config["output_folder_path"])
test_data_dir = Path(config["test_data_path"])
output_model_dir = Path(config["output_model_path"])
test_filename = Path(config["test_filename"])
confusion_matrix_filename = Path(config["confusion_matrix_filename"])

##############Function for reporting
def plot_confusion_matrix(
    data_dir=test_data_dir,
    data_filename=test_filename,
    confusion_matrix_filename=confusion_matrix_filename,
):
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace

    X, y = preprocess_data(data_dir, data_filename)
    y_pred = model_predictions(X)

    model_confusion_matrix = confusion_matrix(y, y_pred)
    display_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix=model_confusion_matrix)
    display_confusion_matrix.plot()
    confusion_matrix_filepath = Path(output_model_dir, confusion_matrix_filename)
    plt.savefig(confusion_matrix_filepath)


if __name__ == "__main__":
    plot_confusion_matrix()

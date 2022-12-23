import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request
from flask import session

from diagnostics import check_missing_test_data
from diagnostics import execution_time
from diagnostics import model_predictions
from diagnostics import model_summary_statistics
from diagnostics import outdated_packages_list
from preprocess import preprocess_data
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    # call the prediction function you created in Step 3
    print(f"{request.method} at {request.path} with {request.args}")

    filename = request.args.get("filename")
    print(f"filename = {filename}")

    X, _ = preprocess_data(*filename.split("/"))
    print(f"X = {X}")

    y_pred = model_predictions(X)
    print(f"y_pred = {y_pred}")

    print(f"return {y_pred}")
    return str(y_pred)  # add return value for prediction outputs


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring():
    # check the score of the deployed model
    print(f"{request.method} at {request.path} with {request.args}")

    f1_score = score_model()

    print(f"return {f1_score}")
    return str(f1_score)  # add return value (a single F1 score number)


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats():
    # check means, medians, and modes for each column
    print(f"{request.method} at {request.path} with {request.args}")

    summary_stats = model_summary_statistics()

    print(f"return {summary_stats}")
    return str(summary_stats)  # return a list of all calculated summary statistics


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    # check timing and percent NA values
    print(f"{request.method} at {request.path} with {request.args}")
    executing_time = execution_time()
    print(f"executing_time {executing_time}")
    percentage_missing = check_missing_test_data()
    print(f"perct_missing {percentage_missing}")
    package_report = outdated_packages_list()
    print(f"package_report {package_report}")

    diagnosis = {
        "executing_time": executing_time,
        "perct_missing": percentage_missing,
        "package_report": package_report,
    }

    return diagnosis  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)

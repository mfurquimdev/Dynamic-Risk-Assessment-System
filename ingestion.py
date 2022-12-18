import datetime
import json
from pathlib import Path

import pandas as pd


#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


#############Function for data ingestion
def merge_multiple_dataframe():
    """
    Check for datasets, compile them together, and write to an output file

    All configuration is done in config.json.
    Data is read from `input_folder_path` and write to `output_folder_path`.
    CSV files are detected automatically.
    Output csv is called `finaldata.csv` and the report is named `ingestedfiles.csv`.
    """
    input_path = Path(input_folder_path)
    input_files = sorted(input_path.rglob("*.csv"))

    output_path = Path(output_folder_path)
    output_path.mkdir(exist_ok=True)

    final_data_filename = "finaldata.csv"
    final_data_path = Path(output_folder_path, final_data_filename)

    records_filename = "ingestedfiles.csv"
    records_path = Path(output_folder_path, records_filename)

    all_records = pd.DataFrame(columns=["source_location", "filename", "data_length", "timestamp"])

    final_df = pd.DataFrame(
        columns=["corporation", "lastmonth_activity", "lastyear_activity", "number_of_employees", "exited"]
    )
    for filepath in input_files:
        datetimestr = datetime.datetime.now().isoformat()

        df = pd.read_csv(filepath)
        record = {
            "source_location": input_path,
            "filename": filepath.name,
            "data_length": len(df.index),
            "timestamp": datetimestr,
        }

        all_records = all_records.append(record, ignore_index=True)
        final_df = final_df.append(df).reset_index(drop=True)

    final_df.drop_duplicates(ignore_index=True, inplace=True)

    all_records.to_csv(records_path, index=False)
    final_df.to_csv(final_data_path, index=False)


if __name__ == "__main__":
    merge_multiple_dataframe()

"""Script to preprocess and return X and y from CSV"""
from pathlib import Path

import pandas as pd


class DataNotFoundError(Exception):
    """Raised when there is no CSV in the given directory"""

    def __init__(self, data_path: Path, message=None):
        """
        Initialise exception setting data_path and message

        Args:
            data_path: the given path to CSV file
            message: message to be displayed on exception
        """
        self.data_path = data_path
        self.set_message(message)
        super().__init__(self._message)

    def set_message(self, message):
        """Set custom message"""
        self._message = message if message else "No data found on given directory"


def preprocess_data(directory, filename, label_column="exited") -> (pd.DataFrame, pd.Series):
    """Filter out categorical columns from X and extract y"""
    data_path = Path(directory, filename)

    if not data_path.exists():
        raise DataNotFoundError(data_path)

    df = pd.read_csv(data_path)

    y = df.pop(label_column) if label_column in df.columns else None

    X = df.select_dtypes(include=["number"])

    return X, y

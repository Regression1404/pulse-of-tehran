import os
import pandas as pd
from fontTools.merge.util import equal

from src.preprocessing import TrafficPreprocessor


def preprocess(file_path):
    """Preprocess a single file and return the cleaned DataFrame."""
    preprocessor = TrafficPreprocessor(file_path)
    preprocessor.preprocess()
    return preprocessor.df


class Aggregator:
    def __init__(self, file_name):
        """
        Initialize the aggregator with a file name.
        """
        self.file_name = file_name
        self.base_path = 'resources'
        self.years = ['1401', '1402', '1403']
        self.months = ["farvardin", "ordibehesht", "khordad", "tir", "mordad", "shahrivar",
                       "mehr", "aban", "azar", "dey", "bahman", "esfand"]
        self.seasons = {"farvardin": "spring", "ordibehesht": "spring", "khordad": "spring", "tir": "summer",
                        "mordad": "summer", "shahrivar": "summer", "mehr": "autumn", "aban": "autumn", "azar": "autumn",
                        "dey": "winter", "bahman": "winter", "esfand": "winter"}

    def aggregate_data(self):
        """Aggregate and preprocess data from multiple files."""
        dataframes = []

        for year in self.years:
            for month in self.months:
                file_path = os.path.join(self.base_path, year, month, self.file_name)

                if os.path.exists(file_path):
                    df = preprocess(file_path)

                    df["year"] = year
                    df["month"] = month
                    df["season"] = self.seasons[month]

                    dataframes.append(df)

        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        else:
            return pd.DataFrame()

    def aggregate_specific_data(self, path="proceed", year=None):
        """Aggregate and preprocess data from multiple files."""
        if year is None:
            year = ['1403']
        dataframes = []

        if path != self.base_path:
            path = os.path.join(self.base_path, path)

        for year in year:
            for month in self.months:
                file_path = os.path.join(path, year, month, self.file_name)

                if os.path.exists(file_path):
                    df = pd.read_excel(file_path)

                    df["year"] = year
                    df["month"] = month
                    df["season"] = self.seasons[month]

                    dataframes.append(df)

        if dataframes:
            return pd.concat(dataframes, ignore_index=True)
        else:
            return pd.DataFrame()

import pandas as pd
import numpy as np
from persiantools.jdatetime import JalaliDateTime


class TrafficPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the preprocessor with a file path.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Load traffic data from an Excel file and process timestamps.
        """
        df = pd.read_excel(self.file_path)

        df.rename(columns={
            df.columns[0]: "axis code",
            df.columns[1]: "axis name",
            df.columns[2]: "start time",
            df.columns[3]: "end time",
            df.columns[4]: "operating time (minutes)",
            df.columns[5]: "total number of vehicles",
            df.columns[6]: "number of Class 1 vehicles",
            df.columns[7]: "number of Class 2 vehicles",
            df.columns[8]: "number of Class 3 vehicles",
            df.columns[9]: "number of Class 4 vehicles",
            df.columns[10]: "number of Class 5 vehicles",
            df.columns[11]: "average speed",
            df.columns[12]: "number of speeding violations",
            df.columns[13]: "number of unauthorized distance violations",
            df.columns[14]: "number of unauthorized overtaking violations",
            df.columns[15]: "estimated number"
        }, inplace=True)

        df = df.iloc[1:].reset_index(drop=True)

        df["start time"] = df["start time"].apply(
            lambda x: JalaliDateTime.strptime(x, "%Y/%m/%d %H:%M:%S").to_gregorian())
        df["end time"] = df["end time"].apply(
            lambda x: JalaliDateTime.strptime(x, "%Y/%m/%d %H:%M:%S").to_gregorian())

        df["date"] = df["start time"].dt.date.astype(str)
        df["start hour"] = df["start time"].dt.hour
        df["end hour"] = df["end time"].dt.hour

        df.set_index("start time", inplace=True)

        self.df = df

    def handle_missing_values(self):
        """
        Handle missing timestamps in the dataset:
        """
        full_index = pd.date_range(
            start=self.df.index.min().normalize(),
            end=self.df.index.max().normalize() + pd.Timedelta(hours=23),
            freq="H"
        )

        self.df = self.df.reindex(full_index)

        missing_per_day = self.df["total number of vehicles"].isna().resample("D").sum()
        days_to_drop = missing_per_day[missing_per_day > 8].index
        self.df = self.df[~self.df.index.date.isin(days_to_drop)]

        vehicle_columns = ["total number of vehicles", "number of Class 1 vehicles",
                           "number of Class 2 vehicles", "number of Class 3 vehicles",
                           "number of Class 4 vehicles", "number of Class 5 vehicles",
                           "estimated number"]

        speed_column = "average speed"

        violation_columns = ["number of speeding violations",
                             "number of unauthorized distance violations",
                             "number of unauthorized overtaking violations"]

        self.df[vehicle_columns] = self.df[vehicle_columns].interpolate(method="time", limit=3)
        self.df[vehicle_columns] = self.df[vehicle_columns].fillna(
            self.df[vehicle_columns].rolling(window=3, min_periods=1).mean())

        self.df[speed_column] = self.df[speed_column].interpolate(method="time")

        self.df[violation_columns] = self.df[violation_columns].fillna(0)

        self.df["end time"] = self.df.index + pd.Timedelta(hours=1)
        self.df["date"] = self.df.index.date.astype(str)
        self.df["start hour"] = self.df.index.hour
        self.df["end hour"] = (self.df["start hour"] + 1) % 24

        self.df["axis code"] = self.df["axis code"].fillna(method="ffill").fillna(method="bfill")
        self.df["axis name"] = self.df["axis name"].fillna(method="ffill").fillna(method="bfill")

    def remove_anomalies(self):
        """
        Replace zero traffic values with interpolated values.
        """
        vehicle_columns = ["total number of vehicles", "number of Class 1 vehicles",
                           "number of Class 2 vehicles", "number of Class 3 vehicles",
                           "number of Class 4 vehicles", "number of Class 5 vehicles",
                           "estimated number"]

        self.df[vehicle_columns] = self.df[vehicle_columns].replace(0, np.nan)
        self.df[vehicle_columns] = self.df[vehicle_columns].interpolate(method="linear", limit=2)

    def preprocess(self):
        """
        Run all preprocessing steps in order.
        """
        self.load_data()
        self.handle_missing_values()
        self.remove_anomalies()

    def save_processed_data(self, output_path):
        """
        Save the processed data to a new CSV file.
        """
        self.df.to_csv(output_path)

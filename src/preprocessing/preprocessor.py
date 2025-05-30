import pandas as pd
from persiantools.jdatetime import JalaliDateTime


class TrafficPreprocessor:
    def __init__(self, file_path):
        """
        Initialize the preprocessor with a file path.
        """
        self.file_path = file_path
        self.df = None

        self.vehicle_columns = ["total number of vehicles", "number of Class 1 vehicles",
                                "number of Class 2 vehicles", "number of Class 3 vehicles",
                                "number of Class 4 vehicles", "number of Class 5 vehicles",
                                "estimated number"]

        self.speed_column = "average speed"

        self.violation_columns = ["number of speeding violations",
                                  "number of unauthorized distance violations",
                                  "number of unauthorized overtaking violations"]

        pd.set_option('future.no_silent_downcasting', True)

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

        for col in self.vehicle_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[self.speed_column] = pd.to_numeric(df[self.speed_column], errors="coerce")
        for col in self.violation_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        self.df = df

    def handle_missing_values(self):
        """
        Handle missing timestamps in the dataset:
        """
        full_index = pd.date_range(
            start=self.df.index.min().normalize(),
            end=self.df.index.max().normalize() + pd.Timedelta(hours=23),
            freq="h"
        )

        self.df = self.df.reindex(full_index)

        missing_per_day = self.df["total number of vehicles"].isna().resample("D").sum()
        days_to_drop = missing_per_day[missing_per_day == 24].index
        self.df = self.df[~self.df.index.normalize().isin(days_to_drop)]

        self.df[self.vehicle_columns] = self.df[self.vehicle_columns].interpolate(method="time", limit=3)
        self.df[self.vehicle_columns] = self.df[self.vehicle_columns].fillna(
            self.df[self.vehicle_columns].rolling(window=3, min_periods=1).mean())
        self.df[self.vehicle_columns] = self.df[self.vehicle_columns].fillna(
            self.df[self.vehicle_columns].rolling(window=6, min_periods=1).mean()
        )
        self.df[self.vehicle_columns] = self.df[self.vehicle_columns].fillna(
            self.df[self.vehicle_columns].rolling(window=12, center=True, min_periods=1).mean()
        )
        self.df[self.vehicle_columns] = self.df[self.vehicle_columns].ffill().bfill()

        self.df[self.speed_column] = self.df[self.speed_column].interpolate(method="time", limit=3)

        self.df[self.violation_columns] = self.df[self.violation_columns].fillna(0)

        self.df["end time"] = self.df.index + pd.Timedelta(hours=1)
        self.df["date"] = self.df.index.date.astype(str)
        self.df["start hour"] = self.df.index.hour
        self.df["end hour"] = (self.df["start hour"] + 1) % 24

        self.df["axis code"] = self.df["axis code"].ffill().bfill()
        self.df["axis name"] = self.df["axis name"].ffill().bfill()

    def remove_duplicates(self):
        """
        Remove duplicate rows from the dataset.
        """
        self.df.drop_duplicates(inplace=True)

    def handle_outliers(self):
        """
        Detect and handle outliers using the IQR method with capping.
        """
        col = "total number of vehicles"

        c = 5

        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * (IQR + c)
        upper_bound = Q3 + 1.5 * (IQR + c)

        self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)

    def preprocess(self):
        """
        Run all preprocessing steps in order.
        """
        self.load_data()
        self.handle_missing_values()
        self.remove_duplicates()
        self.handle_outliers()

    def save_processed_data(self, output_path):
        """
        Save the processed data to a new Excel file.
        """
        self.df.to_excel(output_path, index=False)

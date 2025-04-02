from sklearn.preprocessing import MinMaxScaler


class TrafficNormalizer:
    def __init__(self):
        """
        Initialize the normalizer with MinMaxScaler.
        """
        self.scaler = MinMaxScaler()
        self.numerical_columns = ["total number of vehicles", "number of Class 1 vehicles",
                                  "number of Class 2 vehicles", "number of Class 3 vehicles",
                                  "number of Class 4 vehicles", "number of Class 5 vehicles",
                                  "estimated number", "average speed",
                                  "number of speeding violations",
                                  "number of unauthorized distance violations",
                                  "number of unauthorized overtaking violations"]

    def fit(self, df):
        """
        Fit the scaler to the data.
        """
        self.scaler.fit(df[self.numerical_columns])

    def normalize(self, df):
        """
        Normalize numerical columns.
        """
        df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
        return df

    def denormalize(self, df):
        """
        Denormalize numerical columns back to original scale.
        """
        df[self.numerical_columns] = self.scaler.inverse_transform(df[self.numerical_columns])
        return df

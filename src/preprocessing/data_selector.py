import pandas as pd


def _aggregate_hourly_mean(df):
    """Aggregate the filtered data by computing the mean for each hour of the day."""
    aggregate_columns = [
        "total number of vehicles", "number of Class 1 vehicles",
        "number of Class 2 vehicles", "number of Class 3 vehicles",
        "number of Class 4 vehicles", "number of Class 5 vehicles",
        "estimated number", "average speed",
        "number of speeding violations", "number of unauthorized distance violations",
        "number of unauthorized overtaking violations"
    ]

    return df.groupby("start hour")[aggregate_columns].mean().reset_index()


class DataSelector:
    def __init__(self, df):
        """
        Initialize the DataSelector with a dataframe.
        """
        self.df = df.copy()

        self.df["date"] = pd.to_datetime(self.df["date"])

    def filter_by_year(self, year):
        """Return hourly mean data for a specific year."""
        filtered_df = self.df[self.df["year"] == str(year)]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_month(self, month):
        """Return hourly mean data for a specific month."""
        filtered_df = self.df[self.df["month"] == month]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_month_year(self, month, year):
        """Return hourly mean data for a specific month and year."""
        filtered_df = self.df[(self.df["month"] == month) & (self.df["year"] == str(year))]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_season(self, season):
        """Return hourly mean data for a specific season (spring, summer, autumn, winter)."""
        filtered_df = self.df[self.df["season"] == season]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_season_year(self, season, year):
        """Return hourly mean data for a specific season and year."""
        filtered_df = self.df[(self.df["season"] == season) & (self.df["year"] == str(year))]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekdays(self):
        """Return hourly mean data for weekdays (Saturday to Thursday)."""
        filtered_df = self.df[self.df["date"].dt.weekday != 5]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekdays_year(self, year):
        """Return hourly mean data for weekdays and year"""
        filtered_df = self.df[(self.df["date"].dt.weekday != 5) & (self.df["year"] == str(year))]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekdays_month(self, month):
        """Return hourly mean data for weekdays and month"""
        filtered_df = self.df[(self.df["date"].dt.weekday != 5) & (self.df["month"] == month)]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekdays_season(self, season):
        """Return hourly mean data for weekdays and season"""
        filtered_df = self.df[(self.df["date"].dt.weekday != 5) & (self.df["season"] == season)]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekdays_month_year(self, month, year):
        """Return hourly mean data for weekdays and month and year"""
        filtered_df = self.df[(self.df["date"].dt.weekday != 5)
                              & (self.df["month"] == month)
                              & (self.df["year"] == str(year))]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekdays_season_year(self, season, year):
        """Return hourly mean data for weekdays and season and year"""
        filtered_df = self.df[(self.df["date"].dt.weekday != 5)
                              & (self.df["season"] == season)
                              & (self.df["year"] == str(year))]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekends(self):
        """Return hourly mean data for weekends (Friday)."""
        filtered_df = self.df[self.df["date"].dt.weekday == 5]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekends_year(self, year):
        """Return hourly mean data for weekends and year"""
        filtered_df = self.df[(self.df["date"].dt.weekday == 5) & (self.df["year"] == str(year))]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekends_month(self, month):
        """Return hourly mean data for weekends and month"""
        filtered_df = self.df[(self.df["date"].dt.weekday == 5) & (self.df["month"] == month)]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekends_season(self, season):
        """Return hourly mean data for weekends and season"""
        filtered_df = self.df[(self.df["date"].dt.weekday == 5) & (self.df["season"] == season)]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekends_month_year(self, month, year):
        """Return hourly mean data for weekends and month and year"""
        filtered_df = self.df[(self.df["date"].dt.weekday == 5)
                              & (self.df["month"] == month)
                              & (self.df["year"] == year)]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_weekends_season_year(self, season, year):
        """Return hourly mean data for weekends and season and year"""
        filtered_df = self.df[(self.df["date"].dt.weekday == 5)
                              & (self.df["season"] == season)
                              & (self.df["year"] == year)]
        return _aggregate_hourly_mean(filtered_df)

    def filter_by_date_range(self, start_date, end_date):
        """Return hourly mean data within a specific date range."""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        filtered_df = self.df[(self.df["date"] >= start_date) & (self.df["date"] <= end_date)]
        return _aggregate_hourly_mean(filtered_df)

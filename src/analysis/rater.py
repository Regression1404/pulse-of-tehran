class TrafficDataRanker:
    def __init__(self, df, traffic_column='total number of vehicles'):
        """
        Initializes the ranking system for traffic data.
        :param df: DataFrame containing the traffic data.
        :param traffic_column: The column representing traffic volume.
        """
        self.df = df.copy()
        self.traffic_column = traffic_column

    def calculate_daily_mean(self):
        """
        Groups by the specified 'group_by' column and calculates the daily mean of traffic volume.
        """
        daily_mean = self.df.groupby(['axis code', 'date'])[self.traffic_column].sum().reset_index()
        daily_mean = daily_mean.groupby('axis code')[self.traffic_column].mean().reset_index()
        daily_mean.rename(columns={self.traffic_column: 'daily_mean_traffic'}, inplace=True)
        return daily_mean

    def calculate_hourly_mean(self):
        """
        Groups by the specified 'group_by' column and calculates the mean traffic per hour.
        """
        hourly_mean = self.df.groupby(['axis code', 'start hour'])[self.traffic_column].mean().reset_index()
        hourly_max = hourly_mean.groupby('axis code')[self.traffic_column].max().reset_index()
        hourly_max.rename(columns={self.traffic_column: 'max_hourly_mean_traffic'}, inplace=True)
        return hourly_max

    def rank_by_daily_mean(self):
        """
        Ranks the access points based on the daily mean traffic volume.
        """
        daily_mean = self.calculate_daily_mean()
        daily_mean['daily_mean_rank'] = daily_mean['daily_mean_traffic'].rank(ascending=False)
        return daily_mean.sort_values(by='daily_mean_rank')

    def rank_by_max_hourly_mean(self):
        """
        Ranks the access points based on the maximum of hourly mean traffic volume.
        """
        hourly_max = self.calculate_hourly_mean()
        hourly_max['max_hourly_mean_rank'] = hourly_max['max_hourly_mean_traffic'].rank(ascending=False)
        return hourly_max.sort_values(by='max_hourly_mean_rank')

    def evaluate_ranking(self, method='daily_mean'):
        """
        Evaluates and ranks access points based on the chosen method ('daily_mean' or 'max_hourly').
        """
        if method == 'daily_mean':
            return self.rank_by_daily_mean()
        elif method == 'max_hourly':
            return self.rank_by_max_hourly_mean()
        else:
            raise ValueError("Method must be either 'daily_mean' or 'max_hourly'.")

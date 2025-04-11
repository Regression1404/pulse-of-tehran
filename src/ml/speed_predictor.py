from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class SpeedPredictor:
    def __init__(self, dataframe, model=LinearRegression(), target_column="average speed"):
        self.model = model
        self.features = ["start hour", "day of week", "month", "axis code", "total number of vehicles",
                         "year_from_prediction"]
        dataframe["day of week"] = dataframe["start time"].dt.dayofweek
        self.df = dataframe
        self.target_column = target_column

    def train_model(self, prediction_year=1404):
        """ Train the model using recency-based weighting """

        df = self.df.copy()
        df["year_from_prediction"] = prediction_year - df["year"]

        x = df[self.features]
        y = df[self.target_column]

        max_weight = prediction_year - 1400
        df['weight'] = df['year_from_prediction'].apply(lambda i: max(1, max_weight - abs(i)))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        self.model.fit(x_train, y_train, sample_weight=df.loc[x_train.index, 'weight'])

        y_pred = self.model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        print("Speed Model Evaluation:")
        print(f"MSE: {round(mse, 2)}")
        print(f"R² score: {round(r2, 3)}")

    def predict(self, x_input):
        x_input = x_input.copy()
        x_input["day of week"] = x_input["start time"].dt.dayofweek
        x_input["year_from_prediction"] = 0  # since it's the target year
        x_input = x_input[self.features]
        return self.model.predict(x_input)

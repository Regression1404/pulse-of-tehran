from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def categorize_speed(speed):
    """
    Categorizes speed into Low, Medium, or High.

    :param speed: The speed value to be categorized
    :return: 'Low', 'Medium', or 'High'
    """
    if speed < 40:
        return "Low"
    elif 40 <= speed < 70:
        return "Medium"
    else:
        return "High"


class SpeedClassifier:
    def __init__(self, dataframe, model=RandomForestClassifier(), target_column="average speed"):
        """
        Initializes the TrafficSpeedPredictor class.
        """
        self.model = model
        self.features = ["start hour", "day of week", "month", "axis code", "total number of vehicles",
                         "year_from_prediction"]

        self.df = dataframe.copy()
        self.target_column = target_column

        self.df["day of week"] = self.df["start time"].dt.dayofweek
        self.df['speed category'] = self.df[self.target_column].apply(categorize_speed)

    def train_model(self, prediction_year=1404):
        """
        Trains the Random Forest Classifier model on the training data.
        """
        self.df["year_from_prediction"] = prediction_year - self.df["year"]

        x = self.df[self.features]
        y = self.df['speed category']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        self.model.fit(x_train, y_train)

        y_pred = self.model.predict(x_test)
        print("Speed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))
        print("Model Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    def predict(self, input_data):
        """
        Predicts the traffic speed category for new input data.

        :param input_data: DataFrame containing the features for prediction
        :return: Predicted traffic speed category
        """
        input_data["day of week"] = input_data["start time"].dt.dayofweek
        input_data["year_from_prediction"] = 0

        input_features = input_data[self.features]
        return self.model.predict(input_features)

    def feature_importance(self):
        """
        Prints the importance of each feature in the model.
        """
        feature_importances = self.model.feature_importances_
        for feature, importance in zip(self.features, feature_importances):
            print(f"{feature}: {importance}")

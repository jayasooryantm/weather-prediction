import pandas as pd

from weather_prediction.entity.config_entity import DataTransformationConfig
from weather_prediction.utils.common import logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataTransformation:

    def __init__(self, config:DataTransformationConfig):
        self.config = config

    def data_validation_check(self):
        logger.info("Reading Data Validation Status file...")
        status_filepath = self.config.STATUS_FILE
        status: bool
        with open(status_filepath, 'r') as f:
            line = f.read()
            status = bool(line.split(" ")[-1])
        return status

    def load_dataset(self) -> pd.DataFrame:
        dataset_file_path = self.config.dataset_file
        weather_data = pd.read_csv(dataset_file_path, parse_dates=['date'])
        return weather_data

    def transform_date_feature(self, weather_data) -> pd.DataFrame:
        weather_data['day'] = weather_data['date'].dt.day
        weather_data['month'] = weather_data['date'].dt.month
        weather_data['year'] = weather_data['date'].dt.year
        weather_data.drop('date', axis=1, inplace=True)
        return weather_data

    def make_target_features(self, weather_data) -> pd.DataFrame:
        weather_data.sort_values(by=['year', 'month', 'day', 'Minutes Since 12o Clock'], kind='mergesort', inplace=True)
        target = weather_data[self.config.target_variables].shift(-1)
        target.columns = self.config.target_column_names
        target = pd.concat([weather_data, target], axis=1)
        target.dropna(inplace=True)
        return target

    def split_data(self, x, y):
        X_train, y_train, X_test, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        return X_train, y_train, X_test, y_test

    def min_max_scaler(self, x:pd.DataFrame) -> (pd.DataFrame, MinMaxScaler):
        scaler = MinMaxScaler()

        scaled_x = scaler.fit_transform(x)
        return pd.DataFrame(scaled_x, columns=x.columns), scaler

    def invert_min_max_scaler(self, x:pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
        original_data = scaler.inverse_transform(x)

        return pd.DataFrame(original_data, x.columns)

    def save_weather_data(self, weather_data):
        weather_data.to_csv(self.config.transformed_data_filepath, index=False)



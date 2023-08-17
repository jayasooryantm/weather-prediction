import pandas as pd
import os
from pathlib import Path

from weather_prediction.entity.config_entity import DataValidationConfig
from weather_prediction.utils.common import (load_json, read_yaml)
from weather_prediction.constants import MAPPING_DICT_PATH


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def json_to_dataframe(self) -> pd.DataFrame:
        """
            Converts the JSON data into a pandas dataframe.

            :arg:
                None
            :return:
                weather_data (pandas dataframe): dataframe containing the weather data
            """
        json_filepaths = [os.path.join(self.config.unzip_data_dir, file) for file in os.listdir(self.config.unzip_data_dir) if os.path.isfile(os.path.join(self.config.unzip_data_dir, file))]
        weather_data = None
        for filepath in json_filepaths:
            json_file = load_json(Path(filepath))
            for day in json_file['SiteRep']['DV']['Location']['Period']:
                df = pd.DataFrame(day['Rep'])
                df['date'] = day['value']
                weather_data = pd.concat([weather_data, df], ignore_index=True)
        return weather_data

    def clean_dataframe(self, weather_data: pd.DataFrame):
        mapping_config = read_yaml(MAPPING_DICT_PATH)

        weather_data.rename(columns=mapping_config.data_headers, inplace=True)
        weather_data.dropna(how='any', inplace=True)
        weather_data['Wind Direction(compass)'] = weather_data['Wind Direction(compass)'].map(
                                                    mapping_config.compass_directions_map)
        weather_data['Pressure Tendency'] = weather_data['Pressure Tendency'].map(mapping_config.Pressure_tendency_map)
        weather_data['date'] = pd.to_datetime(weather_data['date'], format='%Y-%m-%dZ')
        weather_data = weather_data.astype(mapping_config.column_datatypes)
        weather_data['day'] = weather_data['date'].dt.day
        weather_data['month'] = weather_data['date'].dt.month
        weather_data['year'] = weather_data['date'].dt.year
        weather_data.drop('date', axis=1, inplace=True)
        weather_data.to_csv(self.config.dataset_file)

import pandas as pd
import os
from pathlib import Path

from weather_prediction.entity.config_entity import DataValidationConfig
from weather_prediction.utils.common import load_json
from weather_prediction.utils.common import logger


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
        logger.info("Converted json files to dataframe")
        return weather_data

    def clean_dataframe(self, weather_data: pd.DataFrame):

        weather_data.rename(columns=self.config.data_headers, inplace=True)
        weather_data.dropna(how='any', inplace=True)
        weather_data['Wind Direction(compass)'] = weather_data['Wind Direction(compass)'].map(
                                                    self.config.compass_directions_map)
        weather_data['Pressure Tendency'] = weather_data['Pressure Tendency'].map(self.config.Pressure_tendency_map)
        weather_data['date'] = weather_data['date'].str.replace('Z','')
        weather_data = weather_data.astype(self.config.column_datatypes)
        weather_data['date'] = pd.to_datetime(weather_data['date'], format='%Y-%m-%d')

        logger.info("Dataframe cleaned, saving to csv...")
        weather_data.to_csv(self.config.dataset_file, index=False)

    def validate_data_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.dataset_file)
            all_columns = list(data.columns)
            schema_columns = list(self.config.column_datatypes.keys())
            logger.info("validating dataframe columns...")
            if not sorted(all_columns) == sorted(schema_columns):
                validation_status = False
                logger.info(f"column check result: {validation_status}")
            else:
                validation_status = True
                for column, datatype in self.config.column_datatypes.items():
                    if not data[column].dtype == datatype:
                        logger.info(f"original: {data[column].dtype} == stored:{datatype}")
                        validation_status = False
                        logger.info(f"column type check result: {validation_status}")
            logger.info("Updating the status file...")
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation Status: {validation_status}")
            return validation_status
        except Exception as e:
            raise e





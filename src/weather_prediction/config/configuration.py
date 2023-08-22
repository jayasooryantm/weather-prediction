import os

from weather_prediction.constants import *
from weather_prediction.utils.common import read_yaml, create_directories
from weather_prediction.entity.config_entity import (DataIngestionConfig,
                                                     DataValidationConfig,
                                                     DataTransformationConfig,
                                                     ModelTrainerConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH,
            schema_filepath=SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        create_directories([config.unzip_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            unzip_data_dir=config.unzip_data_dir,
            dataset_file=config.dataset_file,
            STATUS_FILE=config.STATUS_FILE,
            data_headers=schema.data_headers,
            compass_directions_map=schema.compass_directions_map,
            Pressure_tendency_map=schema.Pressure_tendency_map,
            column_datatypes=schema.column_datatypes
        )
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        schema = self.schema.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            dataset_file=config.dataset_file,
            target_variables=schema.target_variables,
            target_column_names=schema.target_column_names,
            transformed_data_filepath=config.transformed_data_filepath
        )
        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        schema = self.schema.model_trainer
        params = self.params.pytorch_model_parameters

        create_directories([config.root_dir])
        create_directories([os.path.dirname(config.model_path)])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            feature_columns_names=schema.feature_columns_names,
            target_column_names=schema.target_column_names,
            pytorch_model_parameters=params

        )

        return model_trainer_config

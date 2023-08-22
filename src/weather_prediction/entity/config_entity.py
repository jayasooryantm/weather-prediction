from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    dataset_file: Path
    STATUS_FILE: Path
    data_headers: dict
    compass_directions_map: dict
    Pressure_tendency_map: dict
    column_datatypes: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    STATUS_FILE: Path
    dataset_file: Path
    target_variables: list
    target_column_names: list
    transformed_data_filepath: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_path: str
    feature_columns_names: list
    target_column_names: list
    pytorch_model_parameters: dict


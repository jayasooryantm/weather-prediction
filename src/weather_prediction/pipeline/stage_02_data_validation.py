from weather_prediction.components.data_validation import DataValidation
from weather_prediction.config.configuration import ConfigurationManager


class DataValidationPipeline:
    stage_name = "Data Validation Stage"

    def run_validation(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(data_validation_config)
        weather_data = data_validation.json_to_dataframe()
        data_validation.clean_dataframe(weather_data)
        data_validation.validate_data_columns()

if __name__ == "__main__":
    data_validation_pipeline = DataValidationPipeline()
    data_validation_pipeline.run_validation()

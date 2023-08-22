from weather_prediction.config.configuration import ConfigurationManager
from weather_prediction.components.data_transformation import DataTransformation
from weather_prediction.utils.common import logger

class DataTransformationPipeline:
    stage_name = "Data Transformation Stage"

    def run_transformation(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)
        status_flag = data_transformation.data_validation_check()
        logger.info(f"data validation status: {status_flag}")

        if status_flag:
            weather_data = data_transformation.load_dataset()
            weather_data = data_transformation.transform_date_feature(weather_data)
            weather_data = data_transformation.make_target_features(weather_data)
            data_transformation.save_weather_data(weather_data)

        else:
            logger.info(f"Data Validation Status was {status_flag}, process terminated.")
            raise f"Data Validation Status was {status_flag}, process terminated."

from weather_prediction import logger
from weather_prediction.config.configuration import ConfigurationManager
from weather_prediction.components.data_ingestion import DataIngestion




class DataIngestionPipeline:
    stage_name = "Data Ingestion Stage"
    def run(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        ingestion_pipeline = DataIngestionPipeline()
        logger.info(f">>>>>>>>>> Running: {ingestion_pipeline.stage_name} <<<<<<<<<<")
        ingestion_pipeline.run()
        logger.info(f">>>>>>>>>> Completed: {ingestion_pipeline.stage_name} <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e


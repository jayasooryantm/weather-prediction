from weather_prediction import logger
from weather_prediction.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from weather_prediction.pipeline.stage_02_data_validation import DataValidationPipeline


if __name__ == "__main__":
    try:
        ingestion_pipeline = DataIngestionPipeline()
        logger.info(f">>>>>>>>>> Running: {ingestion_pipeline.stage_name} <<<<<<<<<<")
        ingestion_pipeline.run_ingestion()
        logger.info(f">>>>>>>>>> Completed: {ingestion_pipeline.stage_name} <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

    try:
        validation_pipeline = DataValidationPipeline()
        logger.info(f">>>>>>>>>> Running: {validation_pipeline.stage_name} <<<<<<<<<<")
        validation_pipeline.run_validation()
        logger.info(f">>>>>>>>>> Completed: {validation_pipeline.stage_name} <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

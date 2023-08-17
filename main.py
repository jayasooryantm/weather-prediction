from weather_prediction import logger
from weather_prediction.pipeline.stage_01_data_ingestion import DataIngestionPipeline


if __name__ == "__main__":
    try:
        ingestion_pipeline = DataIngestionPipeline()
        logger.info(f">>>>>>>>>> Running: {ingestion_pipeline.stage_name} <<<<<<<<<<")
        ingestion_pipeline.run()
        logger.info(f">>>>>>>>>> Completed: {ingestion_pipeline.stage_name} <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
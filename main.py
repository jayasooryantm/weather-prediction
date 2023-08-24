from weather_prediction import logger
from weather_prediction.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from weather_prediction.pipeline.stage_02_data_validation import DataValidationPipeline
from weather_prediction.pipeline.stage_03_data_transformation import DataTransformationPipeline
from weather_prediction.pipeline.stage_04_model_training import ModelTrainingPipeline
from weather_prediction.pipeline.stage_05_model_eval import ModelEvalPipeline


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

    try:
        transformation_pipeline = DataTransformationPipeline()
        logger.info(f">>>>>>>>>> Running: {transformation_pipeline.stage_name} <<<<<<<<<<")
        transformation_pipeline.run_transformation()
        logger.info(f">>>>>>>>>> Completed: {transformation_pipeline.stage_name} <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

    try:
        training_pipeline = ModelTrainingPipeline()
        logger.info(f">>>>>>>>>> Running: {training_pipeline.stage_name} <<<<<<<<<<")
        training_pipeline.run_training()
        logger.info(f">>>>>>>>>> Completed: {training_pipeline.stage_name} <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e

    try:
        eval_pipeline = ModelEvalPipeline()
        logger.info(f">>>>>>>>>> Running: {eval_pipeline.stage_name} <<<<<<<<<<")
        eval_pipeline.run_eval()
        logger.info(f">>>>>>>>>> Completed: {eval_pipeline.stage_name} <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e


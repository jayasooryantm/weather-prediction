from weather_prediction.components.model_eval import ModelEvaluation
from weather_prediction.config.configuration import ConfigurationManager
from weather_prediction.utils.common import logger


class ModelEvalPipeline:
    stage_name = "Model Evaluation Stage"

    def run_eval(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_eval_config()
        model_eval = ModelEvaluation(model_eval_config)
        logger.info(f"Setting up MLflow")
        model_eval.setup_mlflow_config()
        logger.info(f"Starting Model Evaluation...")
        model_eval.eval()
        logger.info(f"Model evaluation finished")


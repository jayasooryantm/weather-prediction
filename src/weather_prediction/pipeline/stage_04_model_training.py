from weather_prediction.config.configuration import ConfigurationManager
from weather_prediction.components.model_trainer import ModelTrainer
from weather_prediction.utils.common import logger, save_torch_model


class ModelTrainingPipeline:
    stage_name = "Model Training Stage"

    def run_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)
        logger.info(f"Initialising training configuration...")
        model_trainer.init()
        logger.info(f"Model training starting...")
        model_trainer.train()
        logger.info(f"Model training finished")
        save_torch_model(model_trainer.model, model_trainer_config.model_path)
        logger.info(f"Model saved at: {model_trainer_config.model_path}")
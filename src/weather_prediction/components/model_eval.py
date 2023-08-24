import pandas as pd
import torch
import torch.nn.functional as F
import mlflow
import os
from urllib.parse import urlparse

from weather_prediction.utils.common import logger, load_torch_model
from weather_prediction.config.configuration import ModelEvalConfig

class ModelEvaluation:
    def __init__(self, config:ModelEvalConfig):
        self.config = config

    def prepare_test_data(self, data:pd.DataFrame, features: list, target: list):
        X = torch.tensor(data[features].values, dtype=torch.float32)
        y = torch.tensor(data[target].values, dtype=torch.float32)
        return X, y

    def setup_mlflow_config(self):
        os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.mlflow_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.mlflow_key

    def eval(self):
        model = load_torch_model(self.config.model_path)
        test_data = pd.read_csv(self.config.test_data_path)
        X, y = self.prepare_test_data(test_data, self.config.feature_columns_names, self.config.target_column_names)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            model.eval()
            with torch.inference_mode():
                predictions = model(X)

            pred_wind_direction, pred_pressure, pred_wind_speed, pred_temperature, pred_visibility, pred_weather_type = predictions
            stacked_pred = torch.cat([pred_wind_direction, pred_pressure, pred_wind_speed, pred_temperature, pred_visibility, pred_weather_type], dim=1)

            mse = F.mse_loss(stacked_pred, y)
            rmse = torch.sqrt(mse)
            mae = F.l1_loss(stacked_pred, y)
            r2 = 1 - (mse / torch.var(y))

            mlflow.log_params(self.config.parameters)

            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model", registered_model_name="PyTorch Model")

            logger.info(f"Evaluation Metrics:\n"
                        f"MSE: {mse:.4f} |"
                        f" RMSE: {rmse:.4f} |"
                        f" MAE: {mae:.4f} |"
                        f" R2: {r2:.4f}")

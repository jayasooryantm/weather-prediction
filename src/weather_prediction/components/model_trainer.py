import pandas as pd
from torch.utils.data import DataLoader
from torch import nn, optim

from weather_prediction.config.configuration import ModelTrainerConfig
from weather_prediction.components.model import WeatherModel
from weather_prediction.components.dataloader import WeatherDataLoader
from weather_prediction.utils.common import logger
from weather_prediction.utils.common import save_torch_model


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def load_dataset(self) -> pd.DataFrame:
        dataset_file_path = self.config.train_data_path
        weather_data = pd.read_csv(dataset_file_path)
        return weather_data

    def init(self):
        dataset = self.load_dataset()
        self.model = WeatherModel(self.config.pytorch_model_parameters)
        weather_data_class = WeatherDataLoader(data=dataset, features=self.config.feature_columns_names, target=self.config.target_column_names)
        self.dataloader = DataLoader(weather_data_class, batch_size=self.config.pytorch_model_parameters.batch_size, shuffle=True)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.pytorch_model_parameters.learning_rate)

    def train(self):

        for epoch in range(self.config.pytorch_model_parameters.num_epochs):
            for idx, (X, y) in enumerate(self.dataloader):

                # Zero the gradients
                self.optimizer.zero_grad()
                # Forward pass
                wind_direction, pressure, wind_speed, temperature, visibility, weather_type = self.model(X)

                # Compute loss
                loss_wind_direction = self.criterion(wind_direction[0][0], y[0, 0])
                loss_pressure = self.criterion(pressure[0][0], y[0, 1])
                loss_wind_speed = self.criterion(wind_speed[0][0], y[0, 2])
                loss_temperature = self.criterion(temperature[0][0], y[0, 3])
                loss_visibility = self.criterion(visibility[0][0], y[0, 4])
                loss_weather_type = self.criterion(weather_type[0][0], y[0, 5])
                # total loss
                loss = loss_wind_direction + loss_pressure + loss_wind_speed + loss_temperature + loss_visibility + loss_weather_type

                # Backpropagation and optimization
                loss.backward()
                self.optimizer.step()

                # Print progress
                if idx % 100 == 0:
                    logger.info(f"Epoch [{epoch + 1}/{self.config.pytorch_model_parameters.num_epochs}],"
                          f"Batch [{idx + 1}/{self.config.pytorch_model_parameters.num_batches}],"
                          f"Loss: {loss.item():.4f}")






import streamlit as st
import torch
from torch import nn
import webbrowser
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Atmospheric Condition Forecasting",
    page_icon="🌦️"
)

model_parameters = {
  "input_dim": 14,
  "output_dim": 1,
  "hidden_dim": 20,
  "learning_rate": 0.001,
  "batch_size": 100,
  "num_batches": 62,
  "num_epochs": 100
}

class WeatherModel(nn.Module):
    def __init__(self, parameters):
        input_dim=parameters.input_dim
        output_dim=parameters.output_dim
        hidden_dim=parameters.hidden_dim

        super(WeatherModel, self).__init__()

        # Shared hidden layers
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Output layers for each output
        self.temperature_layer = nn.Linear(hidden_dim, output_dim)
        self.visibility_layer = nn.Linear(hidden_dim, output_dim)
        self.wind_direction_layer = nn.Linear(hidden_dim, output_dim)
        self.wind_speed_layer = nn.Linear(hidden_dim, output_dim)
        self.weather_type_layer = nn.Linear(hidden_dim, output_dim)
        self.pressure_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # Pass through shared hidden layer
        shared_output = self.shared_layer(input)

        # Separate output predictions
        temperature = self.temperature_layer(shared_output)
        visibility = self.visibility_layer(shared_output)
        wind_direction = self.wind_direction_layer(shared_output)
        wind_speed = self.wind_speed_layer(shared_output)
        weather_type = self.weather_type_layer(shared_output)
        pressure = self.pressure_layer(shared_output)

        return wind_direction, pressure, wind_speed, temperature, visibility, weather_type




weather_types = { 'NA': 'Not available', '-1': 'Trace rain', '0': 'Clear night', '1': 'Sunny day', '2': 'Partly cloudy (night)', '3': 'Partly cloudy (day)', '4': 'Not used', '5': 'Mist', '6': 'Fog', '7': 'Cloudy', '8': 'Overcast', '9': 'Light rain shower (night)', '10': 'Light rain shower (day)', '11': 'Drizzle', '12': 'Light rain', '13': 'Heavy rain shower (night)', '14': 'Heavy rain shower (day)', '15': 'Heavy rain', '16': 'Sleet shower (night)', '17': 'Sleet shower (day)', '18': 'Sleet', '19': 'Hail shower (night)', '20': 'Hail shower (day)', '21': 'Hail', '22': 'Light snow shower (night)', '23': 'Light snow shower (day)', '24': 'Light snow', '25': 'Heavy snow shower (night)', '26': 'Heavy snow shower (day)', '27': 'Heavy snow', '28': 'Thunder shower (night)', '29': 'Thunder shower (day)', '30': 'Thunder'}
compass_directions_map = {1: 'N', 2: 'NNE', 3: 'NE', 4: 'ENE', 5: 'E', 6: 'ESE', 7: 'SE', 8: 'SSE', 9: 'S', 10: 'SSW', 11: 'SW', 12: 'WSW', 13: 'W', 14: 'WNW', 15: 'NW', 16: 'NNW'}
MODEL_PATH = "artifacts/model_trainer/models/model.pth"

def normalize_data(df):
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df
input_data = normalize_data([8,14.0,83.4,1012.0,8.0,16.5,17000.0,12,0,13.7,120,14,8,2023])
X = torch.tensor(input_data, dtype=torch.float32)



model = WeatherModel(model_parameters)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with torch.inference_mode():
     wind_direction, pressure, wind_speed, temperature, visibility, weather_type = model(X)


st.title("Multi-Output Model Atmospheric Condition Forecasting")
st.warning("Model is inaccurate: Values are misleading [Model enhancement in progress]")

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric("Wind Direction", str(round(wind_direction.item())))
col1.metric("Pressure", str(round(pressure.item())) + " hpa")
col2.metric("Wind Speed", str(round(wind_speed.item())) + " Mph")
col2.metric("Temperature", str(round(temperature.item())) + " °C")
col3.metric("Visibility", str(round(visibility.item())) + " m")
col3.metric("Weather Type", str(round(weather_type.item())))

st.divider()

st.markdown("[GitHub](https://github.com/jayasooryantm/weather-prediction/tree/main)")
st.markdown("[DagsHub](https://dagshub.com/jayasooryan.tm/weather-prediction)")

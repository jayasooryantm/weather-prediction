from torch import nn

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


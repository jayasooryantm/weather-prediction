import streamlit as st
from weather_prediction.utils.common import load_torch_model


st.title("Weather Prediction")
st.write("This is a PyTorch Application made for weather prediction.(Current values are hardcoded. Page under development)")

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric("Wind Direction", "SWW")
col1.metric("Pressure", "1278 P")
col2.metric("Wind Speed", "23 Km/h")
col2.metric("Temperature", "55 °C")
col3.metric("Visibility", "4 KM")
col3.metric("Weather Type", "Sunny")

st.divider()





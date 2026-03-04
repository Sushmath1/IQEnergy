import pandas as pd
import numpy as np
import joblib
import os

from live_weather import get_live_weather


def train_and_predict():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = os.path.join(BASE_DIR, "models", "demand_model.pkl")
    model = joblib.load(model_path)

    temperature, humidity, wind, radiation = get_live_weather()

    future_df = pd.DataFrame({
    "temperature_2m": [temperature],
    "relative_humidity_2m": [humidity],
    "wind_speed_10m": [wind],
    "shortwave_radiation": [radiation],
    "solar_generation_mw": [200],  
    "wind_generation_mw": [150]     
    })
    future_df["solar_generation_mw"] = future_df["shortwave_radiation"] * 0.5
    future_df["wind_generation_mw"] = future_df["wind_speed_10m"] * 2

    features = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
        "solar_generation_mw",
        "wind_generation_mw"
    ]

    X_future = future_df[features]

    future_rows = []

    for i in range(4):
        future_rows.append({
        "temperature_2m": temperature + i * 0.3,
        "relative_humidity_2m": humidity,
        "wind_speed_10m": wind + i * 0.5,
        "shortwave_radiation": radiation - i * 20,
        "solar_generation_mw": 200 - i * 10,
        "wind_generation_mw": 150 + i * 5
    })

    future_df = pd.DataFrame(future_rows)

    future_demand = model.predict(future_df)

    future_renewable = (
        future_df["solar_generation_mw"] +
        future_df["wind_generation_mw"]
    ).values

    past_demand = np.random.uniform(600, 900, 24)
    past_renewable = np.random.uniform(300, 600, 24)

    return past_demand, past_renewable, future_demand, future_renewable

  








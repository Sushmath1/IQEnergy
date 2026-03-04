import pandas as pd
import numpy as np

date_range = pd.date_range(start="2023-01-01", end="2023-12-31 23:00:00", freq="H")

df = pd.DataFrame()
df["time"] = date_range

np.random.seed(42)

df["temperature_2m"] = 25 + 10 * np.sin(2 * np.pi * df.index / 8760) + np.random.normal(0, 2, len(df))
df["relative_humidity_2m"] = 60 + 20 * np.sin(2 * np.pi * df.index / 2000) + np.random.normal(0, 5, len(df))
df["wind_speed_10m"] = np.abs(np.random.normal(6, 2, len(df)))
df["shortwave_radiation"] = np.maximum(0, 800 * np.sin(2 * np.pi * (df.index % 24) / 24))

solar_capacity = 5000  
df["solar_generation_mw"] = (df["shortwave_radiation"] / 1000) * solar_capacity

wind_capacity = 7000  
df["wind_generation_mw"] = np.where(
    df["wind_speed_10m"] < 3,
    0,
    np.minimum(wind_capacity, (df["wind_speed_10m"] / 15) ** 3 * wind_capacity)
)

df.to_csv("simulated_india_renewable_training_data.csv", index=False)

print("Dataset generated successfully.")
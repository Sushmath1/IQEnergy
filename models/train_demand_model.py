import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "dataaa", "simulated_india_renewable_training_data.csv")
df = pd.read_csv(data_path)

features = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "solar_generation_mw",
    "wind_generation_mw"
]

df["total_demand_mw"] = (
    df["solar_generation_mw"] * 0.3 +
    df["wind_generation_mw"] * 0.4 +
    df["temperature_2m"] * 2 +
    500
)

X = df[features]
y = df["total_demand_mw"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))

model_path = os.path.join(BASE_DIR, "models", "demand_model.pkl")
joblib.dump(model, model_path)

print("Model saved successfully.")
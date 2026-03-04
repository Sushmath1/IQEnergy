import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("Renewable forecasting started...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "weather_features.csv")

data = pd.read_csv(DATA_PATH)

renew_col = data.select_dtypes(include=[np.number]).columns[0]
series = data[renew_col].dropna().values

X = np.arange(len(series)).reshape(-1, 1)
y = series

model = LinearRegression()
model.fit(X, y)

def get_renewable_forecast(hours=4):
    past = y[-100:]  

    future_index = np.arange(len(series), len(series) + hours).reshape(-1, 1)
    future = model.predict(future_index)

    return past, future

if __name__ == "__main__":
    past, future = get_renewable_forecast()

    plt.plot(past, label="Past Renewable")
    plt.plot(
        range(len(past), len(past) + len(future)),
        future,
        linestyle="dotted",
        color="green",
        label="Next 4 Hours"
    )
    plt.legend()
    plt.title("Renewable Energy Forecast")
    plt.show()

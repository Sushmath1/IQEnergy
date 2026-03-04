import requests

def get_live_weather():
    print("WEATHER FUNCTION CALLED")

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=13.0827&longitude=80.2707"
        "&current_weather=true"
    )

    response = requests.get(url)
    data = response.json()

    print("API RESPONSE:", data)

    temperature = data["current_weather"]["temperature"]
    wind_speed = data["current_weather"]["windspeed"]

    humidity = 60
    radiation = 500

    return temperature, humidity, wind_speed, radiation



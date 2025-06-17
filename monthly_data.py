import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def get_last_30_day_weather(city_name: str, lat: float = None, lon: float = None) -> dict:
    """
    Fetching 30-day average temperature, humidity (%) and total rainfall
    based on city name. If latitude and longitude are not provided,
    they will be calculated using OpenCage geocoding.
    """

    # Calculate lat/lon if not provided
    if lat is None or lon is None:
        OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")
        if not OPENCAGE_API_KEY:
            raise EnvironmentError("Missing OPENCAGE_API_KEY in .env file")
        geocode_url = f'https://api.opencagedata.com/geocode/v1/json?q={city_name}&key={OPENCAGE_API_KEY}'
        geo_response = requests.get(geocode_url).json()

        if geo_response.get("results"):
            geometry = geo_response["results"][0].get("geometry")
            if geometry:
                lat = geometry.get("lat")
                lon = geometry.get("lng")

        if lat is None or lon is None:
            print(f"[ERROR] Could not retrieve coordinates for {city_name}")
            raise ValueError("Latitude and longitude could not be determined.")

    # Date range (last 30 days excluding present day)
    end_date = datetime.today().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)

    # Fetch weather data from Open-Meteo
    weather_url = (
        f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}'
        f'&start_date={start_date}&end_date={end_date}'
        f'&daily=precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean'
        f'&timezone=Asia%2FSingapore'
    )

    weather_response = requests.get(weather_url).json()

    if 'daily' not in weather_response:
        print(f"Error fetching weather data for {city_name}: {weather_response.get('error', 'Unknown error')}")
        raise ValueError("Weather data not available.")
    
    daily = weather_response['daily']
    rainfall_data = [float(r) for r in daily.get('precipitation_sum', []) if r is not None]
    temp_data = [float(t) for t in daily.get('temperature_2m_mean', []) if t is not None]
    humidity_data = [float(h) for h in daily.get('relative_humidity_2m_mean', []) if h is not None]

    # Calculate total rainfall, average temperature and humidity
    total_rainfall = sum(rainfall_data)
    avg_temp = sum(temp_data) / len(temp_data) if temp_data else 0
    avg_humidity = sum(humidity_data) / len(humidity_data) if humidity_data else 0

    return {
        "city": city_name,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "total_rainfall_mm": round(total_rainfall, 2),
        "average_temperature_c": round(avg_temp, 2),
        "average_humidity_percent": round(avg_humidity, 2)
    }

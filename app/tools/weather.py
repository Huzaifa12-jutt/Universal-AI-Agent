"""Weather tool - fetches current weather from OpenWeatherMap."""

import logging
import requests
from app.config import settings
from app.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WeatherTool(BaseTool):
    """Gets the current weather (temperature, humidity, description) for a city."""

    name = "weather"
    description = "Gets current weather (temperature, humidity, description) for a city."

    def run(self, city: str) -> dict:
        city = (city or "").strip()

        if not city:
            return {"error": "Please provide a city name, e.g. 'weather in Lahore'."}

        if not settings.WEATHER_API_KEY:
            logger.error("WEATHER_API_KEY is not configured")
            return {"error": "Weather service is not configured. Missing API key."}

        params = {
            "q": city,
            "appid": settings.WEATHER_API_KEY,
            "units": "metric",
        }

        logger.info("Weather tool called for city=%s", city)

        try:
            response = requests.get(
                settings.WEATHER_BASE_URL,
                params=params,
                timeout=settings.REQUEST_TIMEOUT,
            )

            if response.status_code == 404:
                return {"error": f"City '{city}' not found. Please check the spelling."}

            response.raise_for_status()
            data = response.json()

            return {
                "city": data["name"],
                "country": data.get("sys", {}).get("country", ""),
                "temperature": data["main"]["temp"],
                "feels_like": data["main"].get("feels_like"),
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"].title(),
            }

        except requests.exceptions.Timeout:
            logger.error("Weather API timed out for city=%s", city)
            return {"error": "Weather service timed out. Please try again."}

        except requests.exceptions.ConnectionError:
            logger.error("Weather API connection failed for city=%s", city)
            return {"error": "Could not connect to the weather service. Check your network."}

        except requests.exceptions.HTTPError as e:
            logger.error("Weather API HTTP error: %s", e)
            return {"error": f"Weather service error: {e}"}

        except Exception as e:  # noqa: BLE001 - surface any unexpected error gracefully
            logger.exception("Unexpected weather tool error")
            return {"error": f"Unexpected error: {e}"}

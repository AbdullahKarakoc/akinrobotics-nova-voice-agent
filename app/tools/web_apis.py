"""Live external-API tools: weather (Open-Meteo, free, no key) and date/time."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import httpx

from app.tools.registry import ToolRegistry
from app.tools.routing import extract_city, mentions

# WMO weather interpretation codes -> (tr, en)
WMO = {
    0: ("açık", "clear sky"), 1: ("az bulutlu", "mainly clear"), 2: ("parçalı bulutlu", "partly cloudy"),
    3: ("kapalı", "overcast"), 45: ("sisli", "fog"), 48: ("kırağılı sis", "rime fog"),
    51: ("hafif çisenti", "light drizzle"), 53: ("çisenti", "drizzle"), 55: ("yoğun çisenti", "dense drizzle"),
    61: ("hafif yağmurlu", "light rain"), 63: ("yağmurlu", "rain"), 65: ("şiddetli yağmurlu", "heavy rain"),
    71: ("hafif kar yağışlı", "light snow"), 73: ("kar yağışlı", "snow"), 75: ("yoğun kar yağışlı", "heavy snow"),
    80: ("sağanak yağışlı", "rain showers"), 81: ("kuvvetli sağanak", "heavy rain showers"),
    82: ("şiddetli sağanak", "violent rain showers"), 95: ("gök gürültülü fırtına", "thunderstorm"),
    96: ("dolu ile fırtına", "thunderstorm with hail"), 99: ("şiddetli dolu ile fırtına", "thunderstorm with heavy hail"),
}  # fmt: skip

DAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]


def route_weather(text: str) -> dict | None:
    if not mentions(text, r"\bhavan?\s*(durumu|nas|kac)|havalar|sicaklik|derece|yagmur|kar yag|weather|temperature|rain|forecast"):
        return None
    city = extract_city(text)
    return {"city": city} if city else None


def route_datetime(text: str) -> dict | None:
    pattern = (
        r"saat kac|bugunun tarihi|tarih ne|ayin kaci|bugun gunlerden|bugun hangi gun|gunlerden ne|"
        r"what time|what day|today's date|what date"
    )
    return {} if mentions(text, pattern) else None


def register(registry: ToolRegistry, timeout_s: float = 5.0, timezone: str = "Europe/Istanbul") -> None:
    client = httpx.AsyncClient(timeout=timeout_s)

    @registry.tool(
        "get_weather",
        "Get the current weather and today's min/max temperature for a city (live data).",
        {"city": {"type": "string", "description": "City name, e.g. Ankara"}},
        router=route_weather,
    )
    async def get_weather(city: str) -> dict:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1, "language": "tr"}
        )
        places = geo.json().get("results") or []
        if not places:
            return {"found": False, "note": f"City '{city}' not found."}
        place = places[0]
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min",
                "forecast_days": 1,
                "timezone": "auto",
            },
        )
        data = resp.json()
        cur, daily = data["current"], data["daily"]
        tr, en = WMO.get(cur["weather_code"], ("bilinmiyor", "unknown"))
        return {
            "found": True,
            "city": place["name"],
            "temperature_c": round(cur["temperature_2m"]),
            "feels_like_c": round(cur["apparent_temperature"]),
            "humidity_percent": cur["relative_humidity_2m"],
            "wind_kmh": round(cur["wind_speed_10m"]),
            "condition_tr": tr,
            "condition_en": en,
            "today_min_c": round(daily["temperature_2m_min"][0]),
            "today_max_c": round(daily["temperature_2m_max"][0]),
        }

    @registry.tool(
        "get_current_datetime", "Get the current local date, day of week and time.", {}, router=route_datetime
    )
    def get_current_datetime() -> dict:
        now = dt.datetime.now(ZoneInfo(timezone))
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "weekday_tr": DAYS_TR[now.weekday()],
            "weekday_en": now.strftime("%A"),
            "timezone": timezone,
        }

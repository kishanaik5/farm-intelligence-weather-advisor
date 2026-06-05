"""Weather and geocoding via the free, keyless Open-Meteo APIs.

- Geocoding:  https://geocoding-api.open-meteo.com/v1/search
- Forecast:   https://api.open-meteo.com/v1/forecast

Both are free and require no API key. Responses are cached to limit calls.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_HEADERS = {"User-Agent": "farm-intelligence-weather-advisor/1.0"}
_TIMEOUT = 30


@st.cache_data(ttl=86400, show_spinner=False)
def geocode(place: str) -> Optional[Tuple[float, float, str]]:
    """Geocode a place name to ``(lat, lon, resolved_name)`` or None.

    Raises:
        RuntimeError: on network failure (so the UI can show a clean message).
    """
    try:
        resp = requests.get(
            _GEO_URL, params={"name": place, "count": 1, "language": "en", "format": "json"},
            headers=_HEADERS, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        r = results[0]
        name = ", ".join(filter(None, [r.get("name"), r.get("admin1"), r.get("country")]))
        return float(r["latitude"]), float(r["longitude"]), name
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Geocoding failed: {exc}") from exc


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_forecast(lat: float, lon: float) -> Dict[str, pd.DataFrame]:
    """Fetch the Open-Meteo forecast; return ``{'daily': df, 'hourly': df}``.

    Raises:
        RuntimeError: on network/parse failure with a user-friendly message.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "precipitation_probability_max", "wind_speed_10m_max",
        ]),
        "hourly": ",".join(["temperature_2m", "relative_humidity_2m", "precipitation"]),
        "timezone": "auto",
        "forecast_days": 7,
    }
    try:
        resp = requests.get(_FORECAST_URL, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Weather request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Open-Meteo returned an unexpected response.") from exc

    daily = pd.DataFrame(data.get("daily", {}))
    if "time" in daily:
        daily["time"] = pd.to_datetime(daily["time"])
    hourly = pd.DataFrame(data.get("hourly", {}))
    if "time" in hourly:
        hourly["time"] = pd.to_datetime(hourly["time"])
    return {"daily": daily, "hourly": hourly}


def forecast_summary(daily: pd.DataFrame, days: int = 3) -> Dict[str, float]:
    """Summarize the next ``days`` for the rule engine (totals/extremes)."""
    head = daily.head(days)
    return {
        "days": int(len(head)),
        "total_rain_mm": round(float(head["precipitation_sum"].fillna(0).sum()), 1),
        "max_rain_prob": float(head["precipitation_probability_max"].fillna(0).max()),
        "max_temp": float(head["temperature_2m_max"].max()),
        "min_temp": float(head["temperature_2m_min"].min()),
        "max_wind": float(head["wind_speed_10m_max"].max()),
    }

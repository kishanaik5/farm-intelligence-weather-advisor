"""Weather and geocoding via the free, keyless Open-Meteo APIs.

- Geocoding:  https://geocoding-api.open-meteo.com/v1/search
- Forecast:   https://api.open-meteo.com/v1/forecast

Both are free and require no API key. Responses are cached to limit calls.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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


def reverse_geocode(lat: float, lon: float) -> str:
    """Reverse-geocode latitude and longitude to a human-readable location."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers=_HEADERS,
            timeout=5,
        )
        if resp.status_code == 200:
            addr = resp.json().get("address", {})
            loc_name = addr.get("village") or addr.get("town") or addr.get("city") or addr.get("county") or addr.get("state_district")
            state = addr.get("state")
            country = addr.get("country")
            parts = [p for p in [loc_name, state, country] if p]
            if parts:
                return ", ".join(parts)
    except Exception:
        pass
    return f"Plot ({lat:.4f}° N, {lon:.4f}° E)"


def fetch_forecast(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch Open-Meteo comprehensive forecast with weather, agro, and soil parameters.

    Returns ``{'daily': df, 'hourly': df, 'current': dict}``.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "surface_pressure", "wind_speed_10m",
            "wind_direction_10m", "cloud_cover",
        ]),
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
            "apparent_temperature_min", "precipitation_sum", "precipitation_probability_max",
            "wind_speed_10m_max", "wind_gusts_10m_max", "uv_index_max",
            "et0_fao_evapotranspiration",
        ]),
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "precipitation_probability",
            "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
            "soil_temperature_0cm", "soil_temperature_6cm",
            "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
            "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm",
            "et0_fao_evapotranspiration", "uv_index",
        ]),
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
    current = data.get("current", {})

    return {"daily": daily, "hourly": hourly, "current": current}


def forecast_summary(
    daily: pd.DataFrame | Dict[str, Any],
    hourly: pd.DataFrame | None = None,
    current: dict | None = None,
    days: int = 3
) -> Dict[str, Any]:
    """Summarize the next ``days`` for the rule engine and telemetry indicators."""
    if isinstance(daily, dict):
        current = current or daily.get("current", {})
        hourly = hourly if hourly is not None else daily.get("hourly")
        daily = daily.get("daily", pd.DataFrame())

    if daily is None or not isinstance(daily, pd.DataFrame):
        daily = pd.DataFrame()

    head = daily.head(days) if not daily.empty else pd.DataFrame()
    curr = current or {}

    mean_humidity = float("nan")
    if hourly is not None and "relative_humidity_2m" in hourly and not hourly.empty:
        mean_humidity = float(hourly.head(days * 24)["relative_humidity_2m"].mean())
    elif "relative_humidity_2m" in curr:
        mean_humidity = float(curr["relative_humidity_2m"])

    # Soil moisture from hourly model (surface 0-1cm and root-zone 3-9cm)
    soil_moisture_surface = 0.28
    soil_moisture_root = 0.32
    soil_temp_0 = 26.0
    soil_temp_6 = 24.5
    dew_point = 18.0
    if hourly is not None and not hourly.empty:
        if "soil_moisture_0_to_1cm" in hourly.columns:
            val = hourly["soil_moisture_0_to_1cm"].dropna()
            if not val.empty:
                soil_moisture_surface = float(val.iloc[0])
        if "soil_moisture_3_to_9cm" in hourly.columns:
            val = hourly["soil_moisture_3_to_9cm"].dropna()
            if not val.empty:
                soil_moisture_root = float(val.iloc[0])
        if "soil_temperature_0cm" in hourly.columns:
            val = hourly["soil_temperature_0cm"].dropna()
            if not val.empty:
                soil_temp_0 = float(val.iloc[0])
        if "soil_temperature_6cm" in hourly.columns:
            val = hourly["soil_temperature_6cm"].dropna()
            if not val.empty:
                soil_temp_6 = float(val.iloc[0])
        if "dew_point_2m" in hourly.columns:
            val = hourly["dew_point_2m"].dropna()
            if not val.empty:
                dew_point = float(val.iloc[0])

    et0_val = 4.5
    if "et0_fao_evapotranspiration" in head.columns:
        et0_val = round(float(head["et0_fao_evapotranspiration"].fillna(4.0).mean()), 2)

    uv_val = 7.5
    if "uv_index_max" in head.columns:
        uv_val = round(float(head["uv_index_max"].fillna(7.0).max()), 1)

    max_gust = float(head["wind_gusts_10m_max"].fillna(head["wind_speed_10m_max"]).max()) if "wind_gusts_10m_max" in head.columns else float(head["wind_speed_10m_max"].max())

    return {
        "days": int(len(head)),
        "current_temp": float(curr.get("temperature_2m", head["temperature_2m_max"].iloc[0] if not head.empty else 28.0)),
        "feels_like": float(curr.get("apparent_temperature", curr.get("temperature_2m", 28.0))),
        "surface_pressure": float(curr.get("surface_pressure", 1012.0)),
        "cloud_cover": float(curr.get("cloud_cover", 30.0)),
        "total_rain_mm": round(float(head["precipitation_sum"].fillna(0).sum()), 1),
        "max_rain_prob": float(head["precipitation_probability_max"].fillna(0).max()),
        "max_temp": float(head["temperature_2m_max"].max()),
        "min_temp": float(head["temperature_2m_min"].min()),
        "max_wind": float(head["wind_speed_10m_max"].max()),
        "max_wind_gust": max_gust,
        "mean_humidity": mean_humidity,
        "dew_point": dew_point,
        "et0_evapotranspiration": et0_val,
        "uv_index": uv_val,
        "soil_moisture_surface_pct": round(soil_moisture_surface * 100, 1),
        "soil_moisture_root_pct": round(soil_moisture_root * 100, 1),
        "soil_temp_surface_c": round(soil_temp_0, 1),
        "soil_temp_subsurface_c": round(soil_temp_6, 1),
    }

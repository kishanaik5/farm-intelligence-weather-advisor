"""Optional live vegetation data from AgroMonitoring (free tier).

Gated behind ``AGRO_API_KEY``. Creates a small polygon around a point, finds the
most recent satellite scene, and returns its NDVI statistics so Tab A can show
live field health instead of the bundled sample. All calls are wrapped so the
core (keyless) app never breaks.

Docs: https://agromonitoring.com/api
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

_BASE = "http://api.agromonitoring.com/agro/1.0"
_TIMEOUT = 30


def _small_polygon(lat: float, lon: float, half: float = 0.005) -> list:
    """Tiny square GeoJSON ring (~1 km) around a point; first == last vertex."""
    return [
        [lon - half, lat - half], [lon + half, lat - half],
        [lon + half, lat + half], [lon - half, lat + half], [lon - half, lat - half],
    ]


def fetch_live_agro_data(api_key: str, lat: float, lon: float, days_back: int = 60) -> Dict[str, Any]:
    """Fetch live NDVI, Soil Moisture, Soil Temperatures, and UV Index from AgroMonitoring.

    Creates a temporary polygon around the coordinate, polls the AgroMonitoring endpoints,
    and reliably deletes the polygon afterwards.
    """
    poly_id: Optional[str] = None
    res: Dict[str, Any] = {
        "ndvi": None,
        "soil": None,
        "uvi": None,
        "weather": None,
        "source": "AgroMonitoring Live Satellite & Soil Probe",
    }

    try:
        created = requests.post(
            f"{_BASE}/polygons", params={"appid": api_key},
            json={"name": f"farm-plot-{int(time.time())}", "geo_json": {
                "type": "Feature", "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [_small_polygon(lat, lon)]}}},
            timeout=_TIMEOUT,
        )
        created.raise_for_status()
        poly_id = created.json().get("id")
        if not poly_id:
            raise RuntimeError("AgroMonitoring did not return a polygon id.")

        end = int(time.time())
        start = end - days_back * 86400

        # 1. Fetch NDVI History
        try:
            hist = requests.get(
                f"{_BASE}/ndvi/history",
                params={"polyid": poly_id, "start": start, "end": end, "appid": api_key},
                timeout=_TIMEOUT,
            )
            if hist.status_code == 200:
                rows = hist.json()
                if rows:
                    latest = max(rows, key=lambda r: r.get("dt", 0))
                    d = latest.get("data", {})
                    res["ndvi"] = {
                        "mean": round(float(d.get("mean", 0)), 3),
                        "min": round(float(d.get("min", 0)), 3),
                        "max": round(float(d.get("max", 0)), 3),
                        "std": round(float(d.get("std", 0)), 3),
                        "median": round(float(d.get("median", 0)), 3),
                        "date": time.strftime("%Y-%m-%d", time.gmtime(latest.get("dt", end))),
                    }
        except Exception:
            pass

        # 2. Fetch Soil Moisture and Soil Temperature
        try:
            soil_resp = requests.get(
                f"{_BASE}/soil",
                params={"polyid": poly_id, "appid": api_key},
                timeout=_TIMEOUT,
            )
            if soil_resp.status_code == 200:
                s_data = soil_resp.json()
                t0_k = float(s_data.get("t0", 273.15))
                t10_k = float(s_data.get("t10", 273.15))
                moist = float(s_data.get("moisture", 0.0))
                res["soil"] = {
                    "moisture_pct": round(moist * 100, 1) if moist <= 1.0 else round(moist, 1),
                    "moisture_m3": round(moist, 3),
                    "surface_temp_c": round(t0_k - 273.15, 1) if t0_k > 100 else round(t0_k, 1),
                    "depth_10cm_temp_c": round(t10_k - 273.15, 1) if t10_k > 100 else round(t10_k, 1),
                }
        except Exception:
            pass

        # 3. Fetch UV Index
        try:
            uvi_resp = requests.get(
                f"{_BASE}/uvi",
                params={"polyid": poly_id, "appid": api_key},
                timeout=_TIMEOUT,
            )
            if uvi_resp.status_code == 200:
                u_data = uvi_resp.json()
                res["uvi"] = round(float(u_data.get("uvi", 0.0)), 1)
        except Exception:
            pass

        # 4. Fetch Current Weather from AgroMonitoring
        try:
            w_resp = requests.get(
                f"{_BASE}/weather",
                params={"lat": lat, "lon": lon, "appid": api_key},
                timeout=_TIMEOUT,
            )
            if w_resp.status_code == 200:
                w_data = w_resp.json()
                main_w = w_data.get("main", {})
                wind_w = w_data.get("wind", {})
                temp_k = float(main_w.get("temp", 273.15))
                res["weather"] = {
                    "temp_c": round(temp_k - 273.15, 1) if temp_k > 100 else round(temp_k, 1),
                    "humidity_pct": main_w.get("humidity"),
                    "pressure_hpa": main_w.get("pressure"),
                    "wind_speed_kmh": round(float(wind_w.get("speed", 0)) * 3.6, 1),
                }
        except Exception:
            pass

        return res

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"AgroMonitoring request failed: {exc}") from exc
    finally:
        if poly_id:
            try:
                requests.delete(f"{_BASE}/polygons/{poly_id}", params={"appid": api_key}, timeout=_TIMEOUT)
            except requests.exceptions.RequestException:
                pass


def fetch_live_ndvi_stats(api_key: str, lat: float, lon: float, days_back: int = 60) -> Dict[str, float]:
    """Return latest NDVI stats ``{mean,min,max,std,median,date}`` for a point.

    Raises:
        RuntimeError: on any API/network failure, with a user-friendly message.
    """
    agro = fetch_live_agro_data(api_key, lat, lon, days_back=days_back)
    if agro.get("ndvi"):
        return agro["ndvi"]
    raise RuntimeError("No recent satellite NDVI scene found for this location in the last 60 days.")

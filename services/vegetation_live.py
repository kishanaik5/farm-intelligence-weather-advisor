"""Optional live vegetation data from AgroMonitoring (free tier).

Gated behind ``AGRO_API_KEY``. Creates a small polygon around a point, finds the
most recent satellite scene, and returns its NDVI statistics so Tab A can show
live field health instead of the bundled sample. All calls are wrapped so the
core (keyless) app never breaks.

Docs: https://agromonitoring.com/api
"""
from __future__ import annotations

import time
from typing import Dict, Optional

import requests

_BASE = "http://api.agromonitoring.com/agro/1.0"
_TIMEOUT = 30


def _small_polygon(lat: float, lon: float, half: float = 0.005) -> list:
    """Tiny square GeoJSON ring (~1 km) around a point; first == last vertex."""
    return [
        [lon - half, lat - half], [lon + half, lat - half],
        [lon + half, lat + half], [lon - half, lat + half], [lon - half, lat - half],
    ]


def fetch_live_ndvi_stats(api_key: str, lat: float, lon: float, days_back: int = 60) -> Dict[str, float]:
    """Return latest NDVI stats ``{mean,min,max,std,median,date}`` for a point.

    Raises:
        RuntimeError: on any API/network failure, with a user-friendly message.
    """
    poly_id: Optional[str] = None
    try:
        created = requests.post(
            f"{_BASE}/polygons", params={"appid": api_key},
            json={"name": "farm-advisor", "geo_json": {
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
        hist = requests.get(
            f"{_BASE}/ndvi/history",
            params={"polyid": poly_id, "start": start, "end": end, "appid": api_key},
            timeout=_TIMEOUT,
        )
        hist.raise_for_status()
        rows = hist.json()
        if not rows:
            raise RuntimeError("No recent NDVI data for this location.")

        latest = max(rows, key=lambda r: r.get("dt", 0))
        d = latest.get("data", {})
        return {
            "mean": round(float(d.get("mean", 0)), 3),
            "min": round(float(d.get("min", 0)), 3),
            "max": round(float(d.get("max", 0)), 3),
            "std": round(float(d.get("std", 0)), 3),
            "median": round(float(d.get("median", 0)), 3),
            "date": time.strftime("%Y-%m-%d", time.gmtime(latest.get("dt", end))),
        }
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"AgroMonitoring request failed: {exc}") from exc
    finally:
        if poly_id:
            try:
                requests.delete(f"{_BASE}/polygons/{poly_id}", params={"appid": api_key}, timeout=_TIMEOUT)
            except requests.exceptions.RequestException:
                pass

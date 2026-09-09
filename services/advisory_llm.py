"""Optional natural-language phrasing of the fused advisory via Gemini REST.

Gated behind ``GEMINI_API_KEY``. When absent, the app shows the rule-based text
directly. The key is sent only in the request and never stored or logged.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import requests

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_TIMEOUT = 30
_PREFERRED = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest"]


@lru_cache(maxsize=4)
def _resolve_model(api_key: str) -> str:
    """Return a usable Gemini model for generateContent (cached per key)."""
    try:
        resp = requests.get(f"{_BASE}/models", params={"key": api_key}, timeout=_TIMEOUT)
        resp.raise_for_status()
        models: List[dict] = resp.json().get("models", [])
        available = {
            m["name"].split("/")[-1]
            for m in models
            if "generateContent" in m.get("supportedGenerationMethods", [])
        }
        for pref in _PREFERRED:
            if pref in available:
                return pref
        flash = sorted(n for n in available if "flash" in n)
        return flash[0] if flash else (sorted(available)[0] if available else _PREFERRED[0])
    except requests.exceptions.RequestException:
        return _PREFERRED[0]


def phrase_advisory(api_key: str, advisory_text: str, language: str) -> str:
    """Restate the fused advisory in ``language``, farmer-friendly and concise."""
    model = _resolve_model(api_key)
    prompt = (
        f"You are an expert Agricultural Advisor and Precision Agronomist. In {language}, write a short, clear, encouraging "
        f"advisory for a farmer based on the field-health and weather summary below. "
        f"Lead with the single most important action, then list the key reasons in simple words. "
        f"Do not invent facts.\n\n{advisory_text}"
    )
    models_to_try = [model] + [m for m in _PREFERRED if m != model]
    last_exc = None
    for cand in models_to_try:
        try:
            resp = requests.post(
                f"{_BASE}/models/{cand}:generateContent",
                params={"key": api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            continue
        except (KeyError, IndexError) as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Gemini request failed: {last_exc}")


def phrase_comprehensive_advisory(
    api_key: str,
    crop: str,
    location: str,
    lat: float,
    lon: float,
    metrics: dict,
    advisories: list,
    language: str = "English",
) -> str:
    """Generate a precision agronomic advisory incorporating all sensor and weather telemetry."""
    model = _resolve_model(api_key)
    prompt = f"""Role: You are a Principal Agronomist and Precision Agriculture Advisory Specialist.

Task:
Synthesize the provided real-time satellite, soil, and microclimate telemetry into an authoritative, actionable farm advisory in {language} for a farmer growing {crop}.

Telemetry Data:
- Location: {location} ({lat:.4f}° N, {lon:.4f}° E)
- Crop: {crop}
- Canopy Health (NDVI): {metrics.get('mean_ndvi', 'N/A')}
- Soil Moisture: {metrics.get('soil_moisture', 'N/A')} (Root-zone: {metrics.get('soil_depth_moisture', 'N/A')})
- Soil Temperature: {metrics.get('soil_temperature', 'N/A')}
- Daily Evapotranspiration (ET₀): {metrics.get('evapotranspiration', 'N/A')}
- Temperature: Current {metrics.get('temperature', 'N/A')}, Min {metrics.get('min_temp', 'N/A')}, Max {metrics.get('max_temp', 'N/A')}, Feels like {metrics.get('apparent_temperature', 'N/A')}
- Relative Humidity: {metrics.get('humidity', 'N/A')}
- Precipitation / Rain: {metrics.get('precipitation', 'N/A')} (Probability: {metrics.get('rain_probability', 'N/A')})
- Wind Speed: {metrics.get('wind_speed', 'N/A')} (Gusts: {metrics.get('wind_gusts', 'N/A')})
- Surface Pressure: {metrics.get('surface_pressure', 'N/A')}, UV Index: {metrics.get('uv_index', 'N/A')}
- Operational Rule Alerts: {'; '.join(advisories) if advisories else 'Standard maintenance'}

Format Guidelines:
Provide a clear, farmer-focused action plan in {language}:
1. **Immediate Field Priority**: Top action required today for {crop}.
2. **Irrigation Protocol**: Specific watering guidance based on the current soil moisture and ET₀ rate.
3. **Plant Protection & Spray Window**: Safe spray timing taking wind and rain probability into account.
4. **Agronomic Care**: Nutrition and stress mitigation (heat/sunlight/moisture).

Keep it direct, practical, and farmer-friendly.
"""
    models_to_try = [model] + [m for m in _PREFERRED if m != model]
    last_exc = None
    for cand in models_to_try:
        try:
            resp = requests.post(
                f"{_BASE}/models/{cand}:generateContent",
                params={"key": api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            continue
        except (KeyError, IndexError) as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Gemini advisory generation failed: {last_exc}")

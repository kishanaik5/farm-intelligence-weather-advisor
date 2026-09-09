"""Configuration & optional-secret loading for Farm Intelligence & Weather Advisor.

No secret is required: weather comes from the keyless Open-Meteo API and field
health runs on a bundled Sentinel-2 sample. Two optional keys unlock extras:
``GEMINI_API_KEY`` (natural-language advisory) and ``AGRO_API_KEY`` (live
vegetation imagery). Keys are read from env or pasted in the UI; never persisted.
"""
from __future__ import annotations

import os
from typing import Optional

GEMINI_API_KEY_ENV: str = "GEMINI_API_KEY"
AGRO_API_KEY_ENV: str = "AGRO_API_KEY"

# Languages available directly in the bundled advisory-rules knowledge base
# (no LLM needed) — maps a display name to its CSV message-column suffix.
LANGUAGE_COL = {
    "English": "en", "Hindi": "hn", "Kannada": "kn", "Tamil": "ta", "Telugu": "te",
    "Malayalam": "ml", "Marathi": "mr", "Gujarati": "gu", "Bengali": "bn",
    "Punjabi": "pa", "Urdu": "ur", "Nepali": "ne", "Odia": "or",
}
LANGUAGES = list(LANGUAGE_COL.keys())

# Standard NDVI health buckets: (label, lower_bound_inclusive).
# A pixel falls in the highest bucket whose lower bound it meets.
NDVI_CLASSES = [
    ("Healthy", 0.6),
    ("Moderate", 0.4),
    ("Stressed", 0.2),
    ("Bare/Soil", -1.0),
]

# Explicit, explainable thresholds for the weather rule engine.
RULES_CONFIG = {
    "irrigation_rain_mm": 5.0,        # below this daily rain (with heat) => irrigate
    "irrigation_temp_c": 32.0,        # at/above this max temp raises water demand
    "spray_wind_kmh": 15.0,           # above this wind => unsafe to spray
    "spray_rain_prob_pct": 40.0,      # above this rain chance => spray will wash off
    "frost_temp_c": 2.0,              # min temp at/below => frost risk
    "heat_stress_temp_c": 38.0,       # max temp at/above => heat stress
    "sowing_min_c": 12.0,             # comfortable sowing window low
    "sowing_max_c": 35.0,             # comfortable sowing window high
}


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_gemini_api_key(ui_key: Optional[str] = None) -> Optional[str]:
    """Resolve the Gemini key: UI input wins, else env var, else st.secrets, else None."""
    if ui_key and ui_key.strip():
        return ui_key.strip()
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    return None


def get_agro_api_key(ui_key: Optional[str] = None) -> Optional[str]:
    """Resolve the AgroMonitoring key: UI input wins, else env var, else st.secrets, else None."""
    if ui_key and ui_key.strip():
        return ui_key.strip()
    env_key = os.environ.get("AGRO_API_KEY") or os.environ.get("AGROMONITORING_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "AGRO_API_KEY" in st.secrets:
                return str(st.secrets["AGRO_API_KEY"]).strip()
            if "AGROMONITORING_API_KEY" in st.secrets:
                return str(st.secrets["AGROMONITORING_API_KEY"]).strip()
    except Exception:
        pass
    return None

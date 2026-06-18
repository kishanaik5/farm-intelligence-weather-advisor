"""Rule-based farm-operations advisory engine.

The forecast is bucketed into four explainable condition levels
(temp_level / moisture / wind_risk / rain_status). Those buckets are looked up in
a bundled knowledge base (`data/weather_advisory_rules.csv`) to produce four
advisories — irrigation, spraying, fungal_risk, field_work — each with a status,
severity score, icon, and a message available in 13 Indian languages. Every rule
is explicit and the triggering bucket values are surfaced as the "why".
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import streamlit as st

from utils.config import LANGUAGE_COL

_RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "weather_advisory_rules.csv")
ADVISORY_TYPES = ["irrigation", "spraying", "fungal_risk", "field_work"]


@dataclass
class Advisory:
    """One advisory item resolved from the bucketed conditions."""

    advisory_type: str
    status: str           # STABLE / CAUTION / URGENT
    severity_score: int
    title: str
    message: str
    icon_type: str
    time_window: str
    why: str              # the triggering bucket combination


def bucket_conditions(summary: Dict[str, float]) -> Dict[str, str]:
    """Map raw forecast metrics to the knowledge-base condition buckets."""
    t = summary["max_temp"]
    temp_level = "Extreme Heat" if t >= 40 else ("Very Hot" if t >= 35 else "Moderate")

    h = summary.get("mean_humidity")
    if h is None or pd.isna(h):  # fall back to rain as a moisture proxy
        moisture = "Humid" if summary["total_rain_mm"] >= 15 else (
            "Balanced" if summary["total_rain_mm"] >= 2.5 else "Very Dry")
    else:
        moisture = "Humid" if h >= 70 else ("Balanced" if h >= 40 else "Very Dry")

    w = summary["max_wind"]
    wind_risk = "Strong Gusts" if w >= 30 else ("Breezy" if w >= 15 else "Calm")

    r = summary["total_rain_mm"]
    rain_status = "Heavy Rain" if r >= 15 else ("Light Rain" if r >= 2.5 else "Dry")

    return {"temp_level": temp_level, "moisture": moisture,
            "wind_risk": wind_risk, "rain_status": rain_status}


@st.cache_data(show_spinner=False)
def load_rules() -> pd.DataFrame:
    """Load the bundled advisory-rules knowledge base."""
    return pd.read_csv(_RULES_PATH)


def get_advisories(summary: Dict[str, float], language: str = "English") -> List[Advisory]:
    """Resolve the four advisories for the bucketed forecast, sorted by severity."""
    buckets = bucket_conditions(summary)
    rules = load_rules()
    col = f"message_{LANGUAGE_COL.get(language, 'en')}"
    why = (f"{buckets['temp_level']} · {buckets['moisture']} · "
           f"{buckets['wind_risk']} · {buckets['rain_status']}")

    match = rules[
        (rules["temp_level"] == buckets["temp_level"])
        & (rules["moisture"] == buckets["moisture"])
        & (rules["wind_risk"] == buckets["wind_risk"])
        & (rules["rain_status"] == buckets["rain_status"])
    ]

    out: List[Advisory] = []
    for atype in ADVISORY_TYPES:
        row = match[match["advisory_type"] == atype]
        if row.empty:
            continue
        r = row.iloc[0]
        message = r.get(col) if col in r and pd.notna(r.get(col)) else r.get("message_en", "")
        out.append(Advisory(
            advisory_type=atype, status=str(r.get("status", "")),
            severity_score=int(r.get("severity_score", 0)),
            title=str(r.get("title", atype.replace("_", " ").title())),
            message=str(message), icon_type=str(r.get("icon_type", "")),
            time_window=str(r.get("time_window", "") or ""), why=why,
        ))
    return sorted(out, key=lambda a: a.severity_score, reverse=True)

"""Explainable, rule-based farm-operations advisory.

Each rule reads the forecast summary and emits a suggestion together with the
*reason* it fired, so nothing is a black box. Thresholds live in
``utils.config.RULES_CONFIG`` and are easy to read and tune.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from utils.config import RULES_CONFIG


@dataclass
class Suggestion:
    """One operational suggestion with its severity and triggering reason."""

    category: str          # Irrigation / Spraying / Frost / Heat / Sowing
    level: str             # info / good / warning / alert
    message: str
    reason: str


def evaluate(summary: Dict[str, float]) -> List[Suggestion]:
    """Run all rules over the forecast summary and return triggered suggestions."""
    cfg = RULES_CONFIG
    out: List[Suggestion] = []
    days = summary.get("days", 3)

    # --- Irrigation: low cumulative rain + heat raises water demand ---
    if summary["total_rain_mm"] < cfg["irrigation_rain_mm"] and summary["max_temp"] >= cfg["irrigation_temp_c"]:
        out.append(Suggestion(
            "Irrigation", "warning",
            "Irrigate soon — the crop will face water stress.",
            f"Only {summary['total_rain_mm']} mm rain forecast over {days} days with "
            f"highs up to {summary['max_temp']:.0f}°C (≥{cfg['irrigation_temp_c']:.0f}°C).",
        ))
    elif summary["total_rain_mm"] >= cfg["irrigation_rain_mm"]:
        out.append(Suggestion(
            "Irrigation", "good",
            "Hold irrigation — rain should cover crop water needs.",
            f"{summary['total_rain_mm']} mm rain expected over {days} days "
            f"(≥{cfg['irrigation_rain_mm']:.0f} mm).",
        ))

    # --- Spraying: safe only in low wind and low imminent-rain chance ---
    if summary["max_wind"] <= cfg["spray_wind_kmh"] and summary["max_rain_prob"] <= cfg["spray_rain_prob_pct"]:
        out.append(Suggestion(
            "Spraying", "good",
            "Good window for spraying in the next few days.",
            f"Wind up to {summary['max_wind']:.0f} km/h (≤{cfg['spray_wind_kmh']:.0f}) and "
            f"rain chance up to {summary['max_rain_prob']:.0f}% (≤{cfg['spray_rain_prob_pct']:.0f}%).",
        ))
    else:
        reasons = []
        if summary["max_wind"] > cfg["spray_wind_kmh"]:
            reasons.append(f"wind up to {summary['max_wind']:.0f} km/h (>{cfg['spray_wind_kmh']:.0f})")
        if summary["max_rain_prob"] > cfg["spray_rain_prob_pct"]:
            reasons.append(f"rain chance up to {summary['max_rain_prob']:.0f}% (>{cfg['spray_rain_prob_pct']:.0f}%)")
        out.append(Suggestion(
            "Spraying", "warning",
            "Avoid spraying — drift or wash-off likely.",
            " and ".join(reasons) + ".",
        ))

    # --- Frost risk ---
    if summary["min_temp"] <= cfg["frost_temp_c"]:
        out.append(Suggestion(
            "Frost", "alert",
            "Frost protection advised (cover seedlings / light irrigation at night).",
            f"Minimum temperature could drop to {summary['min_temp']:.0f}°C "
            f"(≤{cfg['frost_temp_c']:.0f}°C).",
        ))

    # --- Heat stress ---
    if summary["max_temp"] >= cfg["heat_stress_temp_c"]:
        out.append(Suggestion(
            "Heat", "alert",
            "Heat-stress risk — irrigate to cool the canopy and avoid midday operations.",
            f"Maximum temperature could reach {summary['max_temp']:.0f}°C "
            f"(≥{cfg['heat_stress_temp_c']:.0f}°C).",
        ))

    # --- Sowing/harvest suitability ---
    if cfg["sowing_min_c"] <= summary["min_temp"] and summary["max_temp"] <= cfg["sowing_max_c"] \
            and summary["max_rain_prob"] < 60:
        out.append(Suggestion(
            "Sowing", "good",
            "Conditions look suitable for sowing/field operations.",
            f"Temperatures {summary['min_temp']:.0f}–{summary['max_temp']:.0f}°C within the "
            f"{cfg['sowing_min_c']:.0f}–{cfg['sowing_max_c']:.0f}°C band and moderate rain chance.",
        ))

    return out

"""Fuse field-health (vegetation) and weather advisories into one recommendation.

Reads the mean NDVI / stressed-area share alongside the bucketed weather
advisories and condition buckets to produce one prioritised headline plus the
supporting reasons. Deliberately simple and explainable.
"""
from __future__ import annotations

from typing import Dict, List

from services.rules import Advisory


def fuse(
    mean_ndvi: float, class_pct: Dict[str, float],
    advisories: List[Advisory], buckets: Dict[str, str],
) -> Dict[str, object]:
    """Combine vegetation health + weather advisories into one advisory.

    Returns a dict with ``headline``, ``reasons`` (list), and ``priority``.
    """
    stressed_share = class_pct.get("Stressed", 0.0) + class_pct.get("Bare/Soil", 0.0)
    irrigation_needed = any(
        a.advisory_type == "irrigation" and a.status in ("CAUTION", "URGENT") for a in advisories
    )
    any_urgent = any(a.status == "URGENT" for a in advisories)
    hot = buckets.get("temp_level") in ("Very Hot", "Extreme Heat")
    very_dry = buckets.get("moisture") == "Very Dry"

    reasons: List[str] = [
        f"Mean NDVI is {mean_ndvi:.2f} with {stressed_share:.0f}% of the field stressed or bare.",
        f"Weather buckets: {buckets.get('temp_level')} · {buckets.get('moisture')} · "
        f"{buckets.get('wind_risk')} · {buckets.get('rain_status')}.",
    ]

    if stressed_share >= 25 and (irrigation_needed or (hot and very_dry)):
        headline = "Prioritise irrigation: a stressed canopy meets hot, dry conditions."
        priority = "high"
        reasons.append("Stressed vegetation + dry/hot weather compound water stress.")
    elif any_urgent and stressed_share >= 15:
        headline = "Act now: urgent weather risk on an already-stressed field."
        priority = "high"
        reasons.append("An urgent weather advisory coincides with canopy stress.")
    elif stressed_share >= 25:
        headline = "Investigate field stress (nutrition/pests/water) — weather is not the limiter."
        priority = "medium"
        reasons.append("A large stressed area without adverse weather points to an agronomic cause.")
    elif irrigation_needed:
        headline = "Plan irrigation: the canopy is healthy but conditions are drying."
        priority = "medium"
        reasons.append("Healthy vegetation now, but the forecast favours water loss.")
    else:
        headline = "Field looks healthy and weather is favourable — maintain routine care."
        priority = "low"
        reasons.append("No major vegetation or weather red flags detected.")

    for a in advisories:
        if a.status in ("CAUTION", "URGENT"):
            reasons.append(f"{a.advisory_type.replace('_', ' ').title()} ({a.status}): {a.title}")

    return {"headline": headline, "priority": priority, "reasons": reasons}


def fusion_to_text(mean_ndvi: float, class_pct: Dict[str, float], fused: Dict[str, object]) -> str:
    """Flatten the fused advisory to plain text (input for optional LLM phrasing)."""
    lines = [
        f"Headline: {fused['headline']}",
        f"Priority: {fused['priority']}",
        f"Mean NDVI: {mean_ndvi:.2f}",
        "Field health distribution: " + ", ".join(f"{k} {v}%" for k, v in class_pct.items()),
        "Reasons:",
    ]
    lines += [f"- {r}" for r in fused["reasons"]]
    return "\n".join(lines)

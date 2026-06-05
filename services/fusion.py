"""Fuse field-health (vegetation) and weather into one short advisory.

The fusion is deliberately simple and explainable: it reads the mean NDVI /
stressed-area share alongside the weather rule outputs and produces one headline
recommendation plus the supporting reasons.
"""
from __future__ import annotations

from typing import Dict, List

from services.rules import Suggestion


def fuse(mean_ndvi: float, class_pct: Dict[str, float], suggestions: List[Suggestion]) -> Dict[str, object]:
    """Combine vegetation health + weather suggestions into one advisory.

    Returns a dict with ``headline``, ``reasons`` (list), and ``priority``.
    """
    stressed_share = class_pct.get("Stressed", 0.0) + class_pct.get("Bare/Soil", 0.0)
    dry = any(s.category == "Irrigation" and s.level == "warning" for s in suggestions)
    frost = any(s.category == "Frost" for s in suggestions)
    heat = any(s.category == "Heat" for s in suggestions)

    reasons: List[str] = [
        f"Mean NDVI is {mean_ndvi:.2f} with {stressed_share:.0f}% of the field stressed or bare.",
    ]

    # Decision tree — each branch states why it fired.
    if stressed_share >= 25 and dry:
        headline = "Prioritise irrigation: a stressed canopy meets dry, hot days ahead."
        priority = "high"
        reasons.append("Stressed vegetation + a dry/hot forecast compound water stress.")
    elif frost:
        headline = "Protect the crop from frost before addressing vegetation issues."
        priority = "high"
        reasons.append("Frost risk is the most time-critical threat in the forecast.")
    elif heat and stressed_share >= 15:
        headline = "Cool the canopy: heat stress is likely on an already-stressed field."
        priority = "high"
        reasons.append("High temperatures will worsen the existing canopy stress.")
    elif stressed_share >= 25:
        headline = "Investigate field stress (nutrition/pests/water) — weather is not the limiter."
        priority = "medium"
        reasons.append("A large stressed area without adverse weather points to an agronomic cause.")
    elif dry:
        headline = "Plan irrigation: the canopy is healthy but dry days are ahead."
        priority = "medium"
        reasons.append("Healthy vegetation now, but low rain forecast may draw it down.")
    else:
        headline = "Field looks healthy and weather is favourable — maintain routine care."
        priority = "low"
        reasons.append("No major vegetation or weather red flags detected.")

    # Append the concrete weather actions as supporting reasons.
    for s in suggestions:
        if s.level in ("warning", "alert"):
            reasons.append(f"{s.category}: {s.message}")

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

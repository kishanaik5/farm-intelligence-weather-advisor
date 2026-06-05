"""Farm Intelligence & Weather Advisor — Streamlit UI.

Three tabs that fuse into one advisory: (A) field health from satellite NDVI/EVI,
(B) a weather-driven rule-based farm-operations advisory, and (C) a combined
recommendation. Business logic lives in ``services/``; this file is the UI shell.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from services.fusion import fuse, fusion_to_text
from services.rules import evaluate
from services.vegetation import analyze_field, load_sample_field
from services.weather import fetch_forecast, forecast_summary, geocode
from utils.config import LANGUAGES, get_agro_api_key, get_gemini_api_key

st.set_page_config(page_title="Farm Intelligence & Weather Advisor", page_icon="🛰️", layout="wide")

LEVEL_ICON = {"good": "✅", "info": "ℹ️", "warning": "⚠️", "alert": "🚨"}

st.title("🛰️ Farm Intelligence & Weather Advisor")
st.caption(
    "Fuse satellite field-health with a weather-driven, rule-based advisory — "
    "powered by Open-Meteo (no key) and a bundled Sentinel-2 sample field."
)


@st.cache_data(show_spinner=False)
def _sample_health():
    """Analyze the bundled sample field (cached)."""
    red, nir = load_sample_field()
    return analyze_field(red, nir)


def ndvi_heatmap(ndvi: np.ndarray):
    """Render an NDVI heatmap figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    ax.set_title("NDVI heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NDVI")
    fig.tight_layout()
    return fig


# ------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.header("Inputs")
    place = st.text_input("Location (city/village)", value="Pune")

    veg_mode = st.radio("Field data", ["Sample field", "Live (AgroMonitoring)"])
    st.divider()

    language = st.selectbox("Advisory language", options=LANGUAGES, index=0)
    gemini_ui = st.text_input("Gemini API key (optional)", type="password",
                              help="Enables natural-language advisory phrasing. Never stored.")
    agro_ui = ""
    if veg_mode == "Live (AgroMonitoring)":
        agro_ui = st.text_input("AgroMonitoring key", type="password",
                                help="Required for live vegetation. Never stored.")
    gemini_key = get_gemini_api_key(gemini_ui)
    agro_key = get_agro_api_key(agro_ui)

    with st.expander("ℹ️ How it works"):
        st.markdown(
            """
            **Two data engines fused into one advisory:**

            - **Field health (Tab A):** NDVI = (NIR−Red)/(NIR+Red) and EVI are
              computed per pixel (division-by-zero safe), then bucketed into
              *bare / stressed / moderate / healthy* by standard NDVI thresholds.
            - **Weather (Tab B):** the keyless **Open-Meteo** forecast feeds an
              **explicit rule engine** — irrigation need (low rain + heat), safe
              spraying windows (low wind + low rain chance), frost/heat alerts, and
              sowing suitability. Every suggestion shows the reason it fired.
            - **Combined (Tab C):** a small decision tree fuses canopy stress with
              the weather rules into one prioritised headline. With a Gemini key it
              is restated in your chosen language; otherwise the rule text is shown.
            """
        )

# ------------------------------ Weather fetch -------------------------------
geo = None
forecast = None
summary = None
try:
    geo = geocode(place) if place else None
    if geo:
        lat, lon, resolved = geo
        forecast = fetch_forecast(lat, lon)
        summary = forecast_summary(forecast["daily"], days=3)
except RuntimeError as exc:
    st.error(str(exc))

# ------------------------------ Field health --------------------------------
if veg_mode == "Sample field":
    health = _sample_health()
    live_note = None
else:
    health = None
    live_note = None
    if not agro_key:
        st.warning("Enter an AgroMonitoring key in the sidebar to use live vegetation, "
                   "or switch to the sample field.")
    elif geo:
        from services.vegetation_live import fetch_live_ndvi_stats

        try:
            with st.spinner("Fetching live NDVI…"):
                stats = fetch_live_ndvi_stats(agro_key, geo[0], geo[1])
            live_note = stats
        except RuntimeError as exc:
            st.error(str(exc))

tab_a, tab_b, tab_c = st.tabs(["🌿 Field Health", "🌦️ Weather", "🧭 Combined Advisory"])

# --- Tab A: Field Health ---
with tab_a:
    if health is not None:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.pyplot(ndvi_heatmap(health.ndvi))
        with c2:
            st.metric("Mean NDVI", f"{health.mean_ndvi:.2f}")
            st.markdown("**Health distribution**")
            st.bar_chart(pd.Series(health.class_pct, name="% of field"))
        st.caption("Sample Sentinel-2-like field bundled with the app (red + NIR bands).")
    elif live_note is not None:
        st.subheader(f"Live NDVI — scene {live_note['date']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean NDVI", f"{live_note['mean']:.2f}")
        m2.metric("Min", f"{live_note['min']:.2f}")
        m3.metric("Max", f"{live_note['max']:.2f}")
        m4.metric("Std", f"{live_note['std']:.2f}")
        st.caption("Live NDVI statistics from AgroMonitoring for a ~1 km area at your location.")
    else:
        st.info("Field health unavailable. Use the sample field or provide a valid live key + location.")

# --- Tab B: Weather ---
with tab_b:
    if geo and forecast is not None:
        st.subheader(f"7-day forecast — {geo[2]}")
        daily = forecast["daily"].copy()
        chart_df = daily.set_index("time")[
            ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
        ].rename(columns={
            "temperature_2m_max": "Max °C", "temperature_2m_min": "Min °C",
            "precipitation_sum": "Rain mm",
        })
        st.line_chart(chart_df[["Max °C", "Min °C"]])
        st.bar_chart(chart_df[["Rain mm"]])

        st.subheader("Operational suggestions")
        for s in evaluate(summary):
            with st.container(border=True):
                st.markdown(f"{LEVEL_ICON.get(s.level, '•')} **{s.category}: {s.message}**")
                st.caption(f"Why: {s.reason}")
    else:
        st.info("Enter a valid location in the sidebar to load the weather forecast.")

# --- Tab C: Combined ---
with tab_c:
    if health is None or summary is None:
        st.info("The combined advisory needs both field health and weather. "
                "Use the sample field and a valid location to see it.")
    else:
        suggestions = evaluate(summary)
        fused = fuse(health.mean_ndvi, health.class_pct, suggestions)
        prio_color = {"high": "🔴", "medium": "🟠", "low": "🟢"}[fused["priority"]]
        st.subheader(f"{prio_color} {fused['headline']}")

        if gemini_key:
            from services.advisory_llm import phrase_advisory

            try:
                with st.spinner(f"Phrasing in {language}…"):
                    text = phrase_advisory(
                        gemini_key, fusion_to_text(health.mean_ndvi, health.class_pct, fused), language
                    )
                st.markdown(text)
                st.caption("✨ Phrased by Gemini. Reasons below are the source logic.")
            except RuntimeError as exc:
                st.warning(f"{exc} Showing the rule-based advisory instead.")

        st.markdown("**Why this advisory:**")
        for r in fused["reasons"]:
            st.markdown(f"- {r}")

        if not gemini_key:
            st.caption("💡 Add a Gemini key in the sidebar for a natural-language version "
                       "in your chosen language.")

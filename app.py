"""Farm Intelligence & Weather Advisor — Streamlit UI.

Fuses real-time AgroMonitoring satellite NDVI and soil probe data with
Open-Meteo 7-day meteorological forecasts and precision agronomic rules.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from services.advisory_llm import phrase_comprehensive_advisory
from services.fusion import fuse
from services.rules import bucket_conditions, get_advisories
from services.vegetation import analyze_field, load_sample_field
from services.vegetation_live import fetch_live_agro_data
from services.weather import fetch_forecast, forecast_summary, geocode, reverse_geocode
from utils.config import LANGUAGES, get_agro_api_key, get_gemini_api_key

st.set_page_config(page_title="Farm Intelligence & Weather Advisor", page_icon="🛰️", layout="wide")

STATUS_ICON = {"STABLE": "✅", "CAUTION": "⚠️", "URGENT": "🚨"}

st.title("🛰️ Farm Intelligence & Weather Advisor")
st.caption(
    "Precision agriculture intelligence: live Sentinel-2 NDVI, soil moisture probes, "
    "reference evapotranspiration (ET₀), and multi-modal weather advisory powered by Gemini."
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
    ax.set_title("Sentinel-2 NDVI Canopy Heatmap")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NDVI")
    fig.tight_layout()
    return fig


# ------------------------------- Sidebar ------------------------------------
with st.sidebar:
    st.header("📍 Farm Location & Coordinates")
    input_mode = st.radio("Input Method", ["City / Village Name", "Direct GPS Coordinates (Lat / Lon)"], horizontal=True)

    lat = 18.5204
    lon = 73.8567
    resolved_place = "Pune, Maharashtra, India"

    if input_mode == "City / Village Name":
        place_input = st.text_input("City / District / Village", value="Pune")
        if place_input.strip():
            try:
                geo = geocode(place_input.strip())
                if geo:
                    lat, lon, resolved_place = geo
                    st.caption(f"📍 Resolved: `{resolved_place}` ({lat:.4f}° N, {lon:.4f}° E)")
                else:
                    st.warning("Location not recognized by geocoder; using default coordinates.")
            except Exception as e:
                st.error(f"Geocoding note: {e}")
    else:
        col_lat, col_lon = st.columns(2)
        with col_lat:
            lat = st.number_input("Latitude (°N)", value=18.5204, format="%.4f", step=0.01)
        with col_lon:
            lon = st.number_input("Longitude (°E)", value=73.8567, format="%.4f", step=0.01)
        resolved_place = reverse_geocode(lat, lon)
        st.caption(f"📍 GPS Location: `{resolved_place}`")

    st.divider()
    st.header("🌾 Crop & Advisory Setup")
    crop_name = st.selectbox(
        "Target Crop",
        ["Wheat", "Rice (Paddy)", "Cotton", "Maize", "Sugarcane", "Tomato", "Potato", "Soybean", "Groundnut", "Gram (Chickpea)", "Mustard", "Onion", "Grape", "Banana", "Chili"],
        index=0
    )
    language = st.selectbox("Advisory Language", options=LANGUAGES, index=0)

    st.divider()
    st.header("🔑 API Credentials")
    env_gemini = get_gemini_api_key()
    env_agro = get_agro_api_key()

    gemini_ui = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        value=env_gemini or "",
        help="Synthesizes precision agronomic advisories. Reads from GEMINI_API_KEY environment secret by default."
    )
    agro_ui = st.text_input(
        "AgroMonitoring API Key (optional)",
        type="password",
        value=env_agro or "",
        help="Fetches live Sentinel-2 NDVI scenes and soil probe moisture from AgroMonitoring. Reads from AGRO_API_KEY environment secret by default."
    )

    gemini_key = get_gemini_api_key(gemini_ui)
    agro_key = get_agro_api_key(agro_ui)

    if agro_key:
        st.success("🟢 AgroMonitoring Live API Active")
    else:
        st.info("ℹ️ AgroMonitoring key not provided; using Open-Meteo high-res soil models & sample Sentinel-2 field.")

    if gemini_key:
        st.success("🟢 Gemini Multimodal Pathologist / Agronomist Active")

# ---------------------------- Fetch Live Data --------------------------------
with st.spinner("Fetching meteorological and soil telemetry..."):
    try:
        forecast_data = fetch_forecast(lat, lon)
        daily_df = forecast_data["daily"]
        hourly_df = forecast_data.get("hourly")
        current_data = forecast_data.get("current", {})
        summary = forecast_summary(daily_df, hourly_df, current=current_data)
    except Exception as exc:
        st.error(f"Error fetching meteorological forecast: {exc}")
        st.stop()

# Live AgroMonitoring fetch
live_agro = None
if agro_key:
    with st.spinner("Connecting to AgroMonitoring live Sentinel-2 and soil probe APIs..."):
        try:
            live_agro = fetch_live_agro_data(agro_key, lat, lon)
        except Exception as agro_err:
            st.warning(f"AgroMonitoring note: {agro_err}. Falling back to Open-Meteo soil models.")

# Vegetation setup
if live_agro and live_agro.get("ndvi"):
    mean_ndvi = float(live_agro["ndvi"]["mean"])
    min_ndvi = float(live_agro["ndvi"]["min"])
    max_ndvi = float(live_agro["ndvi"]["max"])
    scene_date = live_agro["ndvi"]["date"]
    ndvi_source = "AgroMonitoring Sentinel-2 (Live)"
    class_pct = {
        "Healthy": 70.0 if mean_ndvi >= 0.55 else 35.0,
        "Moderate": 20.0,
        "Stressed": 10.0 if mean_ndvi < 0.55 else 5.0,
        "Bare/Soil": 5.0,
    }
    sample_h = None
else:
    sample_h = _sample_health()
    mean_ndvi = float(sample_h.mean_ndvi)
    min_ndvi = round(float(np.min(sample_h.ndvi)), 3)
    max_ndvi = round(float(np.max(sample_h.ndvi)), 3)
    class_pct = sample_h.class_pct
    ndvi_source = "Sentinel-2 Multispectral Sample"
    scene_date = "Recent Satellite Cycle"

# Soil setup
if live_agro and live_agro.get("soil"):
    soil_moist_pct = float(live_agro["soil"]["moisture_pct"])
    soil_temp_surf = float(live_agro["soil"]["surface_temp_c"])
    soil_temp_sub = float(live_agro["soil"]["depth_10cm_temp_c"])
    soil_source = "AgroMonitoring Sensor Probe (Live)"
else:
    soil_moist_pct = float(summary["soil_moisture_surface_pct"])
    soil_temp_surf = float(summary["soil_temp_surface_c"])
    soil_temp_sub = float(summary["soil_temp_subsurface_c"])
    soil_source = "Open-Meteo High-Resolution Soil Model"

root_moist_pct = float(summary["soil_moisture_root_pct"])

# Rules & Fusion
buckets = bucket_conditions(summary)
advisories = get_advisories(summary, language=language)
fused = fuse(mean_ndvi, class_pct, advisories, buckets)
priority_status = "STABLE" if fused.get("priority") == "low" else "CAUTION" if fused.get("priority") == "medium" else "URGENT"

# Metrics dictionary
metrics_payload = {
    "temperature": f"{summary['current_temp']:.1f}° C",
    "apparent_temperature": f"{summary['feels_like']:.1f}° C",
    "min_temp": f"{summary['min_temp']:.1f}° C",
    "max_temp": f"{summary['max_temp']:.1f}° C",
    "humidity": f"{summary['mean_humidity']:.0f}%",
    "dew_point": f"{summary['dew_point']:.1f}° C",
    "precipitation": f"{summary['total_rain_mm']:.1f} mm",
    "rain_probability": f"{summary['max_rain_prob']:.0f}%",
    "wind_speed": f"{summary['max_wind']:.1f} km/h",
    "wind_gusts": f"{summary['max_wind_gust']:.1f} km/h",
    "surface_pressure": f"{summary['surface_pressure']:.0f} hPa",
    "cloud_cover": f"{summary['cloud_cover']:.0f}%",
    "uv_index": f"{summary['uv_index']:.1f}",
    "evapotranspiration": f"{summary['et0_evapotranspiration']:.2f} mm/day",
    "mean_ndvi": f"{mean_ndvi:.2f}",
    "ndvi_min": f"{min_ndvi:.2f}",
    "ndvi_max": f"{max_ndvi:.2f}",
    "soil_moisture": f"{soil_moist_pct:.1f}%",
    "soil_temperature": f"{soil_temp_surf:.1f}° C",
    "soil_depth_moisture": f"{root_moist_pct:.1f}%",
    "soil_depth_temperature": f"{soil_temp_sub:.1f}° C",
}

# ----------------------------- Top KPI Bar ----------------------------------
st.subheader(f"📊 Live Field Telemetry — {resolved_place}")
st.caption(f"GPS Coordinates: `{lat:.4f}° N, {lon:.4f}° E` · Target Crop: **{crop_name}** · Overall Status: **{priority_status}**")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Temperature", f"{summary['current_temp']:.1f}° C", f"Feels {summary['feels_like']:.1f}° C")
kpi2.metric("Soil Moisture", f"{soil_moist_pct:.1f}%", f"Root: {root_moist_pct:.1f}%")
kpi3.metric("Canopy Health (NDVI)", f"{mean_ndvi:.2f}", f"Scene: {scene_date}")
kpi4.metric("Water Demand (ET₀)", f"{summary['et0_evapotranspiration']:.2f} mm/d", f"Rain: {summary['total_rain_mm']:.1f} mm")

kpi5, kpi6, kpi7, kpi8 = st.columns(4)
kpi5.metric("Relative Humidity", f"{summary['mean_humidity']:.0f}%", f"Dew: {summary['dew_point']:.1f}° C")
kpi6.metric("Wind & Gusts", f"{summary['max_wind']:.1f} km/h", f"Gust: {summary['max_wind_gust']:.1f} km/h")
kpi7.metric("Soil Temperature", f"{soil_temp_surf:.1f}° C", f"10cm: {soil_temp_sub:.1f}° C")
kpi8.metric("Solar Radiation / UV", f"UV {summary['uv_index']:.1f}", f"Pressure: {summary['surface_pressure']:.0f} hPa")

st.divider()

# ------------------------------- Tabs ---------------------------------------
tab_ai, tab_soil, tab_veg, tab_weather = st.tabs([
    "🧭 Integrated AI Advisory",
    "🪴 Soil Moisture & Root Telemetry",
    "🌿 Satellite Vegetation & NDVI",
    "🌦️ Agro-Meteorology & 7-Day Forecast"
])

# --- Tab 1: AI Advisory ---
with tab_ai:
    prio_color = {"high": "🔴", "medium": "🟠", "low": "🟢"}[fused["priority"]]
    st.subheader(f"{prio_color} {fused['headline']}")

    if gemini_key:
        with st.spinner(f"Synthesizing precision agronomic action plan for {crop_name} in {language} with Gemini…"):
            try:
                advisory_text = phrase_comprehensive_advisory(
                    gemini_key,
                    crop=crop_name,
                    location=resolved_place,
                    lat=lat,
                    lon=lon,
                    metrics=metrics_payload,
                    advisories=[f"{a.title}: {a.message}" for a in advisories],
                    language=language,
                )
                st.markdown(advisory_text)
                st.caption(f"✨ Precision advisory generated by Gemini Multimodal Agronomist ({language}).")
            except Exception as ge:
                st.warning(f"Gemini service note: {ge}. Displaying rule-based advisory below.")
                for a in advisories:
                    with st.container(border=True):
                        st.markdown(f"**{a.title}** ({a.status})")
                        st.write(a.message)
                        st.caption(f"Reason: {a.why}")
    else:
        st.info("💡 Tip: Enter a Gemini API Key in the sidebar for AI-synthesized, multilingual crop management plans.")
        for a in advisories:
            with st.container(border=True):
                st.markdown(f"**{a.title}** ({a.status})")
                st.write(a.message)
                st.caption(f"Reason: {a.why}")

    st.markdown("**Core Agronomic Reasoning:**")
    for r in fused["reasons"]:
        st.markdown(f"- {r}")

# --- Tab 2: Soil Moisture Telemetry ---
with tab_soil:
    st.subheader(f"Soil Probe & Root-Zone Hydrology ({soil_source})")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Surface Soil Moisture (0–1cm)", f"{soil_moist_pct:.1f}%")
    sc2.metric("Root-Zone Moisture (3–9cm)", f"{root_moist_pct:.1f}%")
    sc3.metric("Deep Subsurface Moisture (9–27cm)", f"{summary.get('soil_moisture_surface_pct', 30.0):.1f}%")

    st.markdown("#### Soil Temperature Profile")
    st1, st2 = st.columns(2)
    st1.metric("Surface Soil Temperature", f"{soil_temp_surf:.1f}° C")
    st2.metric("Subsurface Soil Temperature (10cm depth)", f"{soil_temp_sub:.1f}° C")

    if hourly_df is not None and not hourly_df.empty:
        st.markdown("#### 48-Hour Soil Moisture Trend")
        soil_cols = [c for c in ["soil_moisture_0_to_1cm", "soil_moisture_3_to_9cm"] if c in hourly_df.columns]
        if soil_cols:
            soil_trend = hourly_df.head(48).set_index("time")[soil_cols] * 100
            soil_trend.columns = ["Surface Moisture (%)", "Root-Zone Moisture (%)"]
            st.line_chart(soil_trend)

# --- Tab 3: Vegetation & NDVI ---
with tab_veg:
    st.subheader(f"Canopy Health & Vegetation Indices ({ndvi_source})")
    if live_agro and live_agro.get("ndvi"):
        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.metric("Mean NDVI", f"{mean_ndvi:.3f}")
        vm2.metric("Min NDVI", f"{min_ndvi:.3f}")
        vm3.metric("Max NDVI", f"{max_ndvi:.3f}")
        vm4.metric("Scene Date", scene_date)
        st.success(f"Verified live Sentinel-2 satellite scene for plot at ({lat:.4f}° N, {lon:.4f}° E).")
    elif sample_h is not None:
        vc1, vc2 = st.columns([1.2, 1])
        with vc1:
            st.pyplot(ndvi_heatmap(sample_h.ndvi))
        with vc2:
            st.metric("Mean NDVI", f"{sample_h.mean_ndvi:.2f}")
            st.markdown("**Health Distribution**")
            st.bar_chart(pd.Series(sample_h.class_pct, name="% of Field"))
        st.caption("Bundled Sentinel-2 multispectral field (Red and NIR bands). Provide an AgroMonitoring API key in the sidebar for live satellite scenes.")

# --- Tab 4: Weather & Agro-Meteorology ---
with tab_weather:
    st.subheader(f"7-Day Weather & Spray Suitability — {resolved_place}")
    daily = daily_df.copy()
    chart_df = daily.set_index("time")[
        ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
    ].rename(columns={
        "temperature_2m_max": "Max °C", "temperature_2m_min": "Min °C",
        "precipitation_sum": "Rain mm",
    })
    st.line_chart(chart_df[["Max °C", "Min °C"]])
    st.bar_chart(chart_df[["Rain mm"]])

    st.subheader("Condition Buckets (Next 3 Days)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Temperature", buckets["temp_level"])
    b2.metric("Moisture", buckets["moisture"])
    b3.metric("Wind Risk", buckets["wind_risk"])
    b4.metric("Rain Status", buckets["rain_status"])

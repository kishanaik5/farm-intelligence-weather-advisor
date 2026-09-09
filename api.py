"""FastAPI microservice for Farm Intelligence & Weather Advisor (Hugging Face dual-mode)."""
from __future__ import annotations

import asyncio
import datetime
import json
import os
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel

from services.advisory_llm import phrase_comprehensive_advisory
from services.fusion import fuse
from services.rules import bucket_conditions, get_advisories
from services.vegetation import analyze_field, load_sample_field
from services.vegetation_live import fetch_live_agro_data
from services.weather import fetch_forecast, forecast_summary, geocode, reverse_geocode
from utils.config import get_agro_api_key, get_gemini_api_key

app = FastAPI(title="Farm Intelligence & Weather Advisor API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_sse(event: str, data: Any) -> str:
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def make_log(tag: str, msg: str, log_type: str = "normal") -> dict:
    now = datetime.datetime.now()
    time_str = now.strftime("%H:%M:%S") + f".{now.microsecond // 1000:03d}"
    return {"time": time_str, "tag": tag, "msg": msg, "type": log_type}


class FarmIntelQuery(BaseModel):
    location: Optional[str] = "Pune"
    lat: Optional[float] = None
    lon: Optional[float] = None
    crop: str = "Wheat"
    language: str = "English"
    gemini_key: Optional[str] = None
    agro_key: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "farm-intelligence-weather-advisor",
        "version": "2.0.0",
        "has_gemini": bool(get_gemini_api_key()),
        "has_agro": bool(get_agro_api_key()),
        "port": int(os.environ.get("PORT", 8000)),
    }


@app.post("/advise/stream")
async def advise_stream(
    query: FarmIntelQuery,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_agro_key: Optional[str] = Header(None, alias="X-Agro-Key"),
):
    resolved_gemini_key = x_gemini_key or query.gemini_key or get_gemini_api_key()
    resolved_agro_key = x_agro_key or query.agro_key or get_agro_api_key()

    async def generator() -> AsyncGenerator[str, None]:
        yield format_sse("log", make_log("API", f"POST /advise/stream crop='{query.crop}', lang='{query.language}'"))
        await asyncio.sleep(0.05)

        try:
            # 1. Resolve Coordinates & Location Name
            if query.lat is not None and query.lon is not None:
                lat = float(query.lat)
                lon = float(query.lon)
                if query.location and query.location.strip() and not query.location.startswith("Coordinates"):
                    resolved_name = query.location.strip()
                else:
                    resolved_name = reverse_geocode(lat, lon)
                yield format_sse("log", make_log("GPS", f"Coordinates received: {lat:.4f}° N, {lon:.4f}° E → {resolved_name}", "success"))
            else:
                loc_str = (query.location or "Pune").strip()
                yield format_sse("log", make_log("Open-Meteo", f"Geocoding '{loc_str}' via Open-Meteo REST service..."))
                geo_res = geocode(loc_str)
                if geo_res:
                    lat, lon, resolved_name = geo_res
                    yield format_sse("log", make_log("Open-Meteo", f"Resolved: {resolved_name} ({lat:.4f}° N, {lon:.4f}° E)", "success"))
                else:
                    lat, lon, resolved_name = 18.5204, 73.8567, f"{loc_str} (Default Coord)"
                    yield format_sse("log", make_log("Open-Meteo", "Using fallback coordinates for weather query.", "warn"))

            # 2. Fetch Live AgroMonitoring Data if key is configured
            live_agro = None
            if resolved_agro_key:
                yield format_sse("log", make_log("AgroMonitoring", "Querying AgroMonitoring Sentinel-2 & Soil Probe APIs...", "normal"))
                try:
                    live_agro = fetch_live_agro_data(resolved_agro_key, lat, lon)
                    if live_agro.get("ndvi"):
                        yield format_sse("log", make_log("AgroMonitoring", f"Live NDVI: mean={live_agro['ndvi']['mean']:.3f} (Scene: {live_agro['ndvi']['date']})", "success"))
                    if live_agro.get("soil"):
                        yield format_sse("log", make_log("AgroMonitoring", f"Live Soil Probe: Moisture={live_agro['soil']['moisture_pct']}%, Surface Temp={live_agro['soil']['surface_temp_c']}°C", "success"))
                    if live_agro.get("uvi"):
                        yield format_sse("log", make_log("AgroMonitoring", f"Live UV Index: {live_agro['uvi']}", "success"))
                except Exception as agro_err:
                    yield format_sse("log", make_log("AgroMonitoring", f"Notice: {agro_err}. Falling back to Open-Meteo high-res soil telemetry.", "warn"))

            # 3. Fetch Comprehensive Weather & Agro Parameters from Open-Meteo
            yield format_sse("log", make_log("Weather", "Fetching 7-day meteorological & soil forecast array..."))
            forecast_data = fetch_forecast(lat, lon)
            daily_df = forecast_data["daily"]
            hourly_df = forecast_data.get("hourly")
            current_dict = forecast_data.get("current", {})
            summary = forecast_summary(daily_df, hourly_df, current=current_dict)

            # 4. Integrate Field Health & Canopy Reflectance
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
            else:
                red, nir = load_sample_field()
                field_health = analyze_field(red, nir)
                mean_ndvi = float(field_health.mean_ndvi)
                min_ndvi = round(float(np.min(field_health.ndvi)), 3)
                max_ndvi = round(float(np.max(field_health.ndvi)), 3)
                class_pct = field_health.class_pct
                ndvi_source = "Sentinel-2 Multispectral Sample"
                scene_date = "Recent Satellite Cycle"

            stressed_share = class_pct.get("Stressed", 0.0) + class_pct.get("Bare/Soil", 0.0)
            yield format_sse("log", make_log("Vegetation", f"Canopy NDVI: {mean_ndvi:.2f} ({ndvi_source}), Stress: {stressed_share:.1f}%", "success"))

            # 5. Integrate Soil Moisture and Temperatures
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
            yield format_sse("log", make_log("Soil", f"Soil Moisture: {soil_moist_pct:.1f}% (Root-zone: {root_moist_pct:.1f}%), Surface Temp: {soil_temp_surf:.1f}°C", "success"))

            # 6. Evaluate Rule Engine & Multi-Modal Fusion
            yield format_sse("log", make_log("RuleEngine", f"Evaluating agronomic operational rules in {query.language}..."))
            buckets = bucket_conditions(summary)
            advisories = get_advisories(summary, language=query.language)

            fused = fuse(mean_ndvi, class_pct, advisories, buckets)
            priority_status = "STABLE" if fused.get("priority") == "low" else "CAUTION" if fused.get("priority") == "medium" else "URGENT"
            yield format_sse("log", make_log("Fusion", f"Integrated priority: {priority_status} · Triggered advisories: {len(advisories)}", "success"))

            # 7. Compile Comprehensive Metrics
            metrics = {
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

            # 8. Optional Gemini LLM Precision Advisory Phrasing
            ai_advisory = None
            if resolved_gemini_key:
                yield format_sse("log", make_log("Gemini", f"Synthesizing precision agricultural advisory for {query.crop} in {query.language}...", "normal"))
                try:
                    ai_advisory = phrase_comprehensive_advisory(
                        resolved_gemini_key,
                        crop=query.crop,
                        location=resolved_name,
                        lat=lat,
                        lon=lon,
                        metrics=metrics,
                        advisories=[f"{a.title}: {a.message}" for a in advisories],
                        language=query.language,
                    )
                    yield format_sse("log", make_log("Gemini", "Agronomic advisory synthesized successfully.", "success"))
                except Exception as ge:
                    yield format_sse("log", make_log("Gemini", f"Gemini advisory note: {ge}", "warn"))

            yield format_sse("log", make_log("Pipeline", "Telemetry fusion & advisory pipeline completed. Status 200 OK.", "success"))

            result_data = {
                "location": resolved_name,
                "lat": lat,
                "lon": lon,
                "crop": query.crop,
                "overall_status": priority_status,
                "metrics": metrics,
                "vegetation": {
                    "source": ndvi_source,
                    "mean_ndvi": round(mean_ndvi, 3),
                    "min_ndvi": round(min_ndvi, 3),
                    "max_ndvi": round(max_ndvi, 3),
                    "scene_date": scene_date,
                },
                "soil": {
                    "source": soil_source,
                    "moisture_pct": round(soil_moist_pct, 1),
                    "root_zone_moisture_pct": round(root_moist_pct, 1),
                    "surface_temp_c": round(soil_temp_surf, 1),
                    "depth_temp_c": round(soil_temp_sub, 1),
                },
                "weather": {
                    "current_temp": summary["current_temp"],
                    "feels_like": summary["feels_like"],
                    "humidity_pct": summary["mean_humidity"],
                    "precipitation_mm": summary["total_rain_mm"],
                    "rain_probability_pct": summary["max_rain_prob"],
                    "wind_speed_kmh": summary["max_wind"],
                    "wind_gusts_kmh": summary["max_wind_gust"],
                    "evapotranspiration_mm": summary["et0_evapotranspiration"],
                    "uv_index": summary["uv_index"],
                    "surface_pressure_hpa": summary["surface_pressure"],
                },
                "advisories": [f"{a.title}: {a.message}" for a in advisories[:4]],
                "ai_advisory": ai_advisory,
                "summary": fused.get("headline", "Optimal microclimate and canopy conditions."),
            }
            yield format_sse("result", result_data)

        except Exception as ex:
            yield format_sse("log", make_log("ERR", f"Farm advisor error: {ex}", "err"))
            yield format_sse("error", {"error": str(ex)})

    return StreamingResponse(generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)

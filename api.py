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
from pydantic import BaseModel

from services.fusion import fuse
from services.rules import bucket_conditions, get_advisories
from services.vegetation import analyze_field, load_sample_field
from services.weather import fetch_forecast, forecast_summary, geocode
from utils.config import get_gemini_api_key

app = FastAPI(title="Farm Intelligence & Weather Advisor API", version="1.0.0")

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
    location: str = "Pune"
    crop: str = "Wheat"
    language: str = "English"
    gemini_key: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "farm-intelligence-weather-advisor", "port": int(os.environ.get("PORT", 8000))}


@app.post("/advise/stream")
async def advise_stream(
    query: FarmIntelQuery,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
):
    resolved_gemini_key = x_gemini_key or query.gemini_key or get_gemini_api_key()

    async def generator() -> AsyncGenerator[str, None]:
        yield format_sse("log", make_log("API", f"POST /advise/stream location='{query.location}', crop='{query.crop}'"))
        await asyncio.sleep(0.1)

        try:
            yield format_sse("log", make_log("Open-Meteo", f"Geocoding '{query.location}' via Open-Meteo REST service..."))
            geo_res = geocode(query.location)
            if geo_res:
                lat, lon, resolved_name = geo_res
                yield format_sse("log", make_log("Open-Meteo", f"Resolved: {resolved_name} ({lat:.4f}° N, {lon:.4f}° E)", "success"))
            else:
                lat, lon, resolved_name = 18.5204, 73.8567, f"{query.location} (Default Coord)"
                yield format_sse("log", make_log("Open-Meteo", "Using fallback coordinates for weather query.", "warn"))

            yield format_sse("log", make_log("Weather", "Fetching 7-day meteorological forecast array..."))
            forecast_data = fetch_forecast(lat, lon)
            daily_df = forecast_data["daily"]
            hourly_df = forecast_data.get("hourly")
            summary = forecast_summary(daily_df, hourly_df)

            yield format_sse("log", make_log("Sentinel-2", "Analyzing 10m Sentinel-2 multispectral Red/NIR surface reflectance..."))
            red, nir = load_sample_field()
            field_health = analyze_field(red, nir)
            stressed_share = field_health.class_pct.get("Stressed", 0.0) + field_health.class_pct.get("Bare/Soil", 0.0)
            yield format_sse("log", make_log("Vegetation", f"Field NDVI mean: {field_health.mean_ndvi:.2f}, Canopy Stress: {stressed_share:.1f}%", "success"))

            yield format_sse("log", make_log("RuleEngine", f"Evaluating agronomic operational rules in {query.language}..."))
            buckets = bucket_conditions(summary)
            advisories = get_advisories(summary, language=query.language)

            fused = fuse(field_health.mean_ndvi, field_health.class_pct, advisories, buckets)
            priority_status = "STABLE" if fused.get("priority") == "low" else "CAUTION" if fused.get("priority") == "medium" else "URGENT"
            yield format_sse("log", make_log("Fusion", f"Integrated status: {priority_status} - Advisories: {len(advisories)}", "success"))
            yield format_sse("log", make_log("Pipeline", "Advisory synthesis completed. Status 200 OK.", "success"))

            today_max_t = float(daily_df["temperature_2m_max"].iloc[0]) if "temperature_2m_max" in daily_df.columns else 31.0
            today_rain = float(daily_df["precipitation_sum"].iloc[0]) if "precipitation_sum" in daily_df.columns else 0.0
            today_wind = float(daily_df["wind_speed_10m_max"].iloc[0]) if "wind_speed_10m_max" in daily_df.columns else 12.0

            result_data = {
                "location": resolved_name,
                "overall_status": priority_status,
                "metrics": {
                    "temperature": f"{today_max_t:.1f}° C",
                    "precipitation": f"{today_rain:.1f} mm",
                    "wind_speed": f"{today_wind:.1f} km/h",
                    "mean_ndvi": f"{field_health.mean_ndvi:.2f}",
                },
                "advisories": [f"{a.title}: {a.message}" for a in advisories[:3]],
                "summary": fused.get("headline", "Field canopy condition is optimal. Continue regular schedule."),
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

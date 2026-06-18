---
title: Farm Intelligence & Weather Advisor
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
---

# 🛰️ Farm Intelligence & Weather Advisor

One app, three tabs that fuse into a single advisory: **(A)** field health from
satellite vegetation indices, **(B)** a weather-driven, **rule-based** farm-
operations advisory, and **(C)** a combined recommendation. Runs with **zero
setup and no API key**.

> **Independent, public-data reimplementation.** Clean-room build from public data
> sources and first principles. It does not reuse, import, or reproduce any
> private/company code or data.

## What it does

- **Field health:** computes NDVI & EVI from red + NIR bands, buckets pixels into
  bare / stressed / moderate / healthy, and shows an NDVI heatmap + distribution.
- **Weather:** pulls the keyless Open-Meteo forecast, buckets it into condition
  levels (temp/moisture/wind/rain), and resolves four advisories — **irrigation,
  spraying, fungal_risk, field_work** — each with a status (STABLE/CAUTION/URGENT),
  severity score, and a message in **13 Indian languages**, from a bundled
  knowledge base. Every advisory shows the triggering bucket combination.
- **Combined:** fuses canopy stress with the weather rules into one prioritised
  headline, optionally restated in an Indian language via Gemini.

## Architecture (pipeline)

1. **Vegetation** (`services/vegetation.py`) — NDVI = (NIR−Red)/(NIR+Red), EVI
   (div-0 safe), pixel bucketing by standard NDVI thresholds.
2. **Weather** (`services/weather.py`) — Open-Meteo geocoding + 7-day forecast,
   cached with `@st.cache_data`.
3. **Rules** (`services/rules.py`) — a transparent threshold engine; every
   `Suggestion` carries its triggering reason.
4. **Fusion** (`services/fusion.py`) — a small decision tree combines field health
   + weather into one headline with supporting reasons.
5. **Optional LLM** (`services/advisory_llm.py`) — Gemini REST phrasing, gated.

## Public data sources

- **Weather:** [Open-Meteo](https://open-meteo.com/) forecast + geocoding APIs —
  **free, no API key**.
- **Vegetation:** a bundled small Sentinel-2-like sample field (`data/*.npy`,
  red + NIR). Optional **live** mode via `AGRO_API_KEY` (AgroMonitoring free tier).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
# Optional extras:
export GEMINI_API_KEY="..."   # natural-language advisory phrasing
export AGRO_API_KEY="..."     # live vegetation instead of the sample
```

## Set the secrets on Hugging Face (both optional)

In your Space: **Settings → Variables and secrets → New secret**

- `GEMINI_API_KEY` — for natural-language advisory phrasing.
- `AGRO_API_KEY` — for live vegetation mode.

The app reads them via `os.environ` and also accepts them via sidebar inputs;
keys are never written to disk or logs. With no keys set, the app runs fully on
Open-Meteo + the bundled sample field.

## Geocoding / data notes

Locations are geocoded with Open-Meteo's geocoder (global, free). The bundled
field is a synthetic Sentinel-2-like sample for demonstration; use live mode or
upload your own red/NIR arrays for a real field.

"""Optional natural-language phrasing of the fused advisory via Gemini REST.

Gated behind ``GEMINI_API_KEY``. When absent, the app shows the rule-based text
directly. The key is sent only in the request and never stored or logged.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import requests

_BASE = "https://generativelanguage.googleapis.com/v1beta"
_TIMEOUT = 30
_PREFERRED = ["gemini-flash-latest", "gemini-2.0-flash", "gemini-1.5-flash"]


@lru_cache(maxsize=4)
def _resolve_model(api_key: str) -> str:
    """Return a usable Gemini model for generateContent (cached per key)."""
    try:
        resp = requests.get(f"{_BASE}/models", params={"key": api_key}, timeout=_TIMEOUT)
        resp.raise_for_status()
        models: List[dict] = resp.json().get("models", [])
        available = {
            m["name"].split("/")[-1]
            for m in models
            if "generateContent" in m.get("supportedGenerationMethods", [])
        }
        for pref in _PREFERRED:
            if pref in available:
                return pref
        flash = sorted(n for n in available if "flash" in n)
        return flash[0] if flash else (sorted(available)[0] if available else _PREFERRED[0])
    except requests.exceptions.RequestException:
        return _PREFERRED[0]


def phrase_advisory(api_key: str, advisory_text: str, language: str) -> str:
    """Restate the fused advisory in ``language``, farmer-friendly and concise.

    Raises:
        RuntimeError: on API/network failure with a user-friendly message.
    """
    model = _resolve_model(api_key)
    prompt = (
        f"You are a farm advisor. In {language}, write a short, clear, encouraging "
        f"advisory for a smallholder farmer based on the field-health and weather "
        f"summary below. Lead with the single most important action, then list the "
        f"key reasons in simple words. Do not invent facts.\n\n{advisory_text}"
    )
    try:
        resp = requests.post(
            f"{_BASE}/models/{model}:generateContent",
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
    except (KeyError, IndexError) as exc:
        raise RuntimeError("Gemini returned an unexpected response.") from exc

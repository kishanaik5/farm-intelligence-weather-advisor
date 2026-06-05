"""Vegetation indices (NDVI / EVI) and field-health classification.

Works on red + near-infrared (NIR) reflectance arrays. The bundled sample is a
small Sentinel-2-like field; users can also upload their own red/NIR ``.npy``
arrays. All index math is division-by-zero safe.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from utils.config import NDVI_CLASSES

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_EPS = 1e-6


@dataclass
class FieldHealth:
    """Computed field-health summary."""

    ndvi: np.ndarray
    evi: np.ndarray
    mean_ndvi: float
    class_pct: Dict[str, float]  # health class -> % of pixels


def load_sample_field() -> Tuple[np.ndarray, np.ndarray]:
    """Load the bundled sample red and NIR reflectance arrays."""
    red = np.load(os.path.join(_DATA_DIR, "sample_field_red.npy"))
    nir = np.load(os.path.join(_DATA_DIR, "sample_field_nir.npy"))
    return red, nir


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red), safe against zero denominators."""
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    denom = nir + red
    ndvi = np.where(np.abs(denom) < _EPS, 0.0, (nir - red) / np.where(denom == 0, _EPS, denom))
    return np.clip(ndvi, -1.0, 1.0)


def compute_evi(red: np.ndarray, nir: np.ndarray, blue: np.ndarray | None = None) -> np.ndarray:
    """EVI with the standard coefficients; uses a 2-band approximation if no blue.

    EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L). Without a blue band we
    drop the C2*Blue term (a common 2-band approximation).
    """
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    G, C1, C2, L = 2.5, 6.0, 7.5, 1.0
    blue_term = C2 * blue.astype(np.float32) if blue is not None else 0.0
    denom = nir + C1 * red - blue_term + L
    evi = G * (nir - red) / np.where(np.abs(denom) < _EPS, _EPS, denom)
    return np.clip(evi, -1.0, 1.0)


def classify_health(ndvi: np.ndarray) -> Dict[str, float]:
    """Bucket NDVI pixels into health classes and return % per class."""
    flat = ndvi.ravel()
    total = flat.size
    pct: Dict[str, float] = {}
    assigned = np.zeros(total, dtype=bool)
    for label, lower in NDVI_CLASSES:  # ordered high -> low
        mask = (~assigned) & (flat >= lower)
        pct[label] = round(100.0 * mask.sum() / total, 1)
        assigned |= mask
    return pct


def analyze_field(red: np.ndarray, nir: np.ndarray) -> FieldHealth:
    """Run the full vegetation analysis on red + NIR arrays."""
    ndvi = compute_ndvi(red, nir)
    evi = compute_evi(red, nir)
    return FieldHealth(
        ndvi=ndvi, evi=evi,
        mean_ndvi=round(float(np.nanmean(ndvi)), 3),
        class_pct=classify_health(ndvi),
    )

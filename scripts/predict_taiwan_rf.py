"""Apply the MARIDA-trained pixel RF to a Taiwan Sentinel-2 crop.

Usage (from the repo root):
    python scripts/predict_taiwan_rf.py runs/marida-pixel-rf/model.joblib \
        data/raw/sentinel2/ne-coast-2025-03-25.tif runs/taiwan-ne-coast

Needs the model from scripts/pixel_rf.py and the GeoTIFF from scripts/fetch_taiwan_s2.py.
Drops B09 to match MARIDA's 11 bands, keeps only water pixels (NDWI > 0) at least
50 m from the coastline, and writes class counts, a class map and a debris overlay.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import rasterio
from scipy.ndimage import binary_erosion

# MARIDA class order after the repo's 1..15 -> 0..14 shift (marine-debris.github.io utils/assets.py)
LABELS = ["Marine Debris", "Dense Sargassum", "Sparse Sargassum", "Natural Organic Material", "Ship",
          "Clouds", "Marine Water", "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water",
          "Waves", "Cloud Shadows", "Wakes", "Mixed Water"]
TAIWAN_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
MARIDA_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
COAST_BUFFER_PX = 5  # 50 m: drop land/water mixed pixels that fooled FDI


def write_png(path, rgb):
    with rasterio.open(path, "w", driver="PNG", width=rgb.shape[2], height=rgb.shape[1],
                       count=3, dtype="uint8") as dst:
        dst.write(rgb)


def main(model_path, tif, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_path)
    with rasterio.open(tif) as src:
        dn = src.read().astype(np.float32)
        profile = src.profile
    refl = dn * 1e-4
    x = refl[[TAIWAN_ORDER.index(b) for b in MARIDA_ORDER]]  # (11, H, W), MARIDA band order
    B = dict(zip(TAIWAN_ORDER, refl))
    ndwi = (B["B03"] - B["B08"]) / (B["B03"] + B["B08"] + 1e-6)
    water = ndwi > 0
    open_water = binary_erosion(water, iterations=COAST_BUFFER_PX)

    feats = x.reshape(11, -1).T[open_water.ravel()]
    proba = model.predict_proba(feats)
    classes = list(model.classes_)
    pred = np.array(classes)[proba.argmax(1)]
    pmax = proba.max(1)

    cls_map = np.full(water.shape, 255, dtype=np.uint8)
    cls_map[open_water] = pred.astype(np.uint8)
    counts = {LABELS[c]: int((pred == c).sum()) for c in classes}
    debris_idx = classes.index(0) if 0 in classes else None
    p_debris = proba[:, debris_idx] if debris_idx is not None else np.zeros(len(pred))

    profile.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(out_dir / "rf-classes.tif", "w", **profile) as dst:
        dst.write(cls_map, 1)

    rgb = np.stack([B["B04"], B["B03"], B["B02"]])
    grey = (np.clip(rgb.mean(0) / 0.15, 0, 1) * 255).astype(np.uint8)
    over = np.stack([grey, grey, grey])
    full_debris = np.zeros(water.shape, bool)
    full_debris[open_water] = pred == 0
    ships = np.zeros(water.shape, bool)
    ships[open_water] = np.isin(pred, [4, 13])  # Ship, Wakes
    over[0][ships], over[1][ships], over[2][ships] = 255, 220, 0
    over[0][full_debris], over[1][full_debris], over[2][full_debris] = 255, 0, 0
    write_png(out_dir / "rf-debris-overlay.png", over)

    summary = {
        "model": str(model_path), "image": str(tif),
        "water_pixels_ndwi": int(water.sum()), "open_water_pixels_after_50m_coast_buffer": int(open_water.sum()),
        "class_counts_open_water": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "debris_pixels_argmax": int((pred == 0).sum()),
        "debris_pixels_p_gt_0.5": int((p_debris > 0.5).sum()),
        "debris_pixels_p_gt_0.8": int((p_debris > 0.8).sum()),
        "median_max_probability": float(np.median(pmax)),
        "note": "Pixel RF trained on MARIDA (spectral only); MARIDA scenes are not Taiwan, so this is transfer without Taiwan labels.",
    }
    (out_dir / "rf-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])

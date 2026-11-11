"""Fetch a Sentinel-2 L2A crop of Taiwan's northeast coast and compute FDI/NDVI.

Usage (from the repo root, after `pip install -e '.[geo]'`):
    python scripts/fetch_taiwan_s2.py .

Picks the least-cloudy scene inside BBOX/DATES from Microsoft Planetary Computer
(free, no account). Change BBOX/DATES below to look at another stretch of coast.
Uses only the standard library + rasterio/numpy.
Outputs (all under the repo's git-ignored runs/ and data/raw/):
  - 12-band uint16 GeoTIFF in marinedebrisdetector's L2ABANDS order (DN, offset-harmonized)
  - RGB and FDI quicklook PNGs, plus a JSON summary
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a"
# Ruifang/Gongliao coast, New Taipei (SOW's worst 2018/2022 segment is in 瑞芳區)
BBOX = [121.78, 25.07, 121.96, 25.17]
DATES = "2025-01-01/2026-11-10"
L2ABANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
WATER, CLOUD = {6}, {3, 8, 9, 10}  # SCL classes: water; cloud shadow + cloud med/high + cirrus

REPO = Path(sys.argv[1])
RAW = REPO / "data/raw/sentinel2"
OUT = REPO / "runs/taiwan-ne-coast"


def get_json(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def read_crop(href, token, dst_crs_bounds=None, shape=None):
    with rasterio.open(f"{href}?{token}") as src:
        b = transform_bounds("EPSG:4326", src.crs, *BBOX)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win, out_shape=shape,
                       resampling=Resampling.bilinear if shape else Resampling.nearest)
        return arr, src.crs, src.window_transform(win), win


def write_png(path, rgb_uint8):
    h, w = rgb_uint8.shape[1:]
    with rasterio.open(path, "w", driver="PNG", width=w, height=h, count=3, dtype="uint8") as dst:
        dst.write(rgb_uint8)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res = get_json(STAC, {"collections": ["sentinel-2-l2a"], "bbox": BBOX, "datetime": DATES,
                          "query": {"eo:cloud_cover": {"lt": 20}}, "limit": 100})
    items = sorted(res["features"], key=lambda f: f["properties"]["eo:cloud_cover"])
    print(f"{len(items)} scenes <20% cloud over the bbox")
    token = get_json(SAS)["token"]  # anonymous read token; never printed

    # Rank candidates by clear-sky fraction inside the crop (SCL at 20 m)
    best = None
    for f in items[:12]:
        scl, *_ = read_crop(f["assets"]["SCL"]["href"], token)
        cloudy = np.isin(scl, list(CLOUD)).mean()
        water = np.isin(scl, list(WATER)).mean()
        print(f"  {f['properties']['datetime'][:10]}  scene_cloud={f['properties']['eo:cloud_cover']:.1f}%"
              f"  crop_cloud={cloudy:.1%}  crop_water={water:.1%}")
        if best is None or cloudy < best[1]:
            best = (f, cloudy, water)
    item, crop_cloud, crop_water = best
    props = item["properties"]
    date = props["datetime"][:10]
    baseline = float(props.get("s2:processing_baseline", "0"))
    print(f"picked {item['id']} (baseline {baseline})")

    ref, crs, transform, _ = read_crop(item["assets"]["B02"]["href"], token)
    shape = ref.shape
    bands = []
    for b in L2ABANDS:
        arr, *_ = read_crop(item["assets"][b]["href"], token, shape=shape)
        arr = arr.astype(np.int32)
        if baseline >= 4.0:  # PB 04.00+ adds a +1000 DN offset (BOA_ADD_OFFSET)
            arr = np.clip(arr - 1000, 0, None)
        bands.append(arr.astype(np.uint16))
    stack = np.stack(bands)
    scl10, *_ = read_crop(item["assets"]["SCL"]["href"], token, shape=shape)

    tif = RAW / f"ne-coast-{date}.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=shape[1], height=shape[0], count=12,
                       dtype="uint16", crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(stack)
        dst.descriptions = tuple(L2ABANDS)

    r = stack.astype(np.float32) * 1e-4
    B = {b: r[i] for i, b in enumerate(L2ABANDS)}
    # Biermann et al. 2020 FDI: NIR minus a baseline interpolated between RE2 (B06) and SWIR1 (B11)
    l_nir, l_red, l_swir = 832.8, 664.6, 1613.7
    fdi = B["B08"] - (B["B06"] + (B["B11"] - B["B06"]) * (l_nir - l_red) / (l_swir - l_red) * 10)
    ndvi = (B["B08"] - B["B04"]) / (B["B08"] + B["B04"] + 1e-6)
    water = np.isin(scl10, list(WATER))
    clear_water = water & ~np.isin(scl10, list(CLOUD))

    rgb = np.stack([B["B04"], B["B03"], B["B02"]])
    rgb = (np.clip(rgb / 0.15, 0, 1) * 255).astype(np.uint8)
    write_png(OUT / f"rgb-{date}.png", rgb)

    # FDI quicklook: grey RGB background, water pixels with high FDI in red
    thr = float(np.percentile(fdi[clear_water], 99.9)) if clear_water.any() else np.nan
    grey = rgb.mean(axis=0).astype(np.uint8)
    over = np.stack([grey, grey, grey])
    hot = clear_water & (fdi > max(thr, 0.0))
    over[0][hot], over[1][hot], over[2][hot] = 255, 0, 0
    write_png(OUT / f"fdi-hotspots-{date}.png", over)

    fw = fdi[clear_water]
    summary = {
        "scene": item["id"], "date": date, "processing_baseline": baseline,
        "bbox_wgs84": BBOX, "shape_10m": list(shape), "crs": str(crs),
        "crop_cloud_fraction": round(float(crop_cloud), 4), "crop_water_fraction": round(float(crop_water), 4),
        "clear_water_pixels": int(clear_water.sum()),
        "fdi_clear_water": {"p50": float(np.percentile(fw, 50)), "p99": float(np.percentile(fw, 99)),
                            "p99.9": thr, "max": float(fw.max())} if fw.size else None,
        "hotspot_pixels_fdi_gt_p99.9_and_gt_0": int(hot.sum()),
        "ndvi_of_hotspots_median": float(np.median(ndvi[hot])) if hot.any() else None,
        "geotiff_for_detector": str(tif), "band_order": L2ABANDS, "scaling_for_model": "DN * 1e-4",
    }
    (OUT / f"summary-{date}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

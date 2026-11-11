# Research log

Newest entry first. Each entry should say what you tried, what happened (with numbers), and what you'll try next. Link the `runs/` folder or script that produced the numbers. Dead ends belong here too; they save the next person a week.

```
## YYYY-MM-DD — short title
**Tried:** …
**Result:** …
**Next:** …
```

---

## 2026-11-11 — First look at Taiwan (instructor, with Claude)

**Tried:** downloaded and inspected HAIDA; pulled a Sentinel-2 crop of Taiwan's northeast coast; ran FDI on it; trained a random forest on full MARIDA and applied it to the Taiwan crop.

**Result:**

*HAIDA* (downloaded from the [HAIDA repo](https://github.com/LiaoSteve/HAIDA-Trash-Dataset-High-Resolution-Aerial-image) Drive links into `data/raw/haida/`):

| | HAIDA | HAIDA2 |
|---|---|---|
| Frames | 1,319 (456 contain no litter) | 391 (67 contain no litter) |
| Boxes | 6,421: `garbage` 3,904, `bottle` 2,517 | 1,616: `garbage` 1,597, `bottle` 19 |
| Flight folders | 10 (`data1`…`data18`; `data13` alone has 483 frames) | 8 (`data20`…`data_28`) |
| Frame sizes | mostly 1920×1080 (also 2688×1512, 1024×576) | 1920×1080, 1280×720, 2688×1512 |
| Median box size | 0.22% of the frame (p10 0.05%, p90 2.6%) | 0.26% |

The labels are Pascal VOC XML (not Ezlabel, as earlier notes said). Frames are cut from drone videos (`out_120.jpg`). The zips use PPMd compression: Python's `zipfile` fails, `bsdtar` works. The sample frame `data13_10.jpg` is a low pass over a harbour seawall with tetrapods; the litter is a few pixels across.

*Sentinel-2* (`scripts/fetch_taiwan_s2.py`): scene `S2C_MSIL2A_20250325T022551_R046_T51RUH`, 2025-03-25, a crop over lon 121.78–121.96, lat 25.07–25.17 (Ruifang/Gongliao). 1824×1123 pixels at 10 m, 0% cloud in the crop, 54% water.

*FDI:* the top 0.1% of FDI pixels over water (1,117 pixels) sit along the coastline, on boats and on part of a river. Their median NDVI is 0.36, which suggests land/vegetation mixed in. Nothing in open water looked like debris. A river sediment plume visible in the RGB image was *not* flagged.

*Random forest on MARIDA* (`scripts/pixel_rf.py`, 23 s on an M4 Pro CPU, `runs/marida-pixel-rf/`):

| Split | Macro-F1 | Mean IoU |
|---|---|---|
| Validation | 0.578 | 0.474 |
| Test | 0.499 | 0.393 |

Marine Debris on test: F1 0.690. It found 353 of 381 debris pixels (recall 0.93), but only 353 of the 642 pixels it called debris were debris (precision 0.55). The weakest classes were Waves, Wakes, Shallow Water and Mixed Water (F1 < 0.1).

*The same random forest on Taiwan* (`scripts/predict_taiwan_rf.py`, `runs/taiwan-ne-coast/`): of 1,066,402 open-water pixels (NDWI > 0, at least 50 m from the coast), **696,488 (65%) were labelled "Clouds"** on a cloud-free day. Another 179,464 were labelled Marine Water, 65,465 Waves and 39,688 Wakes. Marine Debris: 17 pixels, only 1 with probability above 0.5.

*Why "Clouds"?* Median reflectance, Taiwan open water vs MARIDA classes (MARIDA sampled from 300 patches):

| Band | Taiwan open water | MARIDA water | MARIDA turbid water | MARIDA cloud |
|---|---|---|---|---|
| B02 (blue) | 0.049 | 0.033 | 0.047 | 0.088 |
| B04 (red) | 0.031 | 0.017 | 0.030 | 0.066 |
| B08 (near-infrared) | 0.026 | 0.012 | 0.012 | 0.061 |
| B11 (SWIR) | 0.021 | 0.008 | 0.005 | 0.058 |
| B12 (SWIR) | 0.017 | 0.006 | 0.003 | 0.046 |

Taiwan's water is about 2–2.5× brighter than MARIDA's in the infrared, partway toward cloud.

**Interpretation (untested):** the model has never seen water this bright, so it falls back on "Clouds." This is domain shift, the same thing that happened going from PetFinder to AAC. There are two candidate causes:
1. **Processing mismatch.** According to `research/original-combinations-report.md`, MARIDA patches are ACOLITE Rayleigh-corrected, while this crop is Sen2Cor L2A. The model was trained on reflectance processed differently from what it was given, so the comparison isn't apples to apples.
2. **Sun glint.** A flat brightening across the infrared bands is also what glint looks like.

They aren't exclusive. The published pretrained UNet++ (marinedebrisdetector) was trained on Sen2Cor L2A, the same processing as this crop, which makes it the better-matched test.

*Why a random forest?* It was the only MARIDA model in the repo already built and tested, and it trains in seconds on a CPU, which was good enough to check whether anything transfers. MARIDA's labels are sparse: 0.94% of training pixels are labelled and only 1,943 of them are debris. With so few, per-pixel models are competitive. The MARIDA paper reports its random forest beating its U-Net on debris (F1 0.80 vs 0.50). Ours is spectral-only (no indices or texture), hence 0.69.

**Not done:** running the published pretrained detector (marinedebrisdetector) on this crop. Installing it here needs a permission change, so it moves to Kaggle.

**Next (suggestions, see `PATHS.md`):** test the glint idea with a spectral correction (Path B); convert HAIDA to YOLO and train a first detector (Path A).

---

## 2026-11-09 — Repo created (instructor, with Codex)

**Tried:** built the starter repo from the November research notes.

**Result:** the `waste_lab` package (small multi-head model, training, baselines, adapters), a 48-dataset catalog, MARIDA downloaded and checksum-verified (1.16 GB, official splits 694/328/359), the MOENV 2026-09-09 dump-site snapshot converted to GeoJSON (491 sites, 20 with garbled text flagged), and 8 passing tests. A six-patch MARIDA run worked end to end on CPU. See `VALIDATION.md`.

**Not done:** HAIDA download, MADOS, any GPU run, any Taiwan imagery. The Knaeps spectra downloads failed (HTTP 504).

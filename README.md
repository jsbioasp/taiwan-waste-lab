# Taiwan Waste Lab

A student research repo with two goals in Taiwan: **detecting waste**, and **detecting chemicals: air pollution**. It uses free drone and satellite data, starting from models and datasets made elsewhere ([transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)). Everything trains on free hosted GPUs ([Kaggle](https://www.kaggle.com/) or [Colab](https://colab.research.google.com/)) and runs inference on a laptop CPU.

This is a research starter, not finished software. The code that's here runs and is tested. Nothing in it has yet been shown to work on Taiwan; finding out what does is the project. The work is organized into **research paths**. Pick one, combine them, or propose your own; new paths get added as good ideas come up. [`docs/PATHS.md`](docs/PATHS.md) has guidance for each path and a [glossary](docs/PATHS.md#new-terms). Terms in this README are linked or explained where they first appear.

## Research paths

| Path | What you'd detect | Data | Status |
|---|---|---|---|
| [**A. Drone litter**](#path-a--drone-litter-haida) | Trash on Taiwan's coast in drone photos | A Taiwan drone dataset with boxes drawn around trash | Data inspected; label converter still to write |
| [**B. Satellite debris**](#path-b--satellite-debris-sentinel-2) | Floating debris along Taiwan's coast | Free satellite images, plus a labelled benchmark from outside Taiwan | First Taiwan run done; it fails in an informative way |
| [**C. Dump sites**](#path-c--dump-sites-moenv) | Reported illegal dump sites, seen from space | The Taiwan government's list of 491 sites | Data in the repo; review not started |
| [**D. Detecting chemicals: air pollution**](#path-d--detecting-chemicals-air-pollution-sentinel-5p) | Gases such as nitrogen dioxide (NO₂) and sulfur dioxide (SO₂), measured from orbit | A satellite spectrometer, plus Taiwan's ground air-quality stations | Data access checked; no code yet |

**A. Drone litter.** [HAIDA](https://github.com/LiaoSteve/HAIDA-Trash-Dataset-High-Resolution-Aerial-image) is a set of drone photos of Taiwan's coastline from [National Taiwan Ocean University](https://www.ntou.edu.tw/) (NTOU), with boxes drawn around trash labelled `bottle` or `garbage`. Start from a [YOLO](https://docs.ultralytics.com/) detector (a fast, widely used object detector) pretrained on everyday photos, teach it Taiwan litter, and test it on drone flights it never saw.

**B. Satellite debris.** [Sentinel-2](https://sentiwiki.copernicus.eu/web/s2-mission) is a pair of free European satellites that photograph the Earth in 13 colour bands, from blue to infrared, with each pixel covering 10–60 m. [MARIDA](https://marine-debris.github.io/) is an open benchmark of Sentinel-2 image patches where experts labelled pixels as debris, ships, clouds, different kinds of water and so on. None of it is from Taiwan. A model trained on MARIDA calls most of Taiwan's clear sea "Clouds." Figuring out why, and fixing it, is the research.

**C. Dump sites.** [MOENV](https://data.moenv.gov.tw/en) (Taiwan's Ministry of Environment) publishes [a list of reported illegal dump sites](https://data.gov.tw/en/datasets/145968). Many of the inspection notes describe buried, overgrown or cleared waste. Before anyone builds a detector, someone has to find out which sites are visible from space at all.

**D. Detecting chemicals: air pollution.** [Sentinel-5P](https://sentiwiki.copernicus.eu/web/s5p-mission) is a European satellite carrying [TROPOMI](https://www.tropomi.eu/), a spectrometer that identifies gases by the way they absorb sunlight. Examples are nitrogen dioxide (NO₂, mostly from traffic and power plants) and sulfur dioxide (SO₂, from burning coal and oil). MOENV runs ground stations to check it against. This is the one path with real chemical detection, and it's mostly tabular modelling, like the shelter project.

## Getting started (every path)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[geo]'                             # add ,aerial for Path A
python -m unittest discover -s tests -v             # 8 tests should pass
```

For Kaggle and Colab (GPU, uploading data, saving outputs), see [`docs/STUDENT_START.md`](docs/STUDENT_START.md). Keep a [research log](docs/LOG.md) as you go.

---

## Path A — Drone litter (HAIDA)

**What we know so far** ([log](docs/LOG.md)): [HAIDA and HAIDA2](https://github.com/LiaoSteve/HAIDA-Trash-Dataset-High-Resolution-Aerial-image) download fine. Together they have 1,710 frames from 18 drone flights and 8,037 boxes (`garbage` 5,501, `bottle` 2,536; HAIDA2 is almost all `garbage`).
- The labels are in **Pascal VOC** format: one XML file per image listing its boxes, a common format that converts easily to YOLO's.
- Frames are cut from videos, so neighbouring frames are near-copies: split by flight.
- The litter is tiny, with a median box of 0.2% of the frame (about 90×45 px in a 1920×1080 frame).

**Data**

| Data | Where | License / terms |
|---|---|---|
| HAIDA (1,319 frames) + HAIDA2 (391 frames) | Google Drive links in the [HAIDA repo](https://github.com/LiaoSteve/HAIDA-Trash-Dataset-High-Resolution-Aerial-image) | The repo's code is MIT; the image terms aren't stated separately. Cite the NTOU thesis and **keep the images out of this repo** (use a private Kaggle Dataset) |
| Optional extra litter datasets to learn from: [UAVVaste](https://uavvaste.github.io/) (drone litter), [DroneWaste](https://github.com/lucamora/dronewaste) (labelled with 20 material types), MDImageNet (Taiwan coastal photos, 34 litter types) | See [PATHS.md, Path A](docs/PATHS.md#models-to-choose-from-path-a) | CC BY 4.0 per the research notes; access not checked |

**Code**

| File | What it does | Status |
|---|---|---|
| `scripts/aerial.py` | Trains or predicts with [Ultralytics YOLO](https://docs.ultralytics.com/) on a YOLO-format dataset | Untested on real data; `imgsz` (input size) fixed at 640 |
| `scripts/audit_aerial_split.py` | Checks a split CSV: no image twice, no flight in two splits | Untested on real data |
| `data/templates/aerial-split.csv` | Blank split table (`image,group,split`) | Template |

**Get started**

```bash
pip install -e '.[aerial]'
mkdir -p data/raw/haida && bsdtar -xf HAIDA.zip -C data/raw/haida   # the zips use PPMd compression, which Python's zipfile can't read; use 7-Zip on Windows
# write the VOC → YOLO converter and a flight-level split (to-dos below), then:
python scripts/audit_aerial_split.py your-split.csv
python scripts/aerial.py train --data /path/to/data.yaml --device 0          # on a Kaggle GPU
python scripts/aerial.py predict --model runs/aerial/train/weights/best.pt --source some_images/ --device cpu
```

**To-dos**
- [ ] Pascal VOC → YOLO converter, with a visual check of the converted boxes
- [ ] A script that builds a flight-level split into `aerial-split.csv`
- [ ] An `--imgsz` option for `scripts/aerial.py`

---

## Path B — Satellite debris (Sentinel-2)

**What we know so far** ([log](docs/LOG.md)):
- A random forest trained on [MARIDA](https://marine-debris.github.io/) pixels takes 23 s on a CPU. It scores test **macro-F1** 0.50, and 0.69 on the debris class. [F1](https://en.wikipedia.org/wiki/F-score) balances missed detections against false alarms; macro-F1 averages it over all classes.
- On a cloud-free [Sentinel-2](https://sentiwiki.copernicus.eu/web/s2-mission) image of the Ruifang/Gongliao coast (New Taipei, 2025-03-25), it labels **65% of open water "Clouds"** and finds almost no debris.
- Taiwan's water looks about twice as bright as MARIDA's in the infrared bands. There are two suspects, both untested:
  - **Sun glint:** sunlight reflecting off the sea toward the satellite.
  - **Processing mismatch:** MARIDA's images were corrected for the atmosphere with [ACOLITE](https://github.com/acolite/acolite), while ours are ESA's standard [Sen2Cor](https://step.esa.int/main/snap-supported-plugins/sen2cor/) product, called L2A.
- The [Floating Debris Index](https://www.nature.com/articles/s41598-020-62298-z) (FDI, a formula that makes floating material stand out in the near-infrared) flags the coastline and boats on its own, not debris.

**Data**

| Data | Where | License / terms |
|---|---|---|
| MARIDA: 1,381 labelled Sentinel-2 patches (694 train / 328 val / 359 test) | `scripts/fetch_starter_data.py` (1.16 GB, checksum-verified); 6 sample patches ship in `data/samples/marida-smoke/` | CC BY 4.0; credit Kikaki et al. 2022 |
| Sentinel-2 L2A images of Taiwan | [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a): free, no account; `scripts/fetch_taiwan_s2.py` | Copernicus open data, credit required |
| A published, pretrained debris model: UNet++, a deep image-segmentation network ([Rußwurm et al. 2023](https://arxiv.org/abs/2307.02465)) | [marinedebrisdetector](https://github.com/marccoru/marinedebrisdetector); its 655 MB weights download fine | MIT |
| [MADOS](https://marine-pollution.github.io/): a second, larger debris benchmark | Zenodo; not downloaded | CC BY 4.0 per the catalog |

**Code**

| File | What it does | Status |
|---|---|---|
| `scripts/fetch_starter_data.py` | Downloads MARIDA, plus the MOENV archive for Path C and the [Knaeps](https://data.4tu.nl/articles/_/12896312/2) plastic spectra | MARIDA works; the Knaeps server returned HTTP 504 |
| `scripts/prepare_marida.py` | Converts MARIDA's ZIP into the repo format, keeping the official splits | Run on the full set |
| `scripts/pixel_rf.py` | Random forest that classifies each MARIDA pixel from its band values alone | Run |
| `scripts/fetch_taiwan_s2.py` | Downloads a cloud-free Sentinel-2 image of Taiwan's coast (edit `BBOX`/`DATES`); computes FDI and [NDVI](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index) (a vegetation index); writes PNG previews plus a 12-band GeoTIFF (an image file with map coordinates) | Run |
| `scripts/predict_taiwan_rf.py` | Applies the MARIDA random forest to that image (water only, 50 m coastal buffer) | Run |
| `python -m waste_lab train` + `configs/marida-smoke.json`, `configs/marine-template.json` | The repo's small segmentation CNN (one output head per dataset, optional NDVI auxiliary head) | Run on 6 patches only; `marine-template.json` expects a MADOS head too |

**Get started** (all on CPU)

```bash
python scripts/fetch_taiwan_s2.py .
python scripts/fetch_starter_data.py      # MARIDA, 1.16 GB; a Knaeps HTTP 504 error at the end is a known issue and doesn't affect MARIDA
python scripts/prepare_marida.py data/raw/marida/MARIDA.zip --out data/processed/marida --limit-per-split 0
python scripts/pixel_rf.py data/processed/marida/manifest.jsonl --head marida --out runs/marida-pixel-rf
python scripts/predict_taiwan_rf.py runs/marida-pixel-rf/model.joblib data/raw/sentinel2/ne-coast-2025-03-25.tif runs/taiwan-ne-coast
```

**To-dos**
- [ ] A spectral correction option in `predict_taiwan_rf.py`, to test the glint idea
- [ ] A coastline buffer for the FDI hotspots in `fetch_taiwan_s2.py`
- [ ] Run the pretrained UNet++ on the Taiwan image (Kaggle)
- [ ] Per-class precision and recall in `pixel_rf.py`'s metrics (currently F1 and IoU only)
- [ ] A MADOS preparation script (later)

---

## Path C — Dump sites (MOENV)

**What we know so far:** the [MOENV dump-site list](https://data.gov.tw/en/datasets/145968) snapshot from 2026-09-09 has 491 sites, each a single map point. Each record has a county, district, land-parcel numbers, waste type (`一般` general 435, `一般含有害` general with hazardous 39, `有害` hazardous 16), a last-update date and free-text inspection notes (`現場狀態`). There's **no size field**.
- A keyword count over the notes finds buried (掩埋) in 29, overgrown (雜草) in 109, piled up (堆置) in 52 and cleared in 20.
- 474 records were updated June–September 2026, so the date says when the record changed, not when the dumping happened.

**Data**

| Data | Where | License / terms |
|---|---|---|
| MOENV dump sites, 2026-09-09 | `data/samples/moenv-20260909/`: `sites.geojson` (map points), blank `site-review.csv`, original `source.zip` | Open Government Data License; credit the source and snapshot date |
| Newer MOENV snapshots | `python -m waste_lab moenv-index` / `moenv-fetch` (needs a free API key in `MOENV_API_KEY`) | Same |
| Sentinel-2 L2A images | [Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a), free | Copernicus open data |
| Other Taiwan lists (legal landfills, Tainan dumps, contaminated sites) | Listed in `waste_lab/catalog.json` | Check each portal |
| [AerialWaste](https://aerialwaste.org/): labelled 20–50 cm aerial photos of Italian dump sites | [Zenodo 7034381](https://zenodo.org/records/7034381) (v3); tools at [nahitorres/AerialWaste](https://github.com/nahitorres/AerialWaste) | CC BY-NC-ND 4.0 (credit, non-commercial, no modified copies). Google-sourced tiles also fall under Google's terms, which forbid ML training |

**Code**

| File | What it does | Status |
|---|---|---|
| `scripts/prepare_moenv.py` | Converts a MOENV archive (a database file in Big5, an older Chinese text encoding) into GeoJSON plus a review CSV; flags garbled text | Run on the 2026-09-09 snapshot |
| `python -m waste_lab search-sentinel` | Finds Sentinel-2 images for a map box and date range | Run |
| `python -m waste_lab moenv-index` / `moenv-fetch` | Lists and downloads MOENV archives | Written; needs a key |
| `scripts/fetch_taiwan_s2.py` | A template for downloading image chips around a site | Run (for Path B's image) |
| `python -m waste_lab train` with `"task": "image"`, or `baseline --kind rf` | Patch classifiers, once you have visibility labels | `train` tested on toy images; `baseline` tested on toy spectra only |

**Get started**

```bash
# 1. Open data/samples/moenv-20260909/sites.geojson in QGIS (https://qgis.org) or geojson.io, and read the 現場狀態 notes
# 2. Find images around a site (this is site 20, Tainan); about 10 come back for Jun–Sep 2026
python -m waste_lab search-sentinel --bbox 120.16 23.06 120.20 23.09 --dates 2026-06-01/2026-09-30 --out data/raw/moenv-site20-scenes.json
# 3. Optional: a newer snapshot
export MOENV_API_KEY=...        # from a free account on data.moenv.gov.tw; never commit it
python -m waste_lab moenv-index
python -m waste_lab moenv-fetch data/raw/moenv/index.json EXACT_ARCHIVE_FILENAME
python scripts/prepare_moenv.py data/raw/moenv/EXACT_ARCHIVE_FILENAME --out data/processed/taiwan-sites
```

**To-dos**
- [ ] A script that sorts the records by their notes (buried / piled / cleared …), with a hand-checked sample
- [ ] A chip downloader: `fetch_taiwan_s2.py` generalized to take a site list, map box and dates as arguments
- [ ] The visibility review itself: 30–50 sites in `site-review.csv`

---

## Path D — Detecting chemicals: air pollution (Sentinel-5P)

**What we know so far:** no experiments yet, only access checks.
- [Sentinel-5P](https://sentiwiki.copernicus.eu/web/s5p-mission) Level-2 data (gas amounts already calculated from the raw readings) is free on [Planetary Computer](https://planetarycomputer.microsoft.com/dataset/sentinel-5p-l2-netcdf) from 2018-04-30. Products: `no2`, `so2`, `co` (carbon monoxide), `o3` (ozone), `hcho` (formaldehyde), `ch4` (methane), aerosol index and clouds. A search over Taiwan returns daily NO₂ products.
- [MOENV's open-data API](https://data.moenv.gov.tw/en) answers and needs a free key.
- Civil IoT Taiwan runs a network of low-cost air sensors (AirBox). The API address in the research trail served a certificate for the wrong hostname.
- The research notes found no public satellite methane plume over Taiwan, so NO₂ and SO₂ are the better targets.

**Data**

| Data | Where | License / terms |
|---|---|---|
| Sentinel-5P / [TROPOMI](https://www.tropomi.eu/) gas amounts | Planetary Computer, collection `sentinel-5p-l2-netcdf`, as [NetCDF](https://unidata.ucar.edu/software/netcdf/) files (a common format for gridded science data) | Copernicus open data |
| Sentinel-3 aerosol optical depth (how much haze and particles dim the sunlight) | Planetary Computer, `sentinel-3-synergy-aod-l2-netcdf` (from 2020-04-16) | Copernicus open data |
| MOENV ground stations (hourly) | [MOENV open-data API](https://data.moenv.gov.tw/en), e.g. dataset `aqx_p_432` (real-time air-quality index); find the hourly history dataset on the portal. Needs `MOENV_API_KEY` | Open Government Data License |
| Civil IoT / AirBox low-cost PM2.5 (fine particle) sensors | Find the current API on Civil IoT Taiwan's site | Check terms |
| Weather, e.g. [ERA5](https://cds.climate.copernicus.eu/) (global weather reanalysis) or Taiwan's [Central Weather Administration](https://www.cwa.gov.tw/) | Free; access not checked | Check terms |

**Code:** none yet; Path D starts from a blank page. scikit-learn and PyTorch are already dependencies. You'll also need `pip install xarray netCDF4` to read the NetCDF files.

**Get started**

```python
# Find Sentinel-5P NO2 products over Taiwan, using Planetary Computer's STAC search API (standard library only)
import json, urllib.request
body = {"collections": ["sentinel-5p-l2-netcdf"], "bbox": [119.3, 21.8, 122.1, 25.4],
        "datetime": "2026-10-01/2026-10-03",
        "query": {"s5p:product_name": {"eq": "no2"}, "s5p:processing_mode": {"eq": "OFFL"}}}  # OFFL = quality-checked offline processing
req = urllib.request.Request("https://planetarycomputer.microsoft.com/api/stac/v1/search",
                             data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
for item in json.load(urllib.request.urlopen(req))["features"]:
    print(item["id"], item["properties"]["datetime"])
```

To download a file, request a read token from `https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-5p-l2-netcdf` and append it to the asset URL. `scripts/fetch_taiwan_s2.py` does the same for Sentinel-2.

**To-dos**
- [ ] A Sentinel-5P fetch script for Taiwan (download, filter by the quality flag, average to a grid)
- [ ] A MOENV station loader (hourly pollutants plus station coordinates)
- [ ] Find a working Civil IoT / AirBox API

---

## Shared foundation

### Repo layout

| Path | What it is |
|---|---|
| `waste_lab/` | Python package: data loading, a small PyTorch model, training and evaluation, download helpers. Run it as `python -m waste_lab <command>`. |
| `scripts/` | Stand-alone scripts, listed under each path above |
| `configs/` | JSON configs for `waste_lab train` (MARIDA runs plus toy examples) |
| `data/samples/`, `data/snapshots/`, `data/templates/` | Small real data, dataset metadata records, blank CSVs |
| `data/raw/`, `data/processed/`, `runs/` | Git-ignored: downloads, prepared data, run outputs |
| `tests/` | 8 unit tests |
| `notebooks/hosted-starter.ipynb` | A thin Colab/Kaggle launcher |
| `docs/` | Paths, log, setup, dataset notes, validation record (see [Docs](#docs)) |
| `research/` | The original research notes. Useful leads, but **unverified** (see `research/README.md`) |

### The `waste_lab` package

| Module | What it does |
|---|---|
| `models.py` | `SmallModel`: a deliberately small CNN with one shared encoder and one output head per dataset, for 1-D spectra, image classification or pixel segmentation. Segmentation can add an auxiliary head that predicts NDVI. |
| `train.py` | AdamW training; normalization from the training split only; `-1` labels ignored; `best.pt` chosen by validation macro-F1; `last.pt` for `--resume`; per-epoch `history.jsonl`. Also `evaluate` and `predict` (with timing). |
| `data.py` | Manifest checks (no file or group in two splits, labels in range); `toy` synthetic data |
| `baseline.py` | CPU SVM or random-forest baselines on spectra or per-image summary features |
| `prepare.py` | Converts a wide spectra CSV, or a GeoTIFF + mask, into the repo format |
| `download.py` | The 48-dataset catalog (`catalog.json`), capped and checksummed downloads, Zenodo listing, MOENV archive index/fetch |
| `taiwan.py` | Sentinel-2 image search on Planetary Computer |

| Command | What it does |
|---|---|
| `catalog`, `toy` | Print the dataset catalog; make synthetic test data |
| `validate MANIFEST CONFIG` | Check a manifest against a config |
| `train MANIFEST CONFIG --out DIR` | Train `SmallModel` (`--device auto` uses CUDA if available) |
| `evaluate CHECKPOINT MANIFEST` | Score on test (or `--split val`) |
| `predict CHECKPOINT INPUT.npz --head H --out FILE` | Predict one sample and time it |
| `baseline MANIFEST --head H --kind svm\|rf --out DIR` | SVM or random-forest baseline |
| `zenodo`, `moenv-index`, `moenv-fetch` | Dataset downloads |
| `prepare-spectra`, `prepare-geotiff` | Convert source data into the repo format |
| `search-sentinel --bbox W S E N --dates A/B --out FILE` | Find Sentinel-2 images |

### Data format

Each sample is an `.npz` file with float32 `x` and integer `y`: spectra `x=(1,L)` with a class index; images `x=(C,H,W)` with a class index; segmentation `x=(C,H,W)` with `y=(H,W)`, where `-1` means unlabelled. A JSONL manifest lists the samples: `{"path": "patch.npz", "group": "sentinel2:date:tile", "split": "train", "dataset": "marida"}`. A `group` (one scene, flight or site) can only appear in one split, and `waste_lab validate` enforces that.

### Parked: spectra

The spectra code (`prepare-spectra`, `configs/toy-spectra.json`, the 1-D CNN and SVM) stays in case a Taiwan spectral dataset or an instrument turns up. Identifying what plastics are made of needs a spectrometer (e.g. Raman). Satellites like Sentinel-2 can't do it, and there are no joint Raman + image models to borrow. For chemistry without a lab, see Path D.

## Instructor / admin to-dos

- [ ] Add the student as a collaborator (or have them fork)
- [ ] MDImageNet access (Taiwan coastal photos, 34 litter types): a real material breakdown for Path A
- [ ] Request the Society of Wilderness 121-station beach survey table: ground truth for Path B

## Docs

| File | Read it when |
|---|---|
| [`docs/PATHS.md`](docs/PATHS.md) | You're choosing a path: ways in, open questions, model menus, ground rules, glossary |
| [`docs/LOG.md`](docs/LOG.md) | You want to know what's been tried. Add to it as you go. |
| [`docs/STUDENT_START.md`](docs/STUDENT_START.md) | You're setting up Kaggle, Colab or your laptop |
| [`docs/DATASETS.md`](docs/DATASETS.md) | You're downloading or preparing a dataset |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | You want the original 2026-11-09 setup checks |

## Limits worth remembering

- **Claim only what you measured.** A `bottle` box is a PET guess, not chemistry. A flagged satellite pixel is "something floating," not "plastic." Sentinel-5P measures the gas in a whole column of air, not the air at ground level.
- **Weak labels stay weak.** Being on the government dump list doesn't make a site visible waste. Beach-cleanup tonnage depends on cleanup effort and isn't a pixel label.
- **Licenses:** each path's data table lists its terms. Other datasets: check `waste_lab/catalog.json` and `THIRD_PARTY.md`.

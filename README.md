# Taiwan Waste Lab

A student research repo with two goals in Taiwan: **detecting waste**, and **detecting chemicals: air pollution**. It uses free drone and satellite data and models trained elsewhere (transfer learning). Everything trains on free hosted GPUs (Kaggle or Colab) and runs inference on a laptop CPU.

This is a research starter, not finished software. The code that's here runs and is tested. Nothing in it has yet been shown to work on Taiwan; finding out what does is the project. There are four research paths. Pick one, combine them, or propose your own ([`docs/PATHS.md`](docs/PATHS.md) has guidance for each).

## The four paths

| Path | The question | Taiwan data | Main tools | Status |
|---|---|---|---|---|
| [**A. Drone litter**](#path-a--drone-litter-haida) | How well does a pretrained detector learn Taiwan's coastal litter from about 1,700 drone frames, and what transfers? | HAIDA (NTOU), 18 drone flights | YOLO on Kaggle | Data inspected; converter still to write |
| [**B. Satellite debris**](#path-b--satellite-debris-sentinel-2) | A model trained on MARIDA calls most of Taiwan's clear sea "Clouds." Why, and can you fix it? | Free Sentinel-2 images of Taiwan's coast | Random forest → deep networks | First Taiwan run done; it fails in an informative way |
| [**C. Dump sites**](#path-c--dump-sites-moenv) | Can Taiwan's reported illegal dump sites actually be seen from space? | MOENV's list of 491 sites | Visual review, then classifiers | Data in the repo; review not started |
| [**D. Detecting chemicals: air pollution**](#path-d--detecting-chemicals-air-pollution-sentinel-5p) | How well does a satellite spectrometer match ground stations on gases like NO₂? Chemistry without a lab | Sentinel-5P + MOENV air stations | Tabular ML (regression, trees) | Data access checked; no code yet |

- **A.** Taiwan drone photos with boxes around trash, `bottle` or `garbage`. It's the most direct transfer-learning project: start from a detector that knows everyday objects, teach it Taiwan litter, and test it on flights it never saw.
- **B.** Floating debris in 10 m satellite pixels. A model trained on the open MARIDA benchmark doesn't transfer to Taiwan. Taiwan's water looks about twice as bright in the infrared, and figuring out why (sun glint? different image processing?) is the research.
- **C.** MOENV publishes reported dump sites, but many of its notes describe buried, overgrown or cleared waste. Before anyone builds a detector, someone has to find out which sites are visible at all.
- **D.** The one path with real chemical detection. Sentinel-5P measures gases in the air from orbit, and Taiwan has dense ground stations to check it against. It's mostly tabular modelling, like the shelter project.

## Getting started (all paths)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[geo]'                             # add ,aerial for Path A
python -m unittest discover -s tests -v             # 8 tests should pass
```

For Kaggle and Colab (GPU, uploading data, saving outputs), see [`docs/STUDENT_START.md`](docs/STUDENT_START.md). Keep a [research log](docs/LOG.md) as you go.

---

## Path A — Drone litter (HAIDA)

**What we know so far** ([log](docs/LOG.md)): HAIDA and HAIDA2 download fine. Together they have 1,710 frames from 18 flights and 8,037 boxes (`garbage` 5,501, `bottle` 2,536; HAIDA2 is almost all `garbage`). The labels are Pascal VOC XML. Frames are cut from videos, so neighbouring frames are near-copies: split by flight. The litter is tiny, with a median box of 0.2% of the frame (about 90×45 px in 1920×1080).

**Data**

| Data | Where | License / terms |
|---|---|---|
| HAIDA (1,319 frames) + HAIDA2 (391 frames) | Google Drive links in the [HAIDA repo](https://github.com/LiaoSteve/HAIDA-Trash-Dataset-High-Resolution-Aerial-image) | The repo's code is MIT; the image terms aren't stated separately. Cite the NTOU thesis and **keep the images out of this repo** (use a private Kaggle Dataset) |
| Optional extra litter sets: UAVVaste, DroneWaste (20 material classes), MDImageNet (Taiwan, 34 classes) | See [PATHS.md, Path A](docs/PATHS.md#models-to-choose-from-path-a) | CC BY 4.0 per the research notes; access not checked |

**Code**

| File | What it does | Status |
|---|---|---|
| `scripts/aerial.py` | Trains or predicts with Ultralytics YOLO on a YOLO-format dataset | Untested on real data; `imgsz` fixed at 640 |
| `scripts/audit_aerial_split.py` | Checks a split CSV: no image twice, no flight in two splits | Untested on real data |
| `data/templates/aerial-split.csv` | Blank split table (`image,group,split`) | Template |

**Get started**

```bash
pip install -e '.[aerial]'
mkdir -p data/raw/haida && bsdtar -xf HAIDA.zip -C data/raw/haida   # PPMd zips: Python's zipfile can't read them; 7-Zip on Windows
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
- A random forest trained on MARIDA scores test macro-F1 0.50 (debris F1 0.69) and takes 23 s on a CPU.
- On a cloud-free Sentinel-2 crop of the Ruifang/Gongliao coast (2025-03-25), it labels **65% of open water "Clouds"** and finds almost no debris.
- Taiwan's water is about twice as bright as MARIDA's in the infrared. Suspects: sun glint, or a processing mismatch (MARIDA used ACOLITE, our crop is Sen2Cor L2A). Both are untested.
- FDI alone flags the coastline and boats, not debris.

**Data**

| Data | Where | License / terms |
|---|---|---|
| MARIDA: 1,381 labelled Sentinel-2 patches (694 train / 328 val / 359 test) | `scripts/fetch_starter_data.py` (1.16 GB, checksum-verified); 6 sample patches ship in `data/samples/marida-smoke/` | CC BY 4.0; credit Kikaki et al. 2022 |
| Sentinel-2 L2A over Taiwan | Microsoft Planetary Computer, free, no account; `scripts/fetch_taiwan_s2.py` | Copernicus open data, credit required |
| Pretrained UNet++ debris model (Rußwurm 2023) | [marinedebrisdetector](https://github.com/marccoru/marinedebrisdetector); its 655 MB weights download fine | MIT |
| MADOS (second debris benchmark) | Zenodo; not downloaded | CC BY 4.0 per the catalog |

**Code**

| File | What it does | Status |
|---|---|---|
| `scripts/fetch_starter_data.py` | Downloads MARIDA (plus the MOENV archive for Path C and the Knaeps spectra) | MARIDA works; the Knaeps server returned HTTP 504 |
| `scripts/prepare_marida.py` | Converts MARIDA's ZIP into the repo format, keeping the official splits | Run on the full set |
| `scripts/pixel_rf.py` | Spectral-only random forest on labelled MARIDA pixels | Run |
| `scripts/fetch_taiwan_s2.py` | Downloads a cloud-free Sentinel-2 crop of Taiwan (edit `BBOX`/`DATES`), computes FDI/NDVI, writes PNGs plus a 12-band GeoTIFF | Run |
| `scripts/predict_taiwan_rf.py` | Applies the MARIDA random forest to that crop (water only, 50 m coastal buffer) | Run |
| `python -m waste_lab train` + `configs/marida-smoke.json`, `configs/marine-template.json` | The repo's small segmentation CNN (one head per dataset, optional NDVI auxiliary head) | Run on 6 patches only; `marine-template.json` expects a MADOS head too |

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
- [ ] Run the pretrained UNet++ on the Taiwan crop (Kaggle)
- [ ] Per-class precision and recall in `pixel_rf.py`'s metrics (currently F1 and IoU only)
- [ ] A MADOS preparation script (later)

---

## Path C — Dump sites (MOENV)

**What we know so far:** the 2026-09-09 snapshot has 491 sites as points: county, district, land-parcel numbers, waste type (`一般` 435, `一般含有害` 39, `有害` 16), a last-update date and free-text inspection notes (`現場狀態`). There's **no size field**. A keyword count over the notes finds buried (掩埋) in 29, overgrown (雜草) in 109, piled (堆置) in 52 and cleared in 20. 474 records were updated June–September 2026, so the date says when the record changed, not when the dumping happened.

**Data**

| Data | Where | License / terms |
|---|---|---|
| MOENV dump sites, 2026-09-09 | `data/samples/moenv-20260909/`: `sites.geojson`, blank `site-review.csv`, original `source.zip` | Open Government Data License; credit the source and snapshot date |
| Newer MOENV snapshots | `python -m waste_lab moenv-index` / `moenv-fetch` (needs `MOENV_API_KEY`) | Same |
| Sentinel-2 L2A | Planetary Computer, free | Copernicus open data |
| Other Taiwan lists (legal landfills, Tainan dumps, contaminated sites) | Listed in `waste_lab/catalog.json` | Check each portal |
| AerialWaste: labelled 20–50 cm aerial tiles of Italian dumps | [Zenodo 7034381](https://zenodo.org/records/7034381) (v3), tools at [nahitorres/AerialWaste](https://github.com/nahitorres/AerialWaste) | CC BY-NC-ND 4.0; Google-sourced tiles also fall under Google's terms, which forbid ML training |

**Code**

| File | What it does | Status |
|---|---|---|
| `scripts/prepare_moenv.py` | Converts a MOENV archive (Big5 DBF) into GeoJSON plus a review CSV; flags garbled text | Run on the 2026-09-09 snapshot |
| `python -m waste_lab search-sentinel` | Finds Sentinel-2 scenes for a bounding box and dates | Run |
| `python -m waste_lab moenv-index` / `moenv-fetch` | Lists and downloads MOENV archives | Written; needs a key |
| `scripts/fetch_taiwan_s2.py` | A template for downloading image chips around a site | Run (for Path B's crop) |
| `python -m waste_lab train` with `"task": "image"`, or `baseline --kind rf` | Patch classifiers, once you have visibility labels | `train` tested on toy images; `baseline` tested on toy spectra only |

**Get started**

```bash
# 1. Open data/samples/moenv-20260909/sites.geojson in QGIS (or geojson.io) and read the 現場狀態 notes
# 2. Find scenes around a site (this is site 20, Tainan); about 10 come back for Jun–Sep 2026
python -m waste_lab search-sentinel --bbox 120.16 23.06 120.20 23.09 --dates 2026-06-01/2026-09-30 --out data/raw/moenv-site20-scenes.json
# 3. Optional: a newer snapshot
export MOENV_API_KEY=...        # from a free MOENV developer account; never commit it
python -m waste_lab moenv-index
python -m waste_lab moenv-fetch data/raw/moenv/index.json EXACT_ARCHIVE_FILENAME
python scripts/prepare_moenv.py data/raw/moenv/EXACT_ARCHIVE_FILENAME --out data/processed/taiwan-sites
```

**To-dos**
- [ ] A script that sorts the records by their notes (buried / piled / cleared …), with a hand-checked sample
- [ ] A chip downloader: `fetch_taiwan_s2.py` generalized to take a site list, bounding box and dates as arguments
- [ ] The visibility review itself: 30–50 sites in `site-review.csv`

---

## Path D — Detecting chemicals: air pollution (Sentinel-5P)

**What we know so far:** no experiments yet, only access checks.
- Sentinel-5P Level-2 is free on Planetary Computer from 2018-04-30, with products `no2`, `so2`, `co`, `o3`, `hcho`, `ch4`, aerosol index and clouds. A search over Taiwan returns daily NO₂ products.
- MOENV's air-quality API answers and needs a free key.
- The Civil IoT / AirBox address in the research trail served a certificate for the wrong hostname.
- The research notes found no public satellite methane plume over Taiwan, so NO₂ and SO₂ are the better targets.

**Data**

| Data | Where | License / terms |
|---|---|---|
| Sentinel-5P / TROPOMI gas columns | Planetary Computer, collection `sentinel-5p-l2-netcdf` (NetCDF files) | Copernicus open data |
| Sentinel-3 aerosol optical depth | Planetary Computer, `sentinel-3-synergy-aod-l2-netcdf` (from 2020-04-16) | Copernicus open data |
| MOENV ground stations (hourly) | MOENV open-data API, e.g. `aqx_p_432` (real-time AQI); find the hourly history dataset on the portal. Needs `MOENV_API_KEY` | Open Government Data License |
| Civil IoT / AirBox low-cost PM2.5 sensors | Find the current API on Civil IoT Taiwan's site | Check terms |
| Weather (e.g. ERA5, Taiwan's Central Weather Administration) | Free; access not checked | Check terms |

**Code:** none yet. Path D starts from a blank page. scikit-learn and PyTorch are already dependencies; you'll also need `pip install xarray netCDF4` to read the NetCDF files.

**Get started**

```python
# Find Sentinel-5P NO2 products over Taiwan (standard library only)
import json, urllib.request
body = {"collections": ["sentinel-5p-l2-netcdf"], "bbox": [119.3, 21.8, 122.1, 25.4],
        "datetime": "2026-10-01/2026-10-03",
        "query": {"s5p:product_name": {"eq": "no2"}, "s5p:processing_mode": {"eq": "OFFL"}}}
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
| `taiwan.py` | Sentinel-2 scene search on Planetary Computer |

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
| `search-sentinel --bbox W S E N --dates A/B --out FILE` | Find Sentinel-2 scenes |

### Data format

Each sample is an `.npz` file with float32 `x` and integer `y`: spectra `x=(1,L)` with a class index; images `x=(C,H,W)` with a class index; segmentation `x=(C,H,W)` with `y=(H,W)`, where `-1` means unlabelled. A JSONL manifest lists the samples: `{"path": "patch.npz", "group": "sentinel2:date:tile", "split": "train", "dataset": "marida"}`. A `group` (one scene, flight or site) can only appear in one split, and `waste_lab validate` enforces that.

### Parked: spectra

The spectra code (`prepare-spectra`, `configs/toy-spectra.json`, the 1-D CNN and SVM) stays in case a Taiwan spectral dataset or an instrument turns up. Identifying plastics chemically needs a spectrometer. Satellites like Sentinel-2 can't do it, and there are no joint Raman + image models to borrow. For chemistry without a lab, see Path D.

## Instructor / admin to-dos

- [ ] Add the student as a collaborator (or have them fork)
- [ ] MDImageNet access (Taiwan coastal set, 34 litter classes): a real material breakdown for Path A
- [ ] Request the Society of Wilderness 121-station beach survey table: ground truth for Path B

## Docs

| File | Read it when |
|---|---|
| [`docs/PATHS.md`](docs/PATHS.md) | You're choosing a path: ways in, open questions, model menus, ground rules |
| [`docs/LOG.md`](docs/LOG.md) | You want to know what's been tried. Add to it as you go. |
| [`docs/STUDENT_START.md`](docs/STUDENT_START.md) | You're setting up Kaggle, Colab or your laptop |
| [`docs/DATASETS.md`](docs/DATASETS.md) | You're downloading or preparing a dataset |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | You want the original 2026-11-09 setup checks |

## Limits worth remembering

- **Claim only what you measured.** A `bottle` box is a PET guess, not chemistry. A flagged satellite pixel is "something floating," not "plastic." Sentinel-5P measures a gas column, not ground-level air.
- **Weak labels stay weak.** Being on the government dump list doesn't make a site visible waste. Beach-cleanup tonnage depends on cleanup effort and isn't a pixel label.
- **Licenses:** each path's data table lists its terms. Other datasets: check `waste_lab/catalog.json` and `THIRD_PARTY.md`.

# Taiwan Waste Lab

Code and data for detecting waste in Taiwan from **drone photos** and **free satellite images**, using datasets and models built elsewhere (transfer learning). Everything trains on free hosted GPUs (Kaggle or Colab) and runs inference on a laptop CPU.

This is a research starter, not finished software. The code runs and is tested. Nothing in it has yet been shown to detect waste in Taiwan. Finding out what works is the project. The open questions are in [To-dos](#to-dos) and [`docs/PATHS.md`](docs/PATHS.md).

## What's in the repo

| Path | What it is |
|---|---|
| `waste_lab/` | Python package: data loading, a small PyTorch model, training and evaluation, download helpers. Run it as `python -m waste_lab <command>`. |
| `scripts/` | Stand-alone scripts for specific datasets (MARIDA, MOENV, Sentinel-2 over Taiwan, drone detection with YOLO). |
| `configs/` | JSON configs for `waste_lab train`: real MARIDA runs plus toy examples. |
| `data/samples/` | Small real data that ships with the repo: 491 Taiwan dump-site records, 6 MARIDA patches, a synthetic spectra file. |
| `data/snapshots/` | Metadata records for MARIDA, MADOS, Knaeps and a Sentinel-2 search. These are metadata only, not imagery. |
| `data/templates/` | Blank CSVs for a drone train/test split and a dump-site review. |
| `data/raw/`, `data/processed/`, `runs/` | Git-ignored. Downloads, prepared data and run outputs go here. |
| `tests/` | 8 unit tests (`python -m unittest discover -s tests -v`). |
| `notebooks/hosted-starter.ipynb` | A thin launcher for Colab/Kaggle: installs the package and runs a toy job. |
| `docs/` | Research paths, the research log, setup, dataset notes, validation record. See [Docs](#docs). |
| `research/` | The original research notes that started this project. They're useful leads, but **unverified**; see `research/README.md`. |

### The `waste_lab` package

| Module | What it does |
|---|---|
| `models.py` | `SmallModel`: a deliberately small CNN with one **shared encoder** and **one output head per dataset**. It handles three task shapes: 1-D spectra, image classification and pixel segmentation. Segmentation can add an **auxiliary head** that predicts NDVI from the image (`aux_weight`, `red_index`, `nir_index` in the config). |
| `train.py` | Training with AdamW. Normalization statistics come from the training split only. Pixels labelled `-1` are ignored. `best.pt` is chosen by validation macro-F1, `last.pt` supports `--resume`, and every epoch is logged to `history.jsonl`. Also contains `evaluate` (val or test split) and `predict` (one sample, with warm CPU/GPU timing). |
| `data.py` | Reads and checks manifests: no file in two splits, no group in two splits, labels within each head's class range. `toy` generates synthetic data for every task shape. |
| `baseline.py` | CPU baselines (SVM or random forest) for spectra or per-image summary features. |
| `prepare.py` | Converts a wide spectra CSV, or a GeoTIFF image + mask, into the repo's format (explicit bands, scale and label map). |
| `download.py` | Dataset catalog (`catalog.json`, 48 candidate datasets with license notes), checksum- and size-capped downloads, Zenodo file listing, MOENV archive index/fetch. |
| `taiwan.py` | Searches Microsoft Planetary Computer's free Sentinel-2 catalog for scenes in a bounding box (first page of results only). |
| `cli.py` | The commands below. |

| Command | What it does |
|---|---|
| `catalog` | Print the 48-dataset catalog |
| `toy` | Make synthetic data to test the pipeline |
| `validate MANIFEST CONFIG` | Check a manifest against a config before training |
| `train MANIFEST CONFIG --out DIR` | Train `SmallModel` (`--device auto` uses CUDA if available) |
| `evaluate CHECKPOINT MANIFEST` | Score a checkpoint on the test (or `--split val`) split |
| `predict CHECKPOINT INPUT.npz --head H --out FILE` | Predict one sample and time it |
| `baseline MANIFEST --head H --kind svm\|rf --out DIR` | Train an SVM or random-forest baseline |
| `zenodo DATASET [--file F --accept-terms]` | List or download files from a Zenodo record |
| `moenv-index` / `moenv-fetch` | List and download MOENV dump-site archives (needs a MOENV API key in your environment) |
| `prepare-spectra` / `prepare-geotiff` | Convert source data into the repo format |
| `search-sentinel --bbox W S E N --dates A/B --out FILE` | Find Sentinel-2 scenes |

### Scripts

| Script | What it does | Status |
|---|---|---|
| `fetch_starter_data.py` | Downloads MARIDA (1.16 GB, checksum-verified), the MOENV 2026-09-09 archive and the Knaeps spectra | MARIDA and MOENV work; the Knaeps server returned HTTP 504 |
| `prepare_marida.py` | Converts MARIDA's ZIP into the repo format, keeping the official splits (694 train / 328 val / 359 test patches) | Run on the full set |
| `pixel_rf.py` | Spectral-only random forest on labelled MARIDA pixels; writes `metrics.json` and `model.joblib` | Run; results in `docs/LOG.md` |
| `fetch_taiwan_s2.py` | Downloads a cloud-free Sentinel-2 crop of Taiwan's northeast coast, computes FDI/NDVI and writes PNG previews plus a 12-band GeoTIFF | Run |
| `predict_taiwan_rf.py` | Applies the MARIDA random forest to that Taiwan crop: water only, excluding 50 m along the coast | Run; the result is the start of Path B |
| `prepare_moenv.py` | Converts a MOENV dump-site archive (Big5 DBF) into GeoJSON plus a review CSV | Run on the 2026-09-09 snapshot |
| `aerial.py` | Trains or predicts with an Ultralytics YOLO model on a YOLO-format dataset | Untested on real data; image size hard-coded to 640 |
| `audit_aerial_split.py` | Checks a drone split CSV: no image twice, no flight in two splits | Untested on real data (the template is just a header row) |

### Data format

Each sample is an `.npz` file with float32 `x` and integer `y`:

| Task | `x` | `y` |
|---|---|---|
| Spectra | `(1, L)` | class index |
| Image classification | `(C, H, W)` | class index |
| Segmentation | `(C, H, W)` | `(H, W)` class map; `-1` = unlabelled, ignored |

A JSONL manifest lists the samples, one per line: `{"path": "patch.npz", "group": "sentinel2:date:tile", "split": "train", "dataset": "marida"}`. The `group` keeps related samples together. One satellite scene or one drone flight is one group, and a group can only appear in one split. `waste_lab validate` enforces that.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[geo]'                  # add ',aerial' for YOLO
python -m unittest discover -s tests -v  # 8 tests should pass

python -m waste_lab toy                  # synthetic smoke test
python -m waste_lab train data/samples/marida-smoke/manifest.jsonl configs/marida-smoke.json --out runs/marida-smoke --device cpu
python scripts/fetch_taiwan_s2.py .      # a real Taiwan satellite crop, about a minute on CPU
```

For Kaggle/Colab setup, uploading data and saving outputs, see [`docs/STUDENT_START.md`](docs/STUDENT_START.md).

## What we know so far

The details, with numbers, are in [`docs/LOG.md`](docs/LOG.md):

- **HAIDA** (Taiwan drone litter, from NTOU) downloads fine. It has about 1,700 frames from 18 flights, with boxes labelled `garbage` or `bottle`. The litter is tiny, typically 0.2% of the frame.
- **A MARIDA-trained random forest doesn't transfer to Taiwan.** It reaches test macro-F1 0.50 on MARIDA but labels 65% of cloud-free Taiwan sea as "Clouds". Taiwan's water is about twice as bright in the infrared. Suspects: sun glint, or the processing mismatch (MARIDA used ACOLITE, our crop is Sen2Cor L2A). Both are untested.
- **FDI on its own flags the coastline and boats**, not debris, on an ordinary clear day.

## To-dos

### Research: pick a path, or propose your own

These are open questions. [`docs/PATHS.md`](docs/PATHS.md) has guidance for each, plus ideas for your own:

- [ ] **Path A: Taiwan drone litter.** How well does a pretrained detector learn Taiwan's coastal litter from HAIDA, and what transfers?
- [ ] **Path B: Taiwan satellite debris.** Why does the satellite model see clouds over Taiwan's sea, and can you fix it?
- [ ] **Path C: Taiwan dump sites.** Can Taiwan's reported dump sites be seen at 10 m resolution?
- [ ] **Your own path**, agreed with your instructor.

### Code

Good first contributions. Each one unblocks a path:

- [ ] HAIDA Pascal VOC → YOLO converter, with a visual check of the converted boxes (Path A)
- [ ] A script that builds a flight-level HAIDA split into `data/templates/aerial-split.csv` (Path A)
- [ ] An `--imgsz` option for `scripts/aerial.py`, which is hard-coded to 640 (Path A)
- [ ] A spectral correction option in `predict_taiwan_rf.py`, to test the glint idea (Path B)
- [ ] A coastline buffer for the FDI hotspots in `fetch_taiwan_s2.py`; the random-forest script already has one (Path B)
- [ ] Run the published pretrained detector ([marinedebrisdetector](https://github.com/marccoru/marinedebrisdetector)) on the Taiwan crop in Kaggle (Path B)
- [ ] Per-class precision and recall in `pixel_rf.py`'s metrics, which currently has F1 and IoU only (all paths)
- [ ] A MADOS preparation script; `configs/marine-template.json` already expects a `mados` head (Path B, later)

### Instructor / admin

- [ ] GitHub repo visibility, and push
- [ ] MDImageNet access (a Taiwan coastal set with 34 litter classes), which would allow a real material breakdown
- [ ] Request the Society of Wilderness 121-station beach survey table (ground truth for Path B)

## Docs

| File | Read it when |
|---|---|
| [`docs/PATHS.md`](docs/PATHS.md) | You're choosing what to work on |
| [`docs/LOG.md`](docs/LOG.md) | You want to know what's been tried. Add to it as you go. |
| [`docs/STUDENT_START.md`](docs/STUDENT_START.md) | You're setting up Kaggle, Colab or your laptop |
| [`docs/DATASETS.md`](docs/DATASETS.md) | You're downloading or preparing a dataset |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | You want the original 2026-11-09 setup checks |

## Data, licenses and limits

- **Licenses:** MARIDA is CC BY 4.0 (credit Kikaki et al. 2022). MOENV data falls under the Open Government Data License (credit the source and snapshot date). HAIDA images aren't covered by the repo's MIT code license: cite the NTOU thesis and keep the images out of this repo. Other datasets: check `waste_lab/catalog.json` and `THIRD_PARTY.md` before using them.
- **No chemistry from these inputs.** Sentinel-2 can show that something is floating, not what it's made of. A `bottle` box is a reasonable PET guess, not a measurement. Raman/spectral material ID is parked: there's no instrument and no Taiwan spectral library. The spectra code stays in case that changes.
- **Weak labels stay weak.** A site on the government dump list isn't a confirmed visible dump. Beach-cleanup tonnage depends on cleanup effort and isn't a pixel label.

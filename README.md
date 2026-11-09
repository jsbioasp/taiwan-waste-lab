# Taiwan Waste Lab

A student research repository for low-cost waste detection, spectral identification and combined objectives. Train on Kaggle/Colab, run small inference jobs on CPU. No local GPU is required.

## What is included

| Path | Included code | Data readiness |
|---|---|---|
| Raman / reflectance classification | Small 1D CNN, SVM, RF, wide-CSV adapter | Synthetic runnable example; original libraries require per-source conversion |
| Marine debris | Small encoder-decoder, dataset-specific heads, optional NDVI auxiliary loss | GeoTIFF adapter; MARIDA/MADOS catalog + selective Zenodo download |
| Taiwan dumpsites | MOENV index/archive downloader, Sentinel-2 scene search, site-review template, image classifier | Requires dated imagery and manual visible-waste labels |
| Aerial litter | Ultralytics train/predict runner and group-split audit | HAIDA download references; annotation conversion required |
| Methane / hyperspectral | Source and upstream-code references | Research extension, not an implemented methane detector |

This is a runnable experiment foundation, not pretrained production software or a claim to reproduce published architectures. The catalog covers candidate datasets from both discussions. `models/catalog.json` distinguishes implemented baselines, optional runners and upstream model references. Large originals remain in ignored `data/raw/`; access-restricted resources have source links and explicit prerequisites.

## Real data already included

- `data/samples/moenv-20260909/`: 491 public Taiwan site records as GeoJSON and a blank image-review CSV, with source provenance and flagged text-decoding issues.
- `data/samples/marida-smoke/`: six attributed real MARIDA patches (two per official split), prepared for the training code. This deliberately tiny subset is a software test, not a benchmark.
- `data/raw/marida/MARIDA.zip`: the complete 1.16 GB archive downloaded locally and checksum-verified. It is excluded from Git and the sharing ZIP; the fetch script reproduces it.
- `data/snapshots/`: original metadata for MARIDA, MADOS and the Knaeps library, plus the setup download-status record. The Knaeps file requests failed with HTTP 504; no spectral library download is claimed.

Run the real-image software check after installation:

```bash
python -m waste_lab train data/samples/marida-smoke/manifest.jsonl configs/marida-smoke.json --out runs/my-marida-check --device cpu
python scripts/pixel_rf.py data/samples/marida-smoke/manifest.jsonl --head marida --out runs/my-rf-check
```

## Start for free

Use Python 3.10+ (3.11/3.12 recommended in hosted environments).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m waste_lab catalog
python -m waste_lab toy
python -m waste_lab train data/processed/toy/spectra/manifest.jsonl configs/toy-spectra.json --out runs/spectra --device cpu
python -m waste_lab evaluate runs/spectra/best.pt data/processed/toy/spectra/manifest.jsonl
python -m waste_lab predict runs/spectra/best.pt data/processed/toy/spectra/source_a_test_0.npz --head source_a --out runs/spectra/prediction.npz
python -m waste_lab baseline data/processed/toy/spectra/manifest.jsonl --head source_a --kind svm --out runs/svm
```

Synthetic data only tests code. Its accuracy does not measure chemical identification or waste detection. A tiny synthetic wide CSV is also tracked at `data/samples/synthetic-spectra.csv`.

## Combined objectives

```bash
python -m waste_lab train data/processed/toy/segmentation/manifest.jsonl configs/toy-segmentation.json --out runs/seg-baseline
python -m waste_lab train data/processed/toy/segmentation/manifest.jsonl configs/toy-segmentation-aux.json --out runs/seg-aux
```

Both runs use the same shared encoder and per-dataset class heads. The second also predicts NDVI from image features. For real imagery set verified red/NIR channel indices after band harmonization; the toy channels are synthetic. NDVI is derivable from the input and adds no new chemical ground truth. Test whether its supervision helps, with an alpha sweep including zero and a shuffled-target control before interpreting a gain. Dataset heads allow different class vocabularies, but inputs must represent the same physical bands/axis.

Training uses train-only channel normalization, ignores labels `-1`, selects checkpoints using validation macro-F1, and leaves test evaluation to a separate command. For a single-source comparison, filter the manifest to that source and remove the other head from the config. For pooled training, harmonize labels first and rename both dataset IDs to a common head. Preserve globally identical group IDs for overlapping scenes across libraries.

## Actual datasets

```bash
# Prints exact filenames, sizes, checksums and current record license; downloads nothing.
python -m waste_lab zenodo marida
python -m waste_lab zenodo mados
# Substitute one exact filename from the listing, after reviewing its terms.
python -m waste_lab zenodo marida --file EXACT_FILENAME --max-gb 2 --accept-terms
# MOENV developer key belongs in the environment, never in Git.
python -m waste_lab moenv-index
python -m waste_lab moenv-fetch data/raw/moenv/index.json EXACT_ARCHIVE_FILENAME
# First-page scene discovery only; does not fetch full satellite scenes.
python -m waste_lab search-sentinel --bbox 121.7 25.0 121.9 25.2 --dates 2025-01-01/2025-12-31 --out data/raw/taiwan-scenes.json
```

See `docs/DATASETS.md` for acquisition and preparation. `data/snapshots/` contains any small real-data source indexes acquired during setup; it does not contain labelled satellite imagery.

## Real input contract

Each NPZ has float32 `x` and integer `y`, loaded without pickle:

- Spectra: `x=(1,L)`, `y=scalar class index`.
- Image classification: `x=(C,H,W)`, `y=scalar class index`.
- Segmentation: `x=(C,H,W)`, `y=(H,W)`; `-1` is unlabelled/ignored.
- Class indices start at zero and must fit that dataset's head.
- All examples in a run share input shape and physical channel/axis order.

A JSONL manifest has one object per sample: `{"path":"patch.npz","group":"sentinel2:date:tile","split":"train","dataset":"marida"}`. Paths are relative to the manifest. A group may occur in only one split, even across datasets. Keep repeated spectra from the same physical sample, adjacent drone frames from the same flight/site, and repeated satellite observations of the same held-out site together. Preserve official benchmark splits where supplied; document any stronger geographic split separately.

```bash
python -m waste_lab prepare-spectra data/samples/synthetic-spectra.csv --out data/processed/csv-demo
# GeoTIFF adapter needs rasterio; replace every example path and verify bands/scale/classes.
python -m pip install -e '.[geo]'
python -m waste_lab prepare-geotiff configs/geotiff-example.json --out data/processed/marine
python -m waste_lab validate data/processed/marine/manifest.jsonl configs/marine-template.json
```

## Colab / Kaggle

Upload the repo ZIP or clone your own remote, change into its directory, install with `pip install -e .`, then use the commands above. `notebooks/hosted-starter.ipynb` is a thin launcher. Select a GPU only for training; `--device auto` uses CUDA when available. Begin with batch 4, mixed precision, and small patches. Checkpoint to persistent mounted storage (`--out`) and copy data to runtime-local disk for training speed. To resume, increase `epochs` in the same config and pass `--resume /persistent/run/last.pt`. Resume restores optimizer, scaler and Python/Torch RNG state; exact cross-hardware determinism is not guaranteed.

Free compute quotas change. Kaggle: https://www.kaggle.com/docs/efficient-gpu-usage ; Colab: https://research.google.com/colaboratory/faq.html . No paid service is required for the toy workflow or initial small baselines.

## Validation

```bash
python -m unittest discover -s tests -v
```

Test coverage includes leakage rejection, ignored labels, class mapping, checkpoint/resume, all three task shapes, auxiliary loss, CPU prediction and download integrity/size limits. See `docs/VALIDATION.md` for the actual setup-run results and limitations.

## Scientific limits

Beach cleanup tonnage is effort-dependent, not floating-debris pixel truth. Listed dump coordinates are candidate sites, not verified image positives. Legal landfills can be positive for visible waste while negative for illegal dumping; define the target first. NMHC does not measure methane. Reflectance, FTIR and Raman inputs have different axes and physics. An RGB/satellite detector cannot supply a missing Raman measurement. Multimodal learning is possible in principle with suitable paired data and separate encoders; the unpaired datasets here do not establish it.

## Source terms

Original starter code is provided for this project; no blanket license grant is inferred for third-party datasets, weights or copied research notes. See `THIRD_PARTY.md`. The historical notes are preserved in `research/` as unverified research artifacts, with corrections in `research/README.md`. No GitHub remote is created automatically.

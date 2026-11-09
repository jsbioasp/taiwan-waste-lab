# Acquisition and preparation

The authoritative machine-readable inventory is `waste_lab/catalog.json`; `python -m waste_lab catalog` works without ML dependencies. Candidate status is distinct from a verified download or tested adapter. Consult original terms before rehosting data. Commercial-license conclusions in archived notes are not legal determinations.

## MARIDA / MADOS

List Zenodo files, selectively download originals with checksums, then inspect the archive. MARIDA upstream: https://github.com/marine-debris/marine-debris.github.io . MADOS upstream: https://github.com/gkakogeorgiou/mados . MADOS may require upstream `utils/stack_patches.py` to align resolutions. Do not concatenate raw bands from different resolutions.

Create a GeoTIFF preparation spec with explicit image and mask paths, dataset head, split, global scene group, one-based input band selection, reflectance scale/offset and source-to-target label map. Source paths resolve relative to the spec. The example maps source 0 to ignore (-1), source 1..15 to 0..14; verify against your exact release. Do not apply MARIDA's label meanings to MADOS. The adapter rejects unknown labels and misaligned CRS/transforms. Use 128 or 256 px patches consistently. Keep original unlabelled pixels ignored; do not call them water. For real inference, apply identical source calibration, band order and patch geometry before model normalization.

## HAIDA

Repository: https://github.com/LiaoSteve/HAIDA-Trash-Dataset-High-Resolution-Aerial-image . Download its original image/annotation archives manually. README identifies Ezlabel annotations; no verified automatic converter is supplied. Inspect a representative annotation before converting boxes to YOLO normalized x-center/y-center/width/height. Check bounds visually. Group frames by flight and site before making train/val/test directories. Record the actual split in `data/templates/aerial-split.csv` and run `python scripts/audit_aerial_split.py YOUR_SPLIT.csv`.

Install `pip install -e '.[aerial]'`. Then `python scripts/aerial.py train --data /absolute/path/data.yaml --device 0` in a hosted GPU runtime. `python scripts/aerial.py predict --model runs/aerial/train/weights/best.pt --source /path/images --device cpu` runs locally. Ultralytics is a third-party dependency with its own terms. Initial weights download on first use. This runner is not a reproduction of HAIDA's published experiment.

## Taiwan dumping sites

Dataset 145968: https://data.gov.tw/en/datasets/145968 . API `wr_p_244` returns an index of dated ZIPs, not site rows directly. `moenv-index` uses your environment key; `moenv-fetch` gets one named archive. Do not embed keys. Inspect archive members and coordinate system; WGS84 archive naming is not a substitute for checking CRS. Update date is not a confirmed event date.

Sample 30–50 sites first. Obtain imagery near the snapshot date, inspect cloud masks and visible waste extent, and fill the review template. Treat unlisted sites as unlabelled until checked. For classification NPZs, use scalar labels based on image review. Do not label a site positive only because it appears on the government list. Repeated dates of the same site stay in the same split.

`search-sentinel` uses public Planetary Computer STAC and stores the first result page. Sign asset URLs using the provider SDK when downloading imagery; honor per-band calibration metadata and scene cloud masks. Whole-country downloading is outside the starter. Small manually selected patches are cheaper and make validation possible.

## Spectra

Wide CSV adapter requires sample_id, group, split, dataset, label, followed by strictly increasing numeric axis columns. Use one common physical axis/unit and grid per experiment. Save the interpolation/calibration decision with the run. Repeated spectra from a specimen share a group; consider holding out instruments or source libraries as a separate domain-shift test. The starter does not infer polymer labels from filenames or translate between Raman and reflectance.

MADLib, Knaeps, OpenSpecy, RRUFF, RamanBench and other sources have different native schemas. Download/import only after inspecting their schema and license; convert explicitly to the common format. Per-class head sizes and dataset names belong in the config. No Taiwan-specific chemical performance claim follows from testing foreign reference spectra.

## Methane and advanced hyperspectral

Use upstream STARCOP https://github.com/spaceml-org/starcop and MethaneMapper source code linked in the catalog. EMIT products require an acquisition/coverage check, and some products require Earthdata authentication. The generic segmentation code can accept prepared arrays, but methane-specific radiative processing, matched filtering, flux estimation and georeferenced output are not implemented. These are separate research extensions, not hidden behind a runnable demo.

## Tested archive adapters

After `pip install -e '.[geo]'`, the included scripts handle the downloaded releases directly:

```bash
python scripts/prepare_marida.py data/raw/marida/MARIDA.zip --out data/processed/marida --limit-per-split 0
python scripts/prepare_moenv.py data/raw/moenv/WDMS-WGS84-20260909.zip --out data/processed/taiwan-sites
```

The MARIDA adapter preserves official splits (694 train, 328 validation, 359 test patches) and verifies global scene groups do not cross them. Original pixel labels are float32 integers; the adapter verifies integrality before converting. Class 0 remains ignored, represented as -1.

The specific MOENV snapshot contains 491 point records in geographic WGS84. Attribute names are Chinese; DBF encoding is Big5. Some 254-byte text fields truncate a multi-byte character. The converter flags fields containing replacement characters instead of silently treating them as exact text; the original ZIP is retained for byte-exact source reference. This does not create image labels.

`scripts/pixel_rf.py` supplies a CPU spectral-only segmentation baseline, sampling at most 1,024 labelled training pixels per patch. This is not MARIDA's published spectral-index/texture RF reproduction. Tune only on validation data and evaluate the test set after choosing the settings.

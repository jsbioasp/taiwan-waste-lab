# Validation — 2026-11-09

Eight automated tests pass in the isolated `.venv`. They cover all three task shapes, multi-head training, NDVI auxiliary training, checkpoint/resume, CPU prediction, train-only normalization, ignored labels, group leakage, class bounds, CSV import, SVM baseline, GeoTIFF alignment and download checksums/size limits.

Actual setup environment: Python 3.14.3, PyTorch 2.15.0, NumPy 2.5.4, scikit-learn 1.9.1, rasterio 1.5.2, pyshp 3.1.7; Apple M4 Pro / arm64, CPU execution with two Torch threads. Exact machine-readable evidence: `validation-results.json`. Dependency ranges remain portable rather than pinning this macOS environment onto Colab.

## Real data exercised

- MARIDA archive: 1,164,612,748 bytes; provider MD5 checked during download and local SHA256 saved. Official split counts: 694 train / 328 validation / 359 test. Scene IDs do not overlap across these splits.
- Six original 11-band 256×256 patches were converted to NPZ with class 0 ignored. The small segmentation network trained for two epochs, saved checkpoints, evaluated held-out test patches and generated predictions. The spectral-only pixel Random Forest also trained and evaluated successfully.
- Warm CPU model-forward latency for one 11×256×256 input: median 10.28 ms, p95 11.19 ms, 3 warmups and 20 repetitions. This excludes loading, preprocessing, output and satellite-scene tiling. It is not a promise for the student's laptop or a complete satellite scene.
- Taiwan MOENV snapshot 2026-09-09: 491 WGS84 point records, converted from Big5 DBF to UTF-8 GeoJSON. Twenty records have source text truncation/decoding replacements, explicitly flagged. Original source bytes are retained as `data/samples/moenv-20260909/source.zip`.
- Live Sentinel-2 STAC search over a small northeast Taiwan bounding box returned two scenes and a next-page indicator. This validates discovery, not signed asset download or image alignment.

## Boundaries

These are software checks. The six-patch subset is not representative enough to compare architecture accuracy; its small training subset lacks important target classes. No Taiwan waste detector or chemical identifier has been scientifically validated.

CUDA/Colab/Kaggle execution, optional YOLO training, HAIDA download/conversion, MADOS preparation and methane processing were not run. Knaeps metadata was retrieved, but three original file URLs returned HTTP 504. MADOS's 4.04 GB archive was cataloged and not downloaded. All remaining candidates need their documented access/schema checks.

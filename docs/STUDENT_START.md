# First session

1. Upload/extract the starter ZIP in Colab or Kaggle. Open `notebooks/hosted-starter.ipynb`, set ROOT to the extracted repo directory and run the install cell.
2. Run the synthetic workflow to learn the commands; its outputs are not research results.
3. Run the real six-patch MARIDA check from README. Inspect x.shape, labels and the prediction map. Explain why most pixels are ignored.
4. Open `data/samples/moenv-20260909/sites.geojson` in QGIS and inspect the blank review CSV. Pick 30–50 sites for an imagery-visibility study. Do not invent positive/negative labels from listing status.
5. Choose one research path after inspecting real inputs. For marine work, prepare the full MARIDA archive with `scripts/prepare_marida.py --limit-per-split 0`, then establish the single-dataset baseline before adding another head/dataset.

Deliver one run folder containing the exact config, epoch history, checkpoint and evaluation JSON. Report the dataset, split, target, seed, device and measured speed. Keep the test split untouched while tuning. Use a persistent output directory on hosted compute and download the trained artifact before the runtime expires.

The six-patch subset supports a quick software demonstration on CPU. Meaningful accuracy work needs the full benchmark and properly held-out Taiwan labels.

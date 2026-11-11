# Setup

Getting the repo running on Kaggle (the main GPU environment), Colab or your laptop. When it works, head to [`PATHS.md`](PATHS.md) to pick what to work on.

## Laptop (CPU)

Python 3.10 or newer.

```bash
git clone <repo URL from your instructor> taiwan-waste-lab
cd taiwan-waste-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[geo]'
python -m unittest discover -s tests -v              # expect 8 passing tests
```

A laptop is enough for the satellite scripts, the random forest, the dump-site review and running a trained detector on a few images. Use Kaggle for anything that trains a neural network for more than a few minutes.

## Kaggle (free GPU)

1. Create an account at kaggle.com. To turn on GPU and internet in notebooks, Kaggle may ask you to verify a phone number. Your weekly GPU allowance shows in your account settings. Switch the GPU on only while training.
2. Create a notebook and set **Accelerator** to a GPU and **Internet** to on.
3. Get the code: `!git clone <repo URL>` (a private repo needs a token), or upload the repo ZIP as a Kaggle Dataset.
4. Install: `%cd /kaggle/working/taiwan-waste-lab` then `%pip install -e '.[geo,aerial]'`.
5. Add data as **Kaggle Datasets** (*Upload → New Dataset*), not to the repo:
   - HAIDA: extract the zips on your laptop first (`bsdtar -xf HAIDA.zip` on a Mac, 7-Zip on Windows; Python's `zipfile` can't read their PPMd compression), then upload the folders. **Make it private.**
   - The Taiwan Sentinel-2 GeoTIFF from `scripts/fetch_taiwan_s2.py`, if you want to use it in Kaggle.
   - Uploaded datasets appear read-only under `/kaggle/input/<name>/`. Copy anything a script needs to write next to into `/kaggle/working/`.
6. **Save your outputs.** Write runs to `/kaggle/working/runs/...`. Before the session ends, download the run folder or save a notebook version, because an interactive session's files can disappear when it stops.

## Colab

Same idea as Kaggle. `notebooks/hosted-starter.ipynb` is a minimal launcher (set `ROOT` to where the repo is). Free Colab sessions can stop early, so save checkpoints to Google Drive and resume with `--resume /path/to/last.pt`.

## What to keep from every run

Whatever path you're on, keep each run in its own folder with:
- the exact config (and the command you ran)
- the training history (`history.jsonl` for `waste_lab train`, `results.csv` for YOLO)
- the checkpoint or model file
- the evaluation numbers, on validation while tuning and on test once at the end
- the dataset, split, seed, device and how long it took

Then add a short entry to [`LOG.md`](LOG.md).

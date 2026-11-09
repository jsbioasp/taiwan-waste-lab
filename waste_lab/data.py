"""Common NPZ interface. Splits are explicit; group leakage is an error."""
import json
from pathlib import Path
import numpy as np

SPLITS = {"train", "val", "test"}

def read_manifest(path, check_files=True):
    path = Path(path).resolve()
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    groups, files = {}, set()
    for row in rows:
        for key in ("path", "group", "split", "dataset"):
            if not row.get(key):
                raise ValueError(f"Missing {key}: {row}")
        if row["split"] not in SPLITS:
            raise ValueError("split must be train, val, or test")
        # Group IDs must be globally namespaced and identical for shared scenes.
        group = row["group"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Group leakage: {group}")
        groups[group] = row["split"]
        full = (path.parent / row["path"]).resolve()
        if full in files:
            raise ValueError(f"Duplicate sample: {full}")
        files.add(full)
        if check_files and not full.is_file():
            raise FileNotFoundError(full)
        row["path"] = str(full)
    return rows

def sample(row):
    with np.load(row["path"], allow_pickle=False) as z:
        x, y = z["x"].astype("float32"), z["y"]
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite input: {row['path']}")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("Labels must be integers, -1 means ignore")
        return x, y.astype("int64")

def validate(rows, cfg):
    task = cfg["task"]
    if task not in ("spectra", "image", "segmentation"):
        raise ValueError(f"Unknown task: {task}")
    shapes = set()
    for row in rows:
        x, y = sample(row)
        ds = row["dataset"]
        if ds not in cfg["heads"]:
            raise ValueError(f"No class count for head {ds}")
        if task == "spectra":
            ok = x.ndim == 2 and x.shape[0] == 1 and y.ndim == 0
        elif task == "image":
            ok = x.ndim == 3 and y.ndim == 0
        else:
            ok = x.ndim == 3 and y.shape == x.shape[1:]
        if not ok or x.shape[0] != cfg["channels"]:
            raise ValueError(f"Invalid shape: {row['path']}: {x.shape}, {y.shape}")
        if (y < -1).any() or (y >= cfg["heads"][ds]).any():
            raise ValueError(f"Out-of-range label: {row['path']}")
        if not (y >= 0).any():
            raise ValueError(f"No labelled targets: {row['path']}")
        shapes.add(x.shape)
    if len(shapes) != 1:
        raise ValueError("Use a common shape and physical band/axis order across all inputs")
    for ds in cfg["heads"]:
        for split in ("train", "val"):
            if not any(r["dataset"] == ds and r["split"] == split for r in rows):
                raise ValueError(f"Missing {ds}/{split}")
    if cfg.get("aux_weight", 0):
        if task != "segmentation":
            raise ValueError("NDVI auxiliary task requires segmentation inputs")
        indices = [cfg.get("red_index", -1), cfg.get("nir_index", -1)]
        if len(set(indices)) != 2 or not all(0 <= i < cfg["channels"] for i in indices):
            raise ValueError("Set distinct zero-based red_index and nir_index")

def toy(dest):
    """Only synthetic software fixtures; never scientific benchmark evidence."""
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for task in ("spectra", "image", "segmentation"):
        folder = dest / task; folder.mkdir(exist_ok=True)
        rows = []
        for ds in ("source_a", "source_b"):
            for split, count in (("train", 12), ("val", 4), ("test", 4)):
                for i in range(count):
                    cls = i % 2
                    if task == "spectra":
                        axis = np.linspace(0, 1, 64)
                        x = (np.exp(-((axis - (.3 if cls else .7)) / .08)**2)
                             + rng.normal(0, .03, 64))[None].astype("float32")
                        y = np.array(cls, dtype="int64")
                    else:
                        x = rng.uniform(.01, .2, (3, 16, 16)).astype("float32")
                        mask = np.zeros((16, 16), dtype="int64")
                        if cls:
                            x[0, 4:12, 4:12] += .6; mask[4:12, 4:12] = 1
                        y = mask if task == "segmentation" else np.array(cls, dtype="int64")
                        if task == "segmentation": y[:2] = -1
                    name = f"{ds}_{split}_{i}.npz"
                    np.savez_compressed(folder / name, x=x, y=y)
                    rows.append(dict(path=name, group=f"toy:{ds}:{split}:{i}",
                                     split=split, dataset=ds))
        (folder / "manifest.jsonl").write_text("".join(json.dumps(v)+"\n" for v in rows))
    print(f"Synthetic fixtures created: {dest}")

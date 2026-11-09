import csv
import sys
from pathlib import Path
rows=list(csv.DictReader(Path(sys.argv[1]).open()))
assert rows, "Empty split table"
groups={}; images=set()
for r in rows:
    assert r["group"] and r["split"] in {"train","val","test"}
    assert r["image"] not in images, "Repeated image"
    assert r["group"] not in groups or groups[r["group"]]==r["split"], "Flight/site leakage"
    groups[r["group"]]=r["split"]; images.add(r["image"])
print(f"Checked {len(rows)} images, {len(groups)} groups. Also check near-duplicate images manually.")

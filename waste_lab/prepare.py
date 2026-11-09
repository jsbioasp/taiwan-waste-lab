"""Adapters require explicit axes, groups, splits, class maps and band orders."""
import csv
import json
from pathlib import Path
import numpy as np
from .data import read_manifest

def spectra_csv(source,out):
    # Wide CSV: sample_id,group,split,dataset,label,<numeric spectral axis columns>
    source=Path(source); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    with source.open() as f:
        reader=csv.DictReader(f); fields=reader.fieldnames
        required={"sample_id","group","split","dataset","label"}
        if not required.issubset(fields or []): raise ValueError(f"Required columns: {required}")
        bands=[k for k in fields if k not in required]
        axis=[float(k) for k in bands]
        if not axis or not all(a<b for a,b in zip(axis,axis[1:])): raise ValueError("Axis columns must be numeric and increasing")
        rows=[]; seen=set()
        for i,row in enumerate(reader):
            if row["sample_id"] in seen: raise ValueError("Duplicate sample_id")
            seen.add(row["sample_id"])
            name=f"spectrum_{i:06}.npz"
            x=np.array([float(row[k]) for k in bands],dtype="float32")[None]
            np.savez_compressed(out/name,x=x,y=np.array(int(row["label"]),dtype="int64"),axis=np.array(axis))
            rows.append(dict(path=name,group=row["group"],split=row["split"],dataset=row["dataset"],sample_id=row["sample_id"]))
    manifest=out/"manifest.jsonl"; manifest.write_text("".join(json.dumps(v)+"\n" for v in rows))
    read_manifest(manifest)
    return dict(samples=len(rows),manifest=str(manifest))

def geotiff(spec,out):
    import rasterio
    spec=Path(spec); cfg=json.loads(spec.read_text()); root=spec.parent
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rows=[]
    mapping={int(k):int(v) for k,v in cfg["label_map"].items()}
    if not cfg.get("bands") or cfg.get("scale",0)<=0: raise ValueError("Explicit one-based bands and positive scale are required")
    for i,r in enumerate(cfg["samples"]):
        with rasterio.open(root/r["image"]) as src, rasterio.open(root/r["mask"]) as labels:
            if (src.shape!=labels.shape or src.crs!=labels.crs or src.transform!=labels.transform):
                raise ValueError("Image and label grids differ; align them before packing")
            raw=src.read(cfg["bands"]).astype("float32")
            valid=(src.read_masks(cfg["bands"])>0).all(0) & (labels.read_masks(1)>0) & np.isfinite(raw).all(0)
            x=raw*cfg["scale"]+cfg.get("offset",0)
            old=labels.read(1); y=np.full(old.shape,-1,dtype="int64")
            unknown=set(np.unique(old[valid]))-set(mapping)
            if unknown: raise ValueError(f"Unmapped labels: {unknown}")
            for key,value in mapping.items(): y[old==key]=value
            y[~valid]=-1; x[:,~valid]=0
            name=f"patch_{i:06}.npz"; np.savez_compressed(out/name,x=x,y=y)
            rows.append(dict(path=name,group=r["group"],dataset=r["dataset"],split=r["split"]))
    manifest=out/"manifest.jsonl"; manifest.write_text("".join(json.dumps(v)+"\n" for v in rows)); read_manifest(manifest)
    (out/"preparation.json").write_text(json.dumps(cfg,indent=2)+"\n")
    return dict(samples=len(rows),manifest=str(manifest))

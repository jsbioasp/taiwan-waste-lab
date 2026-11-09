"""Pack MARIDA directly from its ZIP, keeping official scene-disjoint splits."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from rasterio.io import MemoryFile
from waste_lab.data import read_manifest

def prepare(archive,out,limit=2):
    if limit<0:raise ValueError("limit must be nonnegative; zero means all")
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];provenance=[]
    with zipfile.ZipFile(archive) as z:
        for split in ("train","val","test"):
            ids=z.read(f"splits/{split}_X.txt").decode().splitlines()
            for ident in ids[:limit] if limit else ids:
                scene="S2_"+ident.rsplit("_",1)[0];base=f"patches/{scene}/S2_{ident}"
                image_bytes=z.read(base+".tif");mask_bytes=z.read(base+"_cl.tif")
                with MemoryFile(image_bytes) as image_mem,MemoryFile(mask_bytes) as mask_mem:
                    with image_mem.open() as src,mask_mem.open() as labels:
                        if src.shape!=labels.shape or src.crs!=labels.crs or src.transform!=labels.transform:
                            raise ValueError("Unaligned image/mask")
                        x=src.read().astype("float32");raw=labels.read(1)
                        if not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all() or raw.min()<0 or raw.max()>15:
                            raise ValueError("Unexpected MARIDA source label map")
                        y=raw.astype("int64")-1
                        valid=(src.read_masks()>0).all(0)&np.isfinite(x).all(0)&(labels.read_masks(1)>0)
                        y[~valid]=-1;x[:,~valid]=0
                        if src.count!=11:raise ValueError("Expected original 11-band MARIDA release")
                filename=f"S2_{ident}.npz";np.savez_compressed(out/filename,x=x,y=y)
                rows.append(dict(path=filename,group="sentinel2:"+scene,split=split,dataset="marida"))
                provenance.append(dict(file=filename,source_image=base+".tif",source_mask=base+"_cl.tif",
                    image_sha256=hashlib.sha256(image_bytes).hexdigest(),mask_sha256=hashlib.sha256(mask_bytes).hexdigest()))
    manifest=out/"manifest.jsonl";manifest.write_text("".join(json.dumps(v)+"\n" for v in rows));read_manifest(manifest)
    (out/"provenance.json").write_text(json.dumps(dict(source="https://zenodo.org/records/5151941",
        attribution="Kikaki et al. (2022), MARIDA, DOI 10.1371/journal.pone.0262247",license="CC-BY-4.0",
        changes="Original float32 bands unchanged; source classes 0..15 mapped to -1..14; invalid pixels ignored; NPZ packaging",
        subset_warning="Limited first patches per official split are for software checks, not representative benchmark results" if limit else "Full official splits",
        samples=provenance),indent=2)+"\n")
    print(json.dumps(dict(samples=len(rows),manifest=str(manifest))))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("archive");p.add_argument("--out",required=True);p.add_argument("--limit-per-split",type=int,default=2)
    a=p.parse_args();prepare(a.archive,a.out,a.limit_per_split)

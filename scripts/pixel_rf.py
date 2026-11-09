"""Spectral-only RF baseline, not the paper's spectral+texture reproduction."""
import argparse
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from waste_lab.data import read_manifest,sample
from waste_lab.train import scores

def run(manifest,head,out,classes=15,max_pixels=1024):
    if classes<2 or max_pixels<1:raise ValueError("Invalid class or pixel count")
    rows=[r for r in read_manifest(manifest) if r["dataset"]==head]
    if not all(any(r["split"]==s for r in rows) for s in ["train","val","test"]):raise ValueError("Need all splits")
    rng=np.random.default_rng(42);xs=[];ys=[]
    for row in rows:
        x,y=sample(row)
        if x.ndim!=3 or y.shape!=x.shape[1:] or (y<-1).any() or (y>=classes).any():raise ValueError("Invalid segmentation sample")
        if row["split"]!="train":continue
        valid=np.flatnonzero(y.ravel()>=0)
        take=rng.choice(valid,min(max_pixels,len(valid)),replace=False)
        xs.append(x.reshape(x.shape[0],-1).T[take]);ys.append(y.ravel()[take])
    if not sum(len(y) for y in ys):raise ValueError("No labelled training pixels")
    model=RandomForestClassifier(n_estimators=100,class_weight="balanced",random_state=42,n_jobs=2)
    model.fit(np.concatenate(xs),np.concatenate(ys));metrics={}
    for split in ["val","test"]:
        cm=np.zeros((classes,classes),dtype="int64")
        for row in rows:
            if row["split"]!=split:continue
            x,y=sample(row);valid=y.ravel()>=0;target=y.ravel()[valid]
            if not len(target):continue
            pred=model.predict(x.reshape(x.shape[0],-1).T[valid]);cm+=np.bincount(target*classes+pred,minlength=classes*classes).reshape(classes,classes)
        metrics[split]=scores(cm)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);joblib.dump(model,out/"model.joblib")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("manifest");p.add_argument("--head",required=True);p.add_argument("--out",required=True);p.add_argument("--classes",type=int,default=15);p.add_argument("--max-pixels",type=int,default=1024)
    a=p.parse_args();run(a.manifest,a.head,a.out,a.classes,a.max_pixels)

"""CPU SVM / random forest for spectra or image summary features."""
import json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
from .data import read_manifest,sample
from .train import scores

def run(manifest,head,kind,out):
    rows=[r for r in read_manifest(manifest) if r["dataset"]==head]
    sets={}
    for split in ("train","val","test"):
        pairs=[sample(r) for r in rows if r["split"]==split]
        if not pairs: raise ValueError(f"Missing {head}/{split}")
        if any(y.ndim!=0 or y.item()<0 for _,y in pairs):
            raise ValueError("Baseline expects scalar class labels; segmentation requires pixel sampling")
        def features(x):
            if x.ndim==2: return x.ravel()
            return np.concatenate((x.mean((1,2)),x.std((1,2)),x.min((1,2)),x.max((1,2))))
        sets[split]=(np.stack([features(x) for x,_ in pairs]),np.array([y.item() for _,y in pairs]))
    model=make_pipeline(StandardScaler(),SVC(C=1)) if kind=="svm" else RandomForestClassifier(
        n_estimators=100,class_weight="balanced",random_state=42,n_jobs=2)
    model.fit(*sets["train"]); result={}
    labels=np.unique(np.concatenate([v[1] for v in sets.values()]))
    for split in ("val","test"):
        x,y=sets[split]; result[split]=scores(confusion_matrix(y,model.predict(x),labels=labels))
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,out/"model.joblib")
    (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

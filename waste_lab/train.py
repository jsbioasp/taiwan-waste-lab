"""Single- and multi-dataset training, held-out evaluation, and timed inference."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import read_manifest, sample, validate
from .models import SmallModel

def select_device(name):
    if name == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def normalizer(rows, channels):
    total = np.zeros(channels); squares = total.copy(); count = 0
    for row in rows:
        x, _ = sample(row); flat = x.reshape(channels,-1).astype("float64")
        total += flat.sum(1); squares += (flat**2).sum(1); count += flat.shape[1]
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(squares/count - mean**2,0)),1e-6)
    return mean.astype("float32"), std.astype("float32")

def batch(rows, mean, std, device):
    xy = [sample(r) for r in rows]
    raw = np.stack([v[0] for v in xy])
    shape = (1,len(mean)) + (1,) * (raw.ndim-2)
    x = (raw - mean.reshape(shape)) / std.reshape(shape)
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(np.stack([v[1] for v in xy])).to(device), raw)

def batches(rows, size, shuffle=False):
    by_head = {}
    for r in rows: by_head.setdefault(r["dataset"],[]).append(r)
    chunks = []
    for values in by_head.values():
        values = values.copy()
        if shuffle: random.shuffle(values)
        chunks.extend(values[i:i+size] for i in range(0,len(values),size))
    if shuffle: random.shuffle(chunks)
    return chunks

def scores(cm):
    tp = cm.diagonal(); union = cm.sum(0)+cm.sum(1)-tp
    iou = np.divide(tp,union,out=np.full(len(tp),np.nan),where=union>0)
    den = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*tp,den,out=np.full(len(tp),np.nan),where=den>0)
    return dict(confusion=cm.tolist(), iou=[None if np.isnan(v) else float(v) for v in iou],
                f1=[None if np.isnan(v) else float(v) for v in f1],
                macro_f1=float(np.nanmean(f1)), mean_iou=float(np.nanmean(iou)))

@torch.no_grad()
def evaluate(model, rows, cfg, mean, std, device):
    model.eval(); matrices = {h:np.zeros((n,n),dtype="int64") for h,n in cfg["heads"].items()}
    seen = set()
    for rs in batches(rows,cfg.get("batch_size",8)):
        head = rs[0]["dataset"]; x,y,_ = batch(rs,mean,std,device)
        logits,_ = model(x,head); pred = logits.argmax(1)
        valid = y>=0; n=cfg["heads"][head]
        codes = (y[valid]*n+pred[valid]).cpu().numpy()
        matrices[head] += np.bincount(codes,minlength=n*n).reshape(n,n); seen.add(head)
    return {h:scores(matrices[h]) for h in sorted(seen)}

def save_checkpoint(path, payload):
    path=Path(path); tmp=path.with_suffix(".tmp")
    torch.save(payload,tmp); tmp.replace(path)

def train(manifest, config, out, device="auto", resume=None):
    cfg=json.loads(Path(config).read_text()); rows=read_manifest(manifest); validate(rows,cfg)
    train_rows=[r for r in rows if r["split"]=="train"]
    val_rows=[r for r in rows if r["split"]=="val"]
    seed=cfg.get("seed",42); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev=select_device(device); torch.set_num_threads(cfg.get("cpu_threads",2))
    model=SmallModel(cfg).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=cfg.get("lr",.001))
    amp=dev.type=="cuda" and cfg.get("amp",True)
    scaler=torch.amp.GradScaler("cuda",enabled=amp)
    mean,std=normalizer(train_rows,cfg["channels"])
    digest=hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    start,best=0,float("-inf")
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/"last.pt").exists() and not resume:
        raise ValueError("Output already has a run; choose a new directory or use --resume")
    if resume:
        ck=torch.load(resume,map_location="cpu",weights_only=True)
        for key in cfg:
            if key != "epochs" and cfg[key] != ck["config"].get(key):
                raise ValueError(f"Resume configuration changed: {key}")
        if digest != ck["manifest_sha256"]: raise ValueError("Resume manifest changed")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"]); start=ck["epoch"]; best=ck["best"]
        mean=np.array(ck["mean"],dtype="float32"); std=np.array(ck["std"],dtype="float32")
        random.setstate(ck["python_rng"]); torch.set_rng_state(ck["torch_rng"])
        if dev.type=="cuda" and ck.get("cuda_rng"): torch.cuda.set_rng_state_all(ck["cuda_rng"])
    history=[]
    for epoch in range(start,cfg.get("epochs",20)):
        model.train(); started=time.perf_counter(); losses=[]
        for rs in batches(train_rows,cfg.get("batch_size",8),shuffle=True):
            head=rs[0]["dataset"]; x,y,raw=batch(rs,mean,std,dev); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=amp):
                logits,aux=model(x,head)
                loss=F.cross_entropy(logits,y,ignore_index=-1)
                if aux is not None:
                    red=raw[:,cfg["red_index"]]; nir=raw[:,cfg["nir_index"]]
                    den=nir+red; valid=np.abs(den)>1e-6
                    ndvi=np.divide(nir-red,den,out=np.zeros_like(den),where=valid).clip(-1,1)
                    valid_t=torch.from_numpy(valid).to(dev)
                    if valid.any():
                        target=torch.from_numpy(ndvi).to(dev)
                        loss=loss+cfg["aux_weight"]*F.mse_loss(aux[:,0][valid_t],target[valid_t])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(loss.item())
        metrics=evaluate(model,val_rows,cfg,mean,std,dev)
        score=float(np.mean([v["macro_f1"] for v in metrics.values()]))
        improved=score>best; best=max(best,score)
        record=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=metrics,seconds=time.perf_counter()-started)
        history.append(record)
        with (out/"history.jsonl").open("a") as f: f.write(json.dumps(record)+"\n")
        payload=dict(config=cfg,model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                     mean=mean.tolist(),std=std.tolist(),epoch=epoch+1,best=best,manifest_sha256=digest,
                     python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if dev.type=="cuda" else [],
                     environment=dict(torch=str(torch.__version__),numpy=np.__version__,device=str(dev)))
        save_checkpoint(out/"last.pt",payload)
        if improved: save_checkpoint(out/"best.pt",payload)
        print(json.dumps(record))
    (out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    return history

def load(checkpoint,device):
    dev=select_device(device); ck=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=SmallModel(ck["config"]).to(dev); model.load_state_dict(ck["model"]); model.eval()
    return model,ck,dev

def test(checkpoint,manifest,split,device="cpu"):
    model,ck,dev=load(checkpoint,device); rows=read_manifest(manifest)
    validate(rows,ck["config"])
    rows=[r for r in rows if r["split"]==split]
    if not rows: raise ValueError(f"No {split} records")
    return evaluate(model,rows,ck["config"],np.array(ck["mean"],dtype="float32"),
                    np.array(ck["std"],dtype="float32"),dev)

@torch.no_grad()
def predict(checkpoint,input_path,head,out,device="cpu",repeats=20):
    model,ck,dev=load(checkpoint,device)
    torch.set_num_threads(ck["config"].get("cpu_threads",2))
    if repeats<1: raise ValueError("repeats must be positive")
    with np.load(input_path,allow_pickle=False) as z: raw=z["x"].astype("float32")
    if not np.isfinite(raw).all(): raise ValueError("Nonfinite input")
    mean=np.array(ck["mean"],dtype="float32"); std=np.array(ck["std"],dtype="float32")
    shape=(len(mean),)+(1,)*(raw.ndim-1)
    x=torch.from_numpy((raw-mean.reshape(shape))/std.reshape(shape))[None].to(dev)
    def sync():
        if dev.type=="cuda": torch.cuda.synchronize()
    for _ in range(3): model(x,head)
    sync(); times=[]
    for _ in range(repeats):
        t=time.perf_counter(); logits,_=model(x,head); sync(); times.append(time.perf_counter()-t)
    probs=logits.softmax(1)[0].cpu().numpy()
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,probabilities=probs,prediction=probs.argmax(0))
    return dict(device=str(dev),shape=list(raw.shape),batch_size=1,warmup=3,repeats=repeats,
                median_ms=float(np.median(times)*1000),p95_ms=float(np.percentile(times,95)*1000),
                timing_scope="warm model forward only; excludes loading, preprocessing and output I/O")

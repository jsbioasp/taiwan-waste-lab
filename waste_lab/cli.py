import argparse
import json

def main():
    p=argparse.ArgumentParser(description="Taiwan waste research starter; see README for schemas")
    s=p.add_subparsers(dest="cmd",required=True)
    s.add_parser("catalog")
    q=s.add_parser("toy"); q.add_argument("--out",default="data/processed/toy")
    q=s.add_parser("validate"); q.add_argument("manifest"); q.add_argument("config")
    q=s.add_parser("train"); q.add_argument("manifest"); q.add_argument("config"); q.add_argument("--out",required=True); q.add_argument("--device",default="auto"); q.add_argument("--resume")
    q=s.add_parser("evaluate"); q.add_argument("checkpoint"); q.add_argument("manifest"); q.add_argument("--split",choices=["val","test"],default="test"); q.add_argument("--device",default="cpu")
    q=s.add_parser("predict"); q.add_argument("checkpoint"); q.add_argument("input"); q.add_argument("--head",required=True); q.add_argument("--out",required=True); q.add_argument("--device",default="cpu"); q.add_argument("--repeats",type=int,default=20)
    q=s.add_parser("baseline"); q.add_argument("manifest"); q.add_argument("--head",required=True); q.add_argument("--kind",choices=["svm","rf"],default="svm"); q.add_argument("--out",required=True)
    q=s.add_parser("zenodo"); q.add_argument("dataset"); q.add_argument("--file"); q.add_argument("--out",default="data/raw"); q.add_argument("--max-gb",type=float,default=2); q.add_argument("--accept-terms",action="store_true")
    q=s.add_parser("moenv-index"); q.add_argument("--out",default="data/raw/moenv/index.json")
    q=s.add_parser("moenv-fetch"); q.add_argument("index"); q.add_argument("filename"); q.add_argument("--out",default="data/raw/moenv"); q.add_argument("--max-mb",type=float,default=20)
    q=s.add_parser("prepare-spectra"); q.add_argument("source"); q.add_argument("--out",required=True)
    q=s.add_parser("prepare-geotiff"); q.add_argument("spec"); q.add_argument("--out",required=True)
    q=s.add_parser("search-sentinel"); q.add_argument("--bbox",type=float,nargs=4,required=True); q.add_argument("--dates",required=True); q.add_argument("--out",required=True); q.add_argument("--cloud",type=float,default=20); q.add_argument("--limit",type=int,default=20)
    a=p.parse_args(); result=None
    if a.cmd=="catalog":
        from .download import catalog
        result=catalog()
    elif a.cmd=="toy":
        from .data import toy
        toy(a.out)
    elif a.cmd=="validate":
        from pathlib import Path
        from .data import read_manifest,validate
        rows=read_manifest(a.manifest); validate(rows,json.loads(Path(a.config).read_text())); result={"samples":len(rows),"valid":True}
    elif a.cmd=="train":
        from .train import train
        train(a.manifest,a.config,a.out,a.device,a.resume)
    elif a.cmd=="evaluate":
        from .train import test
        result=test(a.checkpoint,a.manifest,a.split,a.device)
    elif a.cmd=="predict":
        from .train import predict
        result=predict(a.checkpoint,a.input,a.head,a.out,a.device,a.repeats)
    elif a.cmd=="baseline":
        from .baseline import run
        result=run(a.manifest,a.head,a.kind,a.out)
    elif a.cmd=="zenodo":
        from .download import zenodo
        result=zenodo(a.dataset,a.file,a.out,a.max_gb,a.accept_terms)
    elif a.cmd=="moenv-index":
        from .download import moenv_index
        result=moenv_index(a.out)
    elif a.cmd=="moenv-fetch":
        from .download import moenv_fetch
        result=moenv_fetch(a.index,a.filename,a.out,a.max_mb)
    elif a.cmd=="prepare-spectra":
        from .prepare import spectra_csv
        result=spectra_csv(a.source,a.out)
    elif a.cmd=="prepare-geotiff":
        from .prepare import geotiff
        result=geotiff(a.spec,a.out)
    elif a.cmd=="search-sentinel":
        from .taiwan import search
        result=search(a.bbox,a.dates,a.out,a.cloud,a.limit)
    if result is not None: print(json.dumps(result,ensure_ascii=False,indent=2))

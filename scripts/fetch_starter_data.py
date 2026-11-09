"""Fetch the selected starter data, bounded to ~1.2 GB; no credentials needed."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from waste_lab.download import download, get_json
ROOT=Path(__file__).resolve().parents[1]

def fetch(name,url,path,max_bytes,checksum=None,source=None,license=None):
    path=ROOT/path
    if path.exists():
        print(f"Already exists (not reverified): {path}",flush=True);return
    result=download(url,path,max_bytes,checksum)
    result.update(dataset=name,source=source or url,license=license)
    path.with_name(path.name+".provenance.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)

def main():
    jobs=[]
    rec=json.loads((ROOT/"data/snapshots/marida-record.json").read_text())
    f=next(x for x in rec["files"] if x["key"]=="MARIDA.zip")
    jobs.append(("marida",f["links"]["self"],"data/raw/marida/MARIDA.zip",1_300_000_000,f["checksum"],"https://zenodo.org/records/5151941","CC-BY-4.0"))
    rec=json.loads((ROOT/"data/snapshots/knaeps-record.json").read_text())
    for f in rec["files"]:
        jobs.append(("knaeps",f["download_url"],"data/raw/knaeps/"+f["name"],10_000_000,"md5:"+f["computed_md5"],"https://data.4tu.nl/articles/_/12896312/2",rec.get("license")))
    jobs.append(("moenv-dumps","https://newcdx.moenv.gov.tw/api/files/WR_P_244/2346623e-8279-4788-b8a8-b6b2f810a0d1","data/raw/moenv/WDMS-WGS84-20260909.zip",20_000_000,None,"https://data.gov.tw/en/datasets/145968","OGDL-1.0"))
    failed=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,*j):j[0] for j in jobs}
        for future in as_completed(futures):
            try:future.result()
            except Exception as e:
                failed.append(dict(dataset=futures[future],error=str(e)));print(failed[-1],flush=True)
    (ROOT/"data/snapshots/starter-download-status.json").write_text(json.dumps(dict(failed=failed),indent=2)+"\n")
    if failed:raise SystemExit(1)
if __name__=="__main__":main()

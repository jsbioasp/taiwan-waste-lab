"""Explicit, size-capped downloads; no archive extraction or silent bulk fetches."""
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

CATALOG = Path(__file__).with_name("catalog.json")

def catalog(): return json.loads(CATALOG.read_text())

def get_json(url):
    req=urllib.request.Request(url,headers={"User-Agent":"taiwan-waste-lab/0.1"})
    with urllib.request.urlopen(req,timeout=60) as response:
        return json.load(response)

def download(url,dest,max_bytes,checksum=None):
    dest=Path(dest)
    if dest.exists(): raise FileExistsError(f"Refusing to overwrite {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True); tmp=dest.with_suffix(dest.suffix+".part")
    size=0; sha=hashlib.sha256(); md5=hashlib.md5()
    try:
        req=urllib.request.Request(url,headers={"User-Agent":"taiwan-waste-lab/0.1"})
        with urllib.request.urlopen(req,timeout=60) as response, tmp.open("wb") as f:
            length=int(response.headers.get("Content-Length",0))
            if length>max_bytes: raise ValueError("Download exceeds size cap")
            while True:
                chunk=response.read(1024*1024)
                if not chunk: break
                size+=len(chunk)
                if size>max_bytes: raise ValueError("Download exceeds size cap")
                f.write(chunk); sha.update(chunk); md5.update(chunk)
        if checksum:
            algo,expected=checksum.split(":",1)
            actual={"md5":md5.hexdigest(),"sha256":sha.hexdigest()}.get(algo)
            if actual is None or actual!=expected: raise ValueError("Checksum mismatch")
        tmp.replace(dest)
    finally:
        if tmp.exists(): tmp.unlink()
    return dict(file=str(dest),bytes=size,sha256=sha.hexdigest())

def zenodo(dataset,file=None,out="data/raw",max_gb=2,accept_terms=False):
    entry=next(v for v in catalog() if v["id"]==dataset)
    if "zenodo_record" not in entry: raise ValueError("Use the catalog's manual source instructions")
    data=get_json("https://zenodo.org/api/records/"+entry["zenodo_record"])
    files=data.get("files",[])
    if file is None:
        return dict(dataset=dataset,license=data["metadata"].get("license"),
                    files=[dict(name=f["key"],bytes=f["size"],checksum=f.get("checksum")) for f in files])
    if not accept_terms: raise ValueError("Read the record's terms, then pass --accept-terms")
    if Path(file).name!=file or file in (".",".."): raise ValueError("Unsafe filename")
    f=next(f for f in files if f["key"]==file)
    result=download(f["links"]["self"],Path(out)/dataset/file,int(max_gb*1e9),f.get("checksum"))
    result.update(dataset=dataset,source=entry["url"],license=data["metadata"].get("license"))
    Path(result["file"]+".provenance.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

def moenv_index(out):
    key=os.environ.get("MOENV_API_KEY")
    if not key: raise ValueError("Set MOENV_API_KEY from your MOENV developer account")
    from urllib.parse import urlencode
    url="https://data.moenv.gov.tw/api/v2/wr_p_244?"+urlencode(dict(api_key=key,format="json",limit=1000))
    data=get_json(url)
    rows=data if isinstance(data,list) else data.get("records",[])
    clean=[{k:v for k,v in row.items() if k in ("filename","url","description","upload_date")} for row in rows]
    path=Path(out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean,ensure_ascii=False,indent=2)+"\n")
    return dict(records=len(clean),path=str(path))

def moenv_fetch(index,filename,out,max_mb=20):
    rows=json.loads(Path(index).read_text())
    row=next(r for r in rows if r["filename"]==filename)
    if Path(filename).name!=filename: raise ValueError("Unsafe filename")
    if not re.match(r"^https://newcdx\.moenv\.gov\.tw/",row["url"]): raise ValueError("Unexpected source host")
    result=download(row["url"],Path(out)/filename,int(max_mb*1e6))
    result.update(source="https://data.gov.tw/en/datasets/145968",license="OGDL-1.0",upload_date=row.get("upload_date"))
    Path(result["file"]+".provenance.json").write_text(json.dumps(result,indent=2)+"\n")
    return result

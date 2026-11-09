"""Free Sentinel-2 STAC discovery and manual site-review scaffolding."""
import json
import urllib.request
from pathlib import Path

def search(bbox,dates,out,cloud=20,limit=20):
    if len(bbox)!=4 or bbox[0]>=bbox[2] or bbox[1]>=bbox[3]: raise ValueError("Invalid west,south,east,north bbox")
    if not 1<=limit<=100: raise ValueError("limit must be 1..100")
    body=dict(collections=["sentinel-2-l2a"],bbox=bbox,datetime=dates,limit=limit,
              query={"eo:cloud_cover":{"lt":cloud}})
    req=urllib.request.Request("https://planetarycomputer.microsoft.com/api/stac/v1/search",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json","User-Agent":"taiwan-waste-lab/0.1"})
    with urllib.request.urlopen(req,timeout=60) as response: result=json.load(response)
    path=Path(out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2)+"\n")
    return dict(scenes=len(result.get("features",[])),more_pages=any(l.get("rel")=="next" for l in result.get("links",[])),
                note="First page only. Scene cloud fraction does not ensure a clear site; inspect SCL/cloud masks.")

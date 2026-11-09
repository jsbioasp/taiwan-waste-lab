"""Convert one verified WGS84 MOENV point archive into GeoJSON and review CSV."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import shapefile

def prepare(archive,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        shapes=[n for n in z.namelist() if n.endswith(".shp")]
        if len(shapes)!=1:raise ValueError("Expected one shapefile")
        base=shapes[0][:-4];prj=z.read(base+".prj").decode()
        if "GCS_WGS_1984" not in prj or "PROJCS" in prj:raise ValueError("Expected geographic WGS84; reproject explicitly")
        encoding=z.read(base+".cpg").decode().strip()
        reader=shapefile.Reader(shp=io.BytesIO(z.read(base+".shp")),shx=io.BytesIO(z.read(base+".shx")),
            dbf=io.BytesIO(z.read(base+".dbf")),encoding=encoding,encodingErrors="replace")
        features=[];review=[]
        for sr in reader.iterShapeRecords():
            if sr.shape.shapeType!=1:raise ValueError("Expected point features")
            props=sr.record.as_dict()
            props={k:v.isoformat() if hasattr(v,"isoformat") else v for k,v in props.items()}
            props["__decoding_replacement_fields"]=[k for k,v in props.items() if isinstance(v,str) and "\ufffd" in v]
            lon,lat=sr.shape.points[0]
            if not (-180<=lon<=180 and -90<=lat<=90):raise ValueError("Invalid coordinates")
            features.append(dict(type="Feature",geometry=dict(type="Point",coordinates=[lon,lat]),properties=props))
            ident=str(props.get("案件編號",props.get("CSID","")))
            review.append(dict(site_id=ident,longitude=lon,latitude=lat,record_updated=props.get("最後更新日",""),
                image_date="",scene_id="",waste_visible="",confidence="",reviewer="",notes="",group="moenv:"+ident,split=""))
    (out/"sites.geojson").write_text(json.dumps(dict(type="FeatureCollection",features=features),ensure_ascii=False,indent=2)+"\n")
    with (out/"site-review.csv").open("w",newline="") as f:
        fields=["site_id","longitude","latitude","record_updated","image_date","scene_id","waste_visible","confidence","reviewer","notes","group","split"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(review)
    (out/"provenance.json").write_text(json.dumps(dict(source="https://data.gov.tw/en/datasets/145968",provider="Taiwan Ministry of Environment",
        archive=Path(archive).name,sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),license="Open Government Data License v1.0",
        changes="Converted WGS84 shapefile and Big5 attributes to UTF-8 GeoJSON; added blank manual image-review fields",
        records=len(features),encoding=encoding,
        records_with_decoding_replacements=sum(bool(f["properties"]["__decoding_replacement_fields"]) for f in features),
        text_warning="Some fixed-width source DBF fields truncate a Big5 character; replacement characters are explicitly flagged per feature; original ZIP preserves all source bytes",warning="Listed sites are not verified visible-waste image labels"),indent=2)+"\n")
    print(json.dumps(dict(records=len(features),out=str(out))))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("archive");p.add_argument("--out",required=True)
    a=p.parse_args();prepare(a.archive,a.out)

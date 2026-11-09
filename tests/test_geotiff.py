import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
try:
    import rasterio
    from rasterio.transform import from_origin
except ImportError:
    rasterio=None
from waste_lab.prepare import geotiff
from waste_lab.data import sample,read_manifest

@unittest.skipIf(rasterio is None,"install geo extra")
class GeoTests(unittest.TestCase):
    def test_ignores_unlabelled_and_rejects_misalignment(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);transform=from_origin(120,25,.001,.001)
            for name,arr in [("image.tif",np.ones((3,4,4),dtype="float32")),("mask.tif",np.array([[[0,1,2,1]]*4],dtype="uint8"))]:
                with rasterio.open(root/name,"w",driver="GTiff",height=4,width=4,count=arr.shape[0],dtype=arr.dtype,crs="EPSG:4326",transform=transform) as f:f.write(arr)
            cfg=dict(bands=[1,2,3],scale=1,label_map={"0":-1,"1":0,"2":1},samples=[dict(image="image.tif",mask="mask.tif",group="s1",split="train",dataset="a")])
            spec=root/"spec.json";spec.write_text(json.dumps(cfg));geotiff(spec,root/"out")
            x,y=sample(read_manifest(root/"out/manifest.jsonl")[0]);self.assertEqual(int((y==-1).sum()),4)
            with rasterio.open(root/"mask.tif","r+") as f:f.transform=from_origin(121,25,.001,.001)
            with self.assertRaisesRegex(ValueError,"grids differ"):geotiff(spec,root/"bad")

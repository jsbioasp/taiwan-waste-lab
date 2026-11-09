import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from waste_lab.data import toy, read_manifest, validate, sample
from waste_lab.train import train, test as evaluate, predict, scores
from waste_lab.prepare import spectra_csv
from waste_lab.download import download
from waste_lab.baseline import run as baseline

ROOT=Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def fixture(self):
        with contextlib.redirect_stdout(io.StringIO()):toy(self.root)
    def test_leakage_and_duplicates_fail(self):
        p=self.root/"manifest.jsonl"
        rows=[dict(path="a.npz",group="same",split="train",dataset="a"),
              dict(path="b.npz",group="same",split="test",dataset="b")]
        p.write_text("\n".join(json.dumps(v) for v in rows))
        with self.assertRaisesRegex(ValueError,"leakage"):read_manifest(p,False)
        rows[1].update(group="other",split="train",path="a.npz")
        p.write_text("\n".join(json.dumps(v) for v in rows))
        with self.assertRaisesRegex(ValueError,"Duplicate"):read_manifest(p,False)
    def test_all_tasks_train_resume_predict(self):
        self.fixture()
        for task in ("spectra","image","segmentation"):
            with self.subTest(task=task):
                cfg=json.loads((ROOT/f"configs/toy-{task}.json").read_text())
                cfg["epochs"]=1
                if task=="segmentation":cfg.update(aux_weight=.1,red_index=0,nir_index=2)
                config=self.root/f"{task}.json";config.write_text(json.dumps(cfg))
                manifest=self.root/task/"manifest.jsonl";out=self.root/f"run-{task}"
                with contextlib.redirect_stdout(io.StringIO()):train(manifest,config,out,"cpu")
                result=evaluate(out/"best.pt",manifest,"test")
                self.assertEqual(set(result),{"source_a","source_b"})
                n=sum(sum(row) for row in result["source_a"]["confusion"])
                self.assertEqual(n,4*14*16 if task=="segmentation" else 4)
                latency=predict(out/"best.pt",self.root/task/"source_a_test_0.npz","source_a",out/"pred.npz",repeats=2)
                self.assertGreater(latency["median_ms"],0)
                self.assertTrue((out/"pred.npz").is_file())
                cfg["epochs"]=2;config.write_text(json.dumps(cfg))
                with contextlib.redirect_stdout(io.StringIO()):history=train(manifest,config,out,"cpu",out/"last.pt")
                self.assertEqual(history[0]["epoch"],2)
                with self.assertRaisesRegex(ValueError,"already has"):train(manifest,config,out,"cpu")
    def test_validation_catches_bad_class(self):
        self.fixture();m=self.root/"spectra/manifest.jsonl";rows=read_manifest(m)
        cfg=json.loads((ROOT/"configs/toy-spectra.json").read_text())
        cfg["heads"]["source_a"]=1
        with self.assertRaisesRegex(ValueError,"Out-of-range"):validate(rows,cfg)
    def test_csv_adapter_and_cpu_baseline(self):
        dest=self.root/"csv";result=spectra_csv(ROOT/"data/samples/synthetic-spectra.csv",dest)
        self.assertEqual(result["samples"],20)
        cfg=json.loads((ROOT/"configs/toy-spectra.json").read_text());cfg["heads"]={"source_a":2}
        validate(read_manifest(dest/"manifest.jsonl"),cfg)
        result=baseline(dest/"manifest.jsonl","source_a","svm",self.root/"svm")
        self.assertEqual(result["test"]["macro_f1"],1.)
    def test_train_only_normalization(self):
        from waste_lab.train import normalizer
        self.fixture();rows=read_manifest(self.root/"spectra/manifest.jsonl")
        training=[r for r in rows if r["split"]=="train"]
        before=normalizer(training,1)
        for r in rows:
            if r["split"]=="test":
                x,y=sample(r);np.savez(r["path"],x=x+10000,y=y)
        after=normalizer(training,1)
        np.testing.assert_array_equal(before[0],after[0])
    def test_download_checksum_and_cap(self):
        import hashlib
        payload=b"real test payload"
        class Response(io.BytesIO):
            headers={"Content-Length":str(len(payload))}
        with patch("urllib.request.urlopen",return_value=Response(payload)):
            result=download("https://example.invalid/file",self.root/"ok",100,"md5:"+hashlib.md5(payload).hexdigest())
            self.assertEqual(result["bytes"],len(payload))
        with patch("urllib.request.urlopen",return_value=Response(payload)):
            with self.assertRaisesRegex(ValueError,"size cap"):download("https://example.invalid/file",self.root/"large",1)
        self.assertFalse((self.root/"large.part").exists())
        with patch("urllib.request.urlopen",return_value=Response(payload)):
            with self.assertRaisesRegex(ValueError,"Checksum"):download("https://example.invalid/file",self.root/"bad",100,"md5:000")
        self.assertFalse((self.root/"bad").exists())
    def test_absent_class_metrics(self):
        result=scores(np.array([[2,0],[0,0]]))
        self.assertIsNone(result["f1"][1]);self.assertEqual(result["macro_f1"],1.)

if __name__=="__main__":unittest.main()

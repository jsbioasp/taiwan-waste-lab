"""Run only after conversion to Ultralytics YOLO format and site/flight split audit."""
import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument("mode",choices=["train","predict"])
p.add_argument("--model",default="yolo11n.pt"); p.add_argument("--data")
p.add_argument("--source"); p.add_argument("--epochs",type=int,default=30)
p.add_argument("--device",default="cpu"); p.add_argument("--out",default="runs/aerial")
a=p.parse_args(); model=YOLO(a.model)
if a.mode=="train":
    if not a.data:p.error("--data is required (YOLO dataset YAML)")
    model.train(data=a.data,epochs=a.epochs,imgsz=640,batch=4,device=a.device,
                workers=0,project=a.out,name="train",exist_ok=False,seed=42)
else:
    if not a.source:p.error("--source is required")
    model.predict(source=a.source,device=a.device,save=True,project=a.out,name="predict")

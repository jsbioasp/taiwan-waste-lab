# Prep: four weeks before the research paths

The course gave you image classification, CNNs and pretrained models. The research paths need three more things: **object detection** (drawing boxes around things), **segmentation** (labelling every pixel), and **honest evaluation** (scores you can trust). These four weeks fill that gap, using free courses, tutorials and small projects. Each project uses data close to what the paths use.

Plan on about **10 hours a week**. Go in order if you can, but it's your call: Week 4 has two options, depending on which path you're leaning toward. New terms are linked or explained where they first appear; the [glossary](PATHS.md#new-terms) has the rest.

**Each week has the same four parts:**
- **Learn:** lectures and reading (about 3 h)
- **Practice:** a guided tutorial or exercise set, an "assignment" with a known right answer (about 3 h)
- **Build:** a small project of your own, written up in a short README (about 4 h)
- **Done when:** the checklist for moving on

At the end of each week, add an entry to [`LOG.md`](LOG.md): what you built, your numbers, and one thing that confused you.

**Main resources:**

| Resource | What it's for |
|---|---|
| [Learn PyTorch (Zero to Mastery)](https://www.learnpytorch.io/) | Free online PyTorch book. Chapters 03–06 (each ends with exercises) and 09 are used here |
| [EECS 498 "Deep Learning for Computer Vision" lectures](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) (Justin Johnson, University of Michigan) | Clear, thorough lecture videos. The course website is offline now; the videos are still up |
| [CS231n](https://cs231n.stanford.edu/) (Stanford) and its [course notes](https://cs231n.github.io/) | A second explanation when a lecture doesn't click |
| [PyTorch tutorials](https://pytorch.org/tutorials/) | Official step-by-step notebooks |
| [Ultralytics docs](https://docs.ultralytics.com/) | Training and evaluating YOLO detectors |
| [Kaggle](https://www.kaggle.com/) or [Colab](https://colab.research.google.com/) | Free GPUs (setup in [`STUDENT_START.md`](STUDENT_START.md)) |

---

## Day 0 (optional): refresher

If CNNs feel rusty, spend a day here first; otherwise skip it. You've done this before in the course.
- Re-run your course notebooks **M04-lab2** (build a CNN) and **M04-lab3** (CNN experiments), or
- Work through [Learn PyTorch 03: Computer Vision](https://www.learnpytorch.io/03_pytorch_computer_vision/) (a small CNN on FashionMNIST).
- Want depth instead of speed? [CS231n Assignment #2](https://cs231n.github.io/assignments2026/assignment2/) has you write batch norm, dropout and convolutional layers yourself. It's optional and takes much longer than a day.

---

## Week 1 — Transfer learning, done fairly

**Why:** every path starts from a model someone else trained. This week is about doing that well, and proving with a fair comparison that it actually helped.

**Learn**
- [Learn PyTorch 04: Custom Datasets](https://www.learnpytorch.io/04_pytorch_custom_datasets/), [05: Going Modular](https://www.learnpytorch.io/05_pytorch_going_modular/), [06: Transfer Learning](https://www.learnpytorch.io/06_pytorch_transfer_learning/)
- EECS 498 Lecture 8 (CNN Architectures) and Lecture 11 (Training Neural Networks II), from the [playlist](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)
- CS231n notes: [Transfer Learning](https://cs231n.github.io/transfer-learning/)
- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/): practical habits worth copying
- Optional: [fast.ai Lesson 1](https://course.fast.ai/Lessons/lesson1.html), a top-down, practitioner's view of fine-tuning

**Practice**
- PyTorch tutorial: [Transfer Learning for Computer Vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). Compare fine-tuning the whole network with freezing it and training only the last layer.
- The exercises at the end of Learn PyTorch 06.
- [timm](https://huggingface.co/docs/timm/index), a big library of pretrained image models: find a ResNet with `timm.list_models`, load it with `timm.create_model(..., pretrained=True)`, and use the matching image preprocessing.

**Build: a land-cover classifier from Sentinel-2 images**

[EuroSAT](https://github.com/phelber/eurosat) has 27,000 small [Sentinel-2](https://sentiwiki.copernicus.eu/web/s2-mission) image patches in 10 land-cover classes (forest, river, industrial …). torchvision can download it for you ([`torchvision.datasets.EuroSAT`](https://pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html), RGB version). It's the same satellite Paths B and C use.

Train three models on **one fixed train/val/test split**:
1. a small CNN from scratch
2. a pretrained ResNet with its backbone **frozen** (only the last layer trains)
3. the same ResNet, **fully fine-tuned**

Report accuracy and macro-F1 on the test set, each with a 95% interval: for accuracy *p* on *n* test images, *p* ± 1.96·√(*p*(1−*p*)/*n*). Include a confusion matrix for the best model and one paragraph on whether pretraining was worth it here.

**Done when**
- [ ] Three runs on one split, results table with intervals
- [ ] Test set used once, at the end, after choosing settings on validation
- [ ] Log entry written

---

## Week 2 — How detectors work, and how they're scored

**Why:** Path A is a detection problem, and detection scores are easy to misread. Understand the metric before you trust a number.

**Learn**
- EECS 498 Lecture 15 (Object Detection) and Lecture 16 (Detection and Segmentation), from the [playlist](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)
- Or CS231n's lecture on object detection and image segmentation (slides on the [schedule page](https://cs231n.stanford.edu/schedule.html))
- Ultralytics: [YOLO performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/): IoU, precision, recall and mAP explained

**Practice**
- PyTorch tutorial: [TorchVision Object Detection Finetuning](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). Fine-tune Faster R-CNN, a classic two-stage detector, on a small pedestrian dataset (Penn-Fudan).
- Write your own **IoU** (how much two boxes overlap) and **NMS** (removing duplicate boxes for the same object). Check them against [`torchvision.ops.box_iou`](https://pytorch.org/vision/stable/generated/torchvision.ops.box_iou.html) and [`torchvision.ops.nms`](https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html) on random boxes.
- On paper, no calculus:
  - the IoU of two boxes
  - NMS on 5 overlapping boxes at an IoU threshold of 0.5
  - the precision-recall curve and average precision for 10 ranked detections against 6 true objects
  - why mAP@0.5:0.95 rewards tighter boxes than mAP@0.5

**Build: "How a detector's score can mislead you"**

A one-page note in your own words, covering:
- near-duplicate video frames leaking between train and test (and why you split by flight)
- small objects
- what changes between mAP@0.5 and mAP@0.5:0.95
- a score that looks good overall but is bad for the class you care about

You'll reuse this in Path A.

**Done when**
- [ ] Faster R-CNN tutorial runs end to end
- [ ] Your IoU and NMS match torchvision
- [ ] Paper exercises done; the one-page note written
- [ ] Log entry written

---

## Week 3 — Train a real detector

**Why:** this is a dress rehearsal for Path A: aerial litter, tiny objects, Kaggle GPU, honest evaluation.

**Learn**
- Ultralytics docs: [Quickstart](https://docs.ultralytics.com/quickstart/), [Train](https://docs.ultralytics.com/modes/train/), [Val](https://docs.ultralytics.com/modes/val/), [Predict](https://docs.ultralytics.com/modes/predict/), [Export](https://docs.ultralytics.com/modes/export/)
- [SAHI tiled inference](https://docs.ultralytics.com/guides/sahi-tiled-inference/): cutting big images into tiles so small objects stay visible
- How YOLO datasets are laid out: [Ultralytics detection datasets](https://docs.ultralytics.com/datasets/detect/)

**Practice**
- Run a pretrained YOLO model on a few of your own photos and read the boxes it outputs.

**Build: a drone litter detector**

Use [UAVVaste](https://uavvaste.github.io/) ([GitHub](https://github.com/UAVVaste/UAVVaste)): 772 drone images with 3,718 litter annotations in COCO format, which you'll convert to YOLO's format. The repo's code is Apache-2.0; check the dataset's own terms and note them in your README. An aerial-trash dataset from [Roboflow Universe](https://universe.roboflow.com/) also works, with its license noted.
1. Split the data. If images come from the same place or flight, keep them together, and write down how you split.
2. Do a 5-epoch smoke test, then a full run until validation mAP stops improving.
3. Report mAP@0.5, mAP@0.5:0.95 and per-class AP. Explain the precision-recall curve Ultralytics draws, in your own words.
4. Collect 10 failures (misses, false alarms, bad boxes) and sort them by cause: small object, occlusion, look-alike, label error …
5. Run it on 20 of your own photos and say honestly how it does.
6. Optional: try SAHI tiling on your worst small-object failures; export to ONNX.

**Done when**
- [ ] Trained detector with a metrics table and the PR curve explained
- [ ] 10-failure error analysis and the own-photo test
- [ ] Log entry written

---

## Week 4 — Pick by path

### Option 1: Segmentation with U-Net (Path B; useful for A and C too)

**Why:** MARIDA, Path B's dataset, labels every pixel, and the published debris model is a U-Net variant.

**Learn**
- The [U-Net paper](https://arxiv.org/abs/1505.04597) (Ronneberger et al. 2015). Read the introduction and study Figure 1; skip the rest.
- The segmentation half of EECS 498 Lecture 16
- [segmentation_models_pytorch](https://smp.readthedocs.io/en/latest/) docs ([GitHub](https://github.com/qubvel-org/segmentation_models.pytorch)), especially the [losses page](https://smp.readthedocs.io/en/latest/losses.html) (Dice loss)

**Practice**
- Build a U-Net with segmentation_models_pytorch in a few lines and train it on a toy batch.
- Read MARIDA's own U-Net baseline code in the [marine-debris repo](https://github.com/marine-debris/marine-debris.github.io) (`semantic_segmentation/unet/`).

**Build: a U-Net on MARIDA**

Get MARIDA with this repo's `scripts/fetch_starter_data.py` and `scripts/prepare_marida.py` (Path B in the [README](../README.md#path-b--satellite-debris-sentinel-2)).
- The input has 11 bands, not 3, so check how your U-Net handles extra input channels.
- Ignore unlabelled pixels (`-1`).
- Try Dice + cross-entropy loss.
- Compare per-class F1, especially Marine Debris, with the repo's random-forest baseline (test macro-F1 0.50, debris F1 0.69). Remember that debris is a tiny fraction of the pixels.

**Done when**
- [ ] U-Net trained on the official split, per-class F1 table next to the random forest's
- [ ] One paragraph: did the deep model beat the baseline, and on which classes?
- [ ] Log entry written

### Option 2: Gridded data and tabular models (Path D)

**Why:** Path D is about satellite gas measurements and station data, more like the shelter project than like image work.

**Learn**
- The [xarray tutorial](https://tutorial.xarray.dev/): working with gridded data like NetCDF files ([xarray docs](https://docs.xarray.dev/en/stable/))
- scikit-learn [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html): keeping each group (here, each station) in one fold

**Practice**
- Find a Sentinel-5P NO₂ file with the snippet in the [README's Path D](../README.md#path-d--detecting-chemicals-air-pollution-sentinel-5p), download it, open it with xarray and plot it over Taiwan.

**Build: an NO₂ map of Taiwan**

Average a month of NO₂ and map it. If you have a MOENV API key, pair a few stations with the satellite values and fit a linear regression, scored with `GroupKFold` by station.

**Done when**
- [ ] A monthly NO₂ map with a short description of the hotspots
- [ ] If you have station data: a held-out-station score
- [ ] Log entry written

---

## Extras

- **Share a demo:** [Learn PyTorch 09: Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/) and the [Gradio quickstart](https://www.gradio.app/guides/quickstart). Host it on [Hugging Face Spaces](https://huggingface.co/spaces).
- **More fast.ai:** the full [Practical Deep Learning](https://course.fast.ai/) course
- **More depth:** CS231n [course notes](https://cs231n.github.io/) and assignments; the remaining EECS 498 lectures

## Left out on purpose

- **CNN from scratch:** you did it in the course (M03, M04); Day 0 covers a refresher.
- **Mask R-CNN instance segmentation:** no current path needs a mask for each individual object.
- **GANs and style transfer:** no current path uses them. The course already covered network visualization (M04's Deep Dream and interpretability labs).
- **Hand-derived gradients:** the course favours understanding over derivations. The paper exercises here are about evaluation (IoU, NMS, average precision), because those are what you'll use to judge your own results.

When you're done, pick a path in [`PATHS.md`](PATHS.md).

# Prep: one week for your path

The course covered image classification, CNNs and pretrained models. Each research path needs one more skill on top of that. Pick your path below and spend about a week (8–10 hours) on its prep. You don't need the other paths' weeks, though you're welcome to do more than one.

Each week has the same parts:
- **From your course notebooks:** the course notebooks that cover the background. Copies are in [`course-notebooks/`](../course-notebooks/README.md) and open straight in Colab; review the ones that feel rusty. The M05/M06 ones need the course data bundles from your instructor
- **Learn:** a lecture or two and some reading
- **Practice:** a guided tutorial with a known right answer
- **Build:** a small project of your own, with a short README
- **Done when:** the checklist for moving on to the path

When you finish, add an entry to [`LOG.md`](LOG.md): what you built, your numbers, and one thing that confused you. New terms are linked where they first appear; the [glossary](PATHS.md#new-terms) has the rest.

| Path | Skill you'll learn | Prep project | Free Kaggle / Colab? |
|---|---|---|---|
| [A. Drone litter](#path-a-prep--object-detection) | Object detection (drawing boxes around things) | A YOLO litter detector on drone photos | **Both work.** Needs a GPU |
| [B. Satellite debris](#path-b-prep--segmentation) | Segmentation (labelling every pixel) | A U-Net on MARIDA satellite patches | **Both work.** Needs a GPU |
| [C. Dump sites](#path-c-prep--transfer-learning-on-image-patches) | Transfer learning on image patches | A land-cover classifier on Sentinel-2 patches | **Both work.** A GPU helps; the data is small |
| [D. Detecting chemicals: air pollution](#path-d-prep--gridded-data-and-tabular-models) | Gridded data and tabular models | An NO₂ map of Taiwan | **Both work.** CPU is enough, and so is a laptop |

**Rusty on CNNs?** Before your week, re-run your course notebooks [**M04-lab2**](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-lab2_build_a_cnn.ipynb) (build a CNN) and [**M04-lab3**](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-lab3_cnn_experiments.ipynb), or work through [Learn PyTorch 03](https://www.learnpytorch.io/03_pytorch_computer_vision/).

## Free compute: Kaggle or Colab

Every prep week runs on either a free [Kaggle](https://www.kaggle.com/) account or free [Google Colab](https://colab.research.google.com/). Setup for both is in [`STUDENT_START.md`](STUDENT_START.md).

| | Kaggle (free) | Colab (free) |
|---|---|---|
| GPU | A weekly GPU allowance, commonly reported as about 30 hours. Your account settings show yours | Available when Google has spare capacity; Colab says resources are "not guaranteed" and limits "fluctuate" ([Colab FAQ](https://research.google.com/colaboratory/faq.html)) |
| How long a session lasts | Reports say 9–12 hours per session | "At most 12 hours, depending on availability"; idle sessions are shut down |
| Keeping your files | Upload data as a private Kaggle Dataset; save a notebook version to keep its outputs | Everything is wiped when the session ends; save data and checkpoints to Google Drive |
| Internet (to install packages and download data) | Switch it on in notebook settings (may need phone verification) | On by default |

**Rule of thumb:** use Kaggle for longer GPU training runs, because its weekly hours are predictable. Use Colab for quick experiments, or when you've used up Kaggle's hours. On either one, save checkpoints so a disconnect doesn't cost you the run.

---

## Path A prep — Object detection

**Compute:** Kaggle or Colab, both with a GPU (training YOLO on a CPU is too slow). The dataset is 3 GB. On Kaggle, upload it once as a private Dataset. On Colab, keep it in Google Drive so you don't re-download it every session.

**From your course notebooks**
- [`M04-4_pretrained_models`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-4_pretrained_models.ipynb): starting from pretrained weights, which is how YOLO training starts too
- [`M04-deep_transfer_learning`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-deep_transfer_learning.ipynb): fine-tuning a pretrained model, frozen vs fully trained
- [`M04-deep_data_augmentation`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-deep_data_augmentation.ipynb): YOLO relies heavily on augmentation
- [`M02-6_evaluation`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M02-6_evaluation.ipynb): precision, recall and F1, the building blocks of mAP
- [`M03-lab3_mnist_evaluation`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M03-lab3_mnist_evaluation.ipynb): digging into the mistakes, which is practice for your 10-failure analysis
- [`M05-lab5_audit_your_model`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M05-lab5_audit_your_model.ipynb): group-aware splits and leakage, or why images of the same place stay together

**Learn**
- EECS 498 *Lecture 15: Object Detection*, from Justin Johnson's [Deep Learning for Computer Vision lectures](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) (University of Michigan)
- Ultralytics: [YOLO performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/): what IoU (box overlap), precision, recall and mAP mean

**Practice**
- Ultralytics [Quickstart](https://docs.ultralytics.com/quickstart/): run a pretrained YOLO on a few of your own photos.
- Ultralytics [Train mode](https://docs.ultralytics.com/modes/train/): train on the tiny `coco8` dataset to see the whole loop work.

**Build: a drone litter detector**

[UAVVaste](https://uavvaste.github.io/) ([GitHub](https://github.com/UAVVaste/UAVVaste)) has 772 drone images with 3,718 litter boxes in COCO format; convert them to YOLO's format. It's CC BY 4.0 on [Zenodo](https://zenodo.org/records/8214061) (a 3 GB zip), so credit the authors in your README.
1. Split into train/val/test. Keep images from the same place together if you can tell, and write down how you split.
2. Train a small YOLO model until validation mAP stops improving.
3. Report mAP@0.5 and mAP@0.5:0.95.
4. Collect 10 failures and sort them by cause: tiny object, look-alike, missed, bad box …
5. Try it on 20 photos you took yourself, and say honestly how it does.

**Done when**
- [ ] Trained detector with a metrics table
- [ ] 10-failure error analysis and the own-photo test
- [ ] Log entry written

---

## Path B prep — Segmentation

**Compute:** Kaggle or Colab, both with a GPU. MARIDA is a 1.16 GB download. On Kaggle, upload it as a private Dataset or switch on Internet and download it. On Colab, keep it in Google Drive.

**From your course notebooks**
- [`M04-2_cnn_building_blocks`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-2_cnn_building_blocks.ipynb): convolution and pooling, the U-Net's encoder half
- [`M04-3_cnn_architectures`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-3_cnn_architectures.ipynb): ResNet's skip connections; U-Net uses skip connections too
- [`M05-4_meet_aac`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M05-4_meet_aac.ipynb) and [`M05-lab4_cross_dataset`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M05-lab4_cross_dataset.ipynb): a model that fails on a second dataset, the same problem as MARIDA → Taiwan
- [`M06-3_multitask_with_photos`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-3_multitask_with_photos.ipynb) and [`M06-lab2_multitask_handson`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-lab2_multitask_handson.ipynb): a shared trunk with separate heads and an auxiliary target, the design behind this repo's `SmallModel`
- [`M02-6_evaluation`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M02-6_evaluation.ipynb): per-class precision, recall and F1, for comparing your U-Net with the random forest

**Learn**
- EECS 498 *Lecture 16: Detection and Segmentation* (the segmentation part), from the [lecture playlist](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r)
- The [U-Net paper](https://arxiv.org/abs/1505.04597): read the introduction and study Figure 1; skip the rest.
- [segmentation_models_pytorch](https://smp.readthedocs.io/en/latest/) docs, especially the [losses page](https://smp.readthedocs.io/en/latest/losses.html) (Dice loss)

**Practice**
- Build a U-Net with segmentation_models_pytorch in a few lines and train it on a toy batch.
- Skim MARIDA's own U-Net code in the [marine-debris repo](https://github.com/marine-debris/marine-debris.github.io) (`semantic_segmentation/unet/`).

**Build: a U-Net on MARIDA**

Get [MARIDA](https://marine-debris.github.io/) with this repo's `scripts/fetch_starter_data.py` and `scripts/prepare_marida.py` (see [Path B in the README](../README.md#path-b--satellite-debris-sentinel-2)).
- The input has 11 bands, not 3, so check how your U-Net handles that.
- Ignore unlabelled pixels (`-1`).
- Try Dice + cross-entropy loss.
- Compare per-class F1, especially Marine Debris, with the repo's random forest (test macro-F1 0.50, debris F1 0.69).

**Done when**
- [ ] U-Net trained on the official split, per-class F1 next to the random forest's
- [ ] One paragraph: did it beat the baseline, and on which classes?
- [ ] Log entry written

---

## Path C prep — Transfer learning on image patches

**Compute:** Kaggle or Colab. EuroSAT is only 94 MB and torchvision downloads it in seconds, so either is easy. A GPU makes fine-tuning take minutes instead of a long wait; the from-scratch CNN is fine on CPU.

**From your course notebooks**
- [`M04-lab2_build_a_cnn`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-lab2_build_a_cnn.ipynb) and [`M04-lab3_cnn_experiments`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-lab3_cnn_experiments.ipynb): your from-scratch CNN (run 1 below)
- [`M04-4_pretrained_models`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-4_pretrained_models.ipynb) and [`M04-lab4_pretrained_features`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-lab4_pretrained_features.ipynb): pretrained models and what their feature maps show
- [`M04-deep_transfer_learning`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-deep_transfer_learning.ipynb): freezing vs fine-tuning (runs 2 and 3 below)
- [`M04-deep_data_augmentation`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M04-deep_data_augmentation.ipynb): augmentation for small image datasets
- [`M06-1_photos_and_adoption`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-1_photos_and_adoption.ipynb) and [`M06-lab1_photo_features`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-lab1_photo_features.ipynb): using a pretrained CNN's features as inputs to another model, a handy shortcut for dump-site image chips later

**Learn**
- [Learn PyTorch 06: Transfer Learning](https://www.learnpytorch.io/06_pytorch_transfer_learning/)
- CS231n notes: [Transfer Learning](https://cs231n.github.io/transfer-learning/)

**Practice**
- PyTorch tutorial: [Transfer Learning for Computer Vision](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). Compare fine-tuning the whole network with freezing it and training only the last layer.

**Build: a land-cover classifier**

[EuroSAT](https://github.com/phelber/eurosat) has 27,000 [Sentinel-2](https://sentiwiki.copernicus.eu/web/s2-mission) image patches in 10 land-cover classes, and torchvision downloads it for you ([`torchvision.datasets.EuroSAT`](https://pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html)). Classifying satellite patches is exactly what Path C does later with dump sites.

On **one fixed train/val/test split**, train:
1. a small CNN from scratch
2. a pretrained ResNet with its backbone frozen
3. the same ResNet, fully fine-tuned

Report test accuracy with a 95% interval: for accuracy *p* on *n* images, *p* ± 1.96·√(*p*(1−*p*)/*n*). Add one paragraph on whether pretraining was worth it.

**Done when**
- [ ] Three runs on one split, results table with intervals
- [ ] Test set used once, at the end
- [ ] Log entry written

---

## Path D prep — Gridded data and tabular models

**Compute:** Kaggle or Colab on CPU, or your own laptop; there's no GPU work here. Watch the disk space, though: each Sentinel-5P NO₂ file is about 600 MB, so a month of daily files is roughly 19 GB. Process one file at a time: cut it down to Taiwan, save that small piece, and delete the original. On Kaggle, switch Internet on to download.

**From your course notebooks**
- [`M01-6_numpy`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M01-6_numpy.ipynb) and [`M01-lab5_numpy_matplotlib`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M01-lab5_numpy_matplotlib.ipynb): arrays and plotting. xarray is built on NumPy, and your NO₂ map is a matplotlib plot
- [`M02-3_linear_regression`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M02-3_linear_regression.ipynb) and [`M02-lab2_linear_regression`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M02-lab2_linear_regression.ipynb): linear regression and MSE, for satellite vs station
- [`M05-lab5_audit_your_model`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M05-lab5_audit_your_model.ipynb): group-aware cross-validation, the idea behind `GroupKFold` by station
- [`M06-5_taiwan_deployment`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-5_taiwan_deployment.ipynb) and [`M06-lab3_taiwan_handson`](https://colab.research.google.com/github/jsbioasp/taiwan-waste-lab/blob/main/course-notebooks/M06-lab3_taiwan_handson.ipynb): checking a model against official numbers with Spearman rank correlation, the same kind of check as satellite vs ground

**Learn**
- The [xarray tutorial](https://tutorial.xarray.dev/): working with gridded data such as NetCDF files
- scikit-learn [`GroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html): keeping each station's data in one fold

**Practice**
- Find a [Sentinel-5P](https://sentiwiki.copernicus.eu/web/s5p-mission) NO₂ file with the snippet in the [README's Path D](../README.md#path-d--detecting-chemicals-air-pollution-sentinel-5p), download it, open it with xarray and plot it over Taiwan.

**Build: an NO₂ map of Taiwan**

Average a month of NO₂ and map it, with a short description of the hotspots. If you have a MOENV API key, pair a few stations with the satellite values and fit a linear regression, scored with `GroupKFold` by station.

**Done when**
- [ ] Monthly NO₂ map with the hotspots described
- [ ] If you have station data: a held-out-station score
- [ ] Log entry written

---

## Going further (optional)

- Another path's prep week
- [Learn PyTorch 04](https://www.learnpytorch.io/04_pytorch_custom_datasets/) and [05](https://www.learnpytorch.io/05_pytorch_going_modular/): custom datasets and tidy project code, useful on any path
- Andrej Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- Share a demo: [Learn PyTorch 09](https://www.learnpytorch.io/09_pytorch_model_deployment/), the [Gradio quickstart](https://www.gradio.app/guides/quickstart) and [Hugging Face Spaces](https://huggingface.co/spaces)

When you're done, head to your path in [`PATHS.md`](PATHS.md).

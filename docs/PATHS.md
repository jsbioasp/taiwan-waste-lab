# Research paths

This project is yours to steer. Below are four paths that fit the data and tools in or around this repo, plus guidance for proposing your own. A–C are about waste; D is about chemicals in Taiwan's air. They aren't assignments, and they don't have to be done in order. Pick one, combine two, or go somewhere else if you find a better question.

**How this works:**
- **Choose a path, or write a one-paragraph proposal for your own,** and check it with your instructor before you sink a lot of time into it.
- **Keep a research log** in [`LOG.md`](LOG.md): what you tried, what happened, what you'll try next. Short entries are fine. Dead ends count as results; write them down.
- **Check in** when you've picked a direction, when you get a first result, and whenever you're stuck for more than a session.
- Each path has the same sections: the question, what's already here, a few ways in, open questions, things to watch for, and what a solid result looks like. The ways in are suggestions, not steps.

The ground rules at the bottom apply to every path.

---

## Path A — Taiwan drone litter

**The question:** a detector pretrained on everyday photos (COCO) knows nothing about trash on Taiwan's coast. How much can it learn from about 1,700 Taiwan drone frames, and what actually helps?

**What's already here:**
- HAIDA and HAIDA2 from NTOU: 1,710 frames from 18 flights, 8,037 boxes labelled `garbage` or `bottle`. Labels are Pascal VOC XML. See `LOG.md` for the full breakdown.
- `scripts/aerial.py` (trains or predicts with YOLO) and `scripts/audit_aerial_split.py` (checks for leakage). It's set up for Ultralytics YOLO11 (`yolo11n.pt`), but other detectors, and other litter datasets to learn from, are laid out in [Models to choose from](#models-to-choose-from-path-a) below.
- Kaggle for free GPU training.

**Ways in:**
- Get *any* detector trained end to end first, even a bad one. Convert the labels, split by flight, train a small YOLO model for a few epochs and look at its predictions. You'll learn more from that than from planning.
- Start from the data instead. Look at a few hundred frames. How small is the litter? How different are the flights? What would *you* find hard to label? That will tell you which experiments matter.
- Start from a transfer-learning question you care about, like "Is it enough to retrain just the last layers?", and design around it.

**Open questions you could chase:**
- How much does input size or tiling matter, given the litter is about 0.2% of each frame?
- Does freezing the pretrained backbone help or hurt with this little data?
- Does adding HAIDA2 (almost all `garbage`) help `bottle` detection, or confuse it?
- How does a model trained on drone shots do on ground-level photos you take yourself at a Taiwan beach?
- Which flights does it fail on, and why? Glare? Rocks? Waves?

**Watch out for:**
- **Video frames.** Frames are cut from drone videos, so neighbouring frames are near-copies. Split by flight folder (`data1`, `data13`, …), never by frame, or your test score will be too good to be true.
- **HAIDA's compression.** The zips use PPMd compression. Python's `zipfile` can't open them; use `bsdtar -xf` on a Mac or 7-Zip on Windows.
- **Licensing.** Keep HAIDA's images out of the GitHub repo. Upload them as a *private* Kaggle Dataset and cite the thesis.

**A solid result looks like:** a table of a few runs that differ in one thing each, scored on flights the model never saw (mAP@0.5 overall and per class), some example detections (good and bad), and a paragraph on what transferred and what didn't.

### Models to choose from (Path A)

There's no single right choice. Scores marked † come from `research/land_waste_aerial.md` and `research/taiwan_datasets.md`. They're leads from the papers, not checked here, and each uses a different dataset, split and input size, so don't compare them row against row.

| Model | What it is | In this repo? | Compute | What's reported |
|---|---|---|---|---|
| **YOLO11** (`n`, `s`, `m` sizes) | One-stage detector: fast, with lots of tutorials | Yes: `scripts/aerial.py` (untested on real data; `imgsz` fixed at 640) | `n`/`s` train comfortably on a Kaggle GPU | The notes call a YOLO `s`/`m` at 608–1024 px the documented sweet spot for drone litter. DroneWaste: YOLOv8x 38.2 mAP@50 over 20 material classes†. UAVVaste: YOLOv4 0.785 mAP@50†. HAIDA's own paper used YOLOv3/v4/Scaled-YOLOv4† |
| **YOLO + tiling** ([SAHI](https://github.com/obss/sahi), MIT) | Cut each frame into overlapping tiles so small objects stay big enough to see, then merge the detections | No | Same training cost; inference is slower (more tiles) | No numbers in the notes, but it targets the known weak spot: on UAVVaste, small-object AP was 0.28 vs 0.61 for large objects† |
| **Faster R-CNN** (torchvision, BSD-3) | Two-stage detector with a feature pyramid; slower, often strong on small objects | No; built into torchvision with COCO weights | Kaggle GPU; slower than YOLO | DroneWaste: 36.5 mAP@50 vs 38.2 for YOLOv8x†. Sun et al.'s global dumpsite detector is built on it† |
| **RT-DETR** ([official code](https://github.com/lyuwenyu/RT-DETR), Apache-2.0) | Transformer-based detector; no anchor boxes to tune | No | Kaggle GPU; heavier than YOLO `n`/`s` | No RT-DETR litter numbers in the notes. The older DETR trailed EfficientDet on litter (AP@0.5 50.7 vs 65.5)† |
| **Zero-shot, text-prompted** ([Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), Apache-2.0; [YOLO-World](https://github.com/AILab-CVC/YOLO-World), GPL-3.0) | Type "plastic bottle" and it looks for one, with **no training** | No | Inference only; GPU recommended | None in the notes. Useful as a "how far does zero training get you?" baseline before you train anything |

**Data choices are model choices too.** On the satellite side, better data beat bigger models. For drone litter you could learn from another litter dataset first, then fine-tune on HAIDA. These are all CC BY 4.0 per the notes†, with access not checked here:
- **UAVVaste:** 772 drone images, 3,716 boxes, a single litter class
- **DroneWaste** (Polimi/CERTH, 2025): 4,993 drone tiles labelled with **20 material classes**. That's the closest thing to a material label we have.
- **MDImageNet** (Taiwan, 2026): 20,000+ coastal images, 34 litter classes. The download path is unconfirmed (an instructor to-do).

**Things to know when choosing:**
- **Small objects are the main difficulty.** HAIDA's typical box is about 90×45 px in a 1920×1080 frame, roughly 30×15 px once YOLO shrinks the frame to 640. Input size or tiling may matter more than which model you pick.
- **Check how the papers split their data.** HAIDA's paper used k-fold cross-validation. If those folds mixed frames from the same flight, its scores would be optimistic next to a flight-level split. Find out before comparing.
- **Licenses:** Ultralytics is AGPL-3.0 and YOLO-World GPL-3.0. They're fine for research and coursework, but if you publish a modified app built on them you'd have to share its source. torchvision, RT-DETR and Grounding DINO are permissive.
- **Keep a simple baseline**, such as YOLO11n at default settings, so you can tell whether a fancier model is actually better on *your* held-out flights.

---

## Path B — Why does the satellite model see clouds over Taiwan's sea?

**The question:** a model trained on MARIDA's satellite pixels labels most of Taiwan's cloud-free sea as "Clouds." What's different about Taiwan's water, and can you make a debris model work there?

**What's already here:**
- `scripts/fetch_taiwan_s2.py` downloads a Sentinel-2 crop of the Ruifang/Gongliao coast. Change the box and dates to look anywhere.
- `scripts/pixel_rf.py` and `scripts/predict_taiwan_rf.py`: the MARIDA random forest and its Taiwan run.
- A first clue: Taiwan's open water is about twice as bright as MARIDA's in the infrared bands (`LOG.md`, 2026-11-11). There are two suspects: sun glint, and the fact that MARIDA was processed with ACOLITE while our crop is Sen2Cor L2A.
- A range of models to choose from, from a seconds-long random forest to state-of-the-art deep networks. See [Models to choose from](#models-to-choose-from-path-b) below.

**Ways in:**
- **Test the clue.** If the extra brightness is sun glint, subtracting a shortwave-infrared band (B11 or B12) from the other bands should push the water back toward "water." Try it and measure what fraction of open water the model calls water before and after.
- **Match the processing.** MARIDA's reflectance was made with ACOLITE (free software) from the raw L1C product; ours came from Sen2Cor L2A. Process the Taiwan scene the MARIDA way and see whether the "Clouds" problem shrinks. If it does, it was processing, not Taiwan.
- **Label a little Taiwan data.** A model that fails on a new domain often recovers with a small amount of labelled data from that domain. A Raman project with the same problem ([raman-open-ml](https://github.com/JoulesSpace/raman-open-ml); AI-built, so treat its numbers as claims) reports 94% accuracy on its own measurements, 55% on a new measurement campaign, and 76% after fine-tuning on a small labelled slice of the new one. Try hand-labelling a few hundred Taiwan pixels (water, ships, wakes and anything that looks like debris), add them to training, and test on a *different* date or stretch of coast.
- **Compare models on the same scene.** Pick two or more from the table below, run them on the same Taiwan crop, and compare them with each other and with FDI. Do they agree on anything?
- **Change the scene, not the model.** Look for dates right after heavy rain or a typhoon, when rivers flush debris out to sea. A calm March day may simply have no debris to find.

**Open questions you could chase:**
- Is the brightness the same across dates and seasons, or does it change with the sun angle? That would point to glint.
- Does the published detector have the same problem, or does it handle Taiwan better? If it does, why might that be?
- Can you think of a check that tells you the model is right *without* Taiwan labels? (Hint: the shelter project compared its predictions with published adoption rates.)
- Does a small neural network trained on MARIDA behave differently from the random forest on Taiwan?

**Watch out for:**
- **The coastline.** Pixels that are half land, half water look like floating material. Always exclude a strip along the coast.
- **Boats and wakes.** They're bright, elongated and look a lot like debris.
- **"No debris found" can be a correct answer.** Don't tune until the model finds something.
- **Beach litter isn't floating debris.** Survey counts from beaches are a rough comparison at best.

**A solid result looks like:** before/after numbers for whatever correction you try, maps from at least two methods on the same scene, and an honest paragraph on whether anything sees debris in Taiwan, and how you'd know.

### Models to choose from (Path B)

There's no single right choice; they trade speed, data needs and effort. Scores marked † come from `research/original-combinations-report.md`. They're leads from the papers, not checked here, and they use *different protocols*, so don't compare them row against row.

| Model | What it is | In this repo? | Compute | What's reported |
|---|---|---|---|---|
| **Pixel random forest** | Classifies each pixel from its 11 band values alone | Yes: `scripts/pixel_rf.py` (run) | Seconds, CPU | Ours: MARIDA test debris F1 0.69, macro-F1 0.50. MARIDA's own random forest, which adds spectral indices and texture features: debris F1 0.80† |
| **Small CNN** | The repo's `SmallModel`: a small segmentation network with one head per dataset and an optional NDVI auxiliary head | Yes: `waste_lab train`. Copy `configs/marida-smoke.json` and raise `epochs`/`width` for a full run (`marine-template.json` also expects a MADOS head) | Minutes on a Kaggle GPU; slow on CPU | Only run on 6 patches so far |
| **U-Net from scratch** | The classic encoder-decoder segmentation network | No; MARIDA's baseline code is at [marine-debris.github.io](https://github.com/marine-debris/marine-debris.github.io) | Minutes on a T4 GPU† | MARIDA paper: debris F1 0.50, macro-F1 0.69†, *below* their random forest, probably because there are so few debris pixels |
| **Pretrained UNet++** (marinedebrisdetector, Rußwurm 2023) | A larger U-Net variant trained on MARIDA + FloatingObjects + ship negatives, with refined labels | No; install from [GitHub](https://github.com/marccoru/marinedebrisdetector) (MIT). The 655 MB weights download fine. | Inference only: about 11 min per full scene on a laptop CPU†, seconds on a GPU. You can also fine-tune it. | F1 0.749 on MARIDA test under its binary debris protocol, where MARIDA's random forest scores 0.288†. Expects 12-band Sen2Cor L2A, **the same processing as our Taiwan crop** |
| **Frozen foundation encoder + head** (CROMA, DOFA, TerraMind) | A large model pretrained on lots of satellite imagery; you train only a small head on top | No; see the [PANGAEA benchmark](https://github.com/VMarsocci/pangaea-bench) / TorchGeo | GPU; the head trains quickly | MADOS mIoU: CROMA 67.6, TerraMind-B 69.5, from-scratch U-Net 54.8†. Needs the bands mapped carefully |
| **MariNeXt / MambaMPD** | Current top models on the MADOS benchmark | No; MariNeXt code at [gkakogeorgiou/mados](https://github.com/gkakogeorgiou/mados), MambaMPD in arXiv 2609.15676† | GPU; heavier (MambaMPD is about 37 M parameters) | MADOS F1: MariNeXt about 71–76, MambaMPD 74.8† |

**Things to know when choosing:**
- **State of the art on the benchmarks is deep networks, but with more data.** The UNet++ authors credit their gains *"mainly to dataset design … not architecture"*†. On MARIDA alone, with only about 1,900 training debris pixels, a from-scratch U-Net lost to a random forest.
- **The same model can score 0.80 or 0.288** depending on the protocol (15-class vs binary debris). Compare models on *your* split and *your* Taiwan scene, not on numbers from different papers.
- **Match the processing to what the model was trained on.** MARIDA and MADOS were made with ACOLITE; the UNet++ expects Sen2Cor L2A. Feeding a model the wrong kind of reflectance is a likely way to get "Clouds" over clear water.
- **A strong baseline makes a deep model's result mean something.** Whatever you pick, keep the random forest as the "is this actually better?" check.
- `models/catalog.json` lists these and other upstream models with links.

---

## Path C — Can you see Taiwan's dump sites from space?

**The question:** MOENV publishes the locations of reported illegal dump sites. Are they actually visible in free 10 m Sentinel-2 images? If they are, they could become training labels for a Taiwan dump detector.

**What's already here:**
- `data/samples/moenv-20260909/`: 491 sites as GeoJSON points, plus a blank review CSV (`site_id`, coordinates, `image_date`, `scene_id`, `waste_visible`, `confidence`, `notes`, `group`, `split`). Each record has a county and district, land-parcel numbers, waste type (`一般` general 435, `一般含有害` general with hazardous 39, `有害` hazardous 16), a last-update date, whether soil/groundwater is also regulated, and **free-text inspection notes** (`現場狀態`). There's **no size or area field**.
- `python -m waste_lab search-sentinel` to find scenes, and `scripts/fetch_taiwan_s2.py` as a model for downloading a crop.
- A range of imagery and models to choose from. See [Models and data to choose from](#models-and-data-to-choose-from-path-c) below.

**Ways in:**
- Do a visibility study before any machine learning: pick 30–50 sites, look at imagery of each one, and mark each *visible*, *not visible* or *unsure*. That alone is a real finding.
- **Read the inspection notes first.** A quick keyword count over `現場狀態` finds 掩埋 (buried) in 29 records, 雜草 (overgrown with weeds) in 109, 堆置 (piled up) in 52, and 已清除 / 清理完成 (cleared) in 20. Buried or cleared waste won't show from above at *any* resolution. Sorting the records before picking sites (with keyword rules, or an LLM whose output you spot-check by hand) could make the visibility study far more informative.
- Pick sites to compare, e.g. "piled" vs "buried" vs "cleared", so the result says something about *which* sites are visible, not just how many.
- **Test the resolution before you need it.** [AerialWaste](https://github.com/nahitorres/AerialWaste) ([Zenodo 7034381](https://zenodo.org/records/7034381): v3, CC BY-NC-ND 4.0, images in zips of about 2.7 GB each) has thousands of labelled 20–50 cm aerial tiles of Italian dump sites, plus a data loader. Blur them down to 10 m, train a classifier on the blurred tiles, and see how much survives. That predicts what Sentinel-2 could see in Taiwan before you touch Taiwan imagery. Use only the non-Google tiles: the README says Google imagery must follow Google's terms, and those forbid ML training†.

**Open questions you could chase:**
- What fraction of listed sites are visible at 10 m? Does it depend on waste type, county, or what the inspection notes describe?
- With only a point and no area, how would you decide where a site's edges are in an image?
- What would you use as negatives: legal landfills, farmland, construction sites?
- Is 10 m enough, or does this need sharper imagery? Is any sharper imagery usable for training (see licenses below)?

**Watch out for:**
- The date field (`最後更新日`, called `LDate` in MOENV's source) is when the record was *last updated*, not when the dumping happened. 474 of the 491 records were updated between June and September 2026.
- Being on the list doesn't make a site visible waste. The notes describe sites that are buried, overgrown, cleared or being removed from the list (`解除列管`).
- A point may sit at the centre of a land parcel rather than on the waste itself.
- About 20 records have garbled Chinese text in the notes from the original file; they're flagged in `__decoding_replacement_fields`.

**A solid result looks like:** a filled-in review CSV, the visible fraction with a few example image chips, and a recommendation on whether a detector is worth building.

### Models and data to choose from (Path C)

For this path, the imagery decides more than the model. Scores marked † come from `research/original-combinations-report.md` (the land-waste and free-imagery sections) and `research/land_waste_aerial.md`. They're leads, not checked here.

**Imagery**

| Imagery | Detail | Access | Notes |
|---|---|---|---|
| **Sentinel-2 L2A** | 10 m | Free, no account (Planetary Computer; the repo's scripts use it) | Revisits every ~5 days†. The notes call 10 m "the wrong sensor for landfill-scale targets"†, and your visibility study is the Taiwan test of that claim |
| **Landsat** | 30 m | Free | Coarser, but the archive goes back decades. Useful for *when* a site appeared |
| **Sentinel-1 radar (SAR)** | Radar, not a photo | Free (Copernicus) | Sees through clouds, but it measures how rough and wet surfaces are, not their color. It needs its own processing and model |
| **NLSC aerial orthophotos** | High resolution | Free to *view* | Viewing doesn't grant training or redistribution rights (`waste_lab/catalog.json`), so check the license before training on them |
| **FORMOSAT-2/5** | Higher resolution than Sentinel-2 | Free to Taiwan academics in limited quantity, via an NSTC application† | Worth asking about if 10 m proves too coarse |
| **Google Earth / Maps** | Very high | Free to look at | **Its terms forbid using it to train or test ML models**†. Fine for eyeballing a site, not for a dataset |

**Approaches and models**

| Approach | What it is | In this repo? | What's reported |
|---|---|---|---|
| **Visibility review** (no model) | You look at each site and fill in `site-review.csv` | Yes: the CSV template | This *is* the first result; most other approaches need its labels |
| **Sort the records by their text** | Categorize the `現場狀態` notes (buried / piled / cleared …) with keyword rules or an LLM | The data, yes; no code yet | None. It's your own analysis; hand-check a sample |
| **Before/after comparison** | Compare a site across dates with simple indices (vegetation loss, bare ground) | `scripts/fetch_taiwan_s2.py` is a template | No dump-specific numbers in the notes. It needs no labels, which is its appeal |
| **Small patch classifier on Sentinel-2** | Train on image chips around visible vs non-visible sites | Yes: `waste_lab train` with `"task": "image"`, or `waste_lab baseline --kind rf` | One 2026 Sentinel-2 landfill study: 54% AP, and it needed road and industry context to be usable†. Resolution study (Molina 2024): a ResNet-50 at F1 0.86 at full detail was **still 0.79 when trained at roughly 6–16 m**, 0.43 coarser still, but a model trained on sharp images and *fed* coarse ones collapsed to 0.20†. Train at the resolution you'll use |
| **Pretrained high-res classifier** (AerialWaste models) | Trained on 20–50 cm aerial tiles of Italian dumps | No ([aerialwaste-model](https://github.com/nahitorres/aerialwaste-model); its license isn't a standard one, so read it before using) | ResNet-50+FPN F1 80.7 (v1); Swin-T F1 92.0 at 20 cm (v3)†. Only usable with sharp imagery; AerialWaste v3 is CC BY-NC-ND (no redistributing modified copies)† |
| **Foundation encoder + small head** | A large model pretrained on satellite imagery; train only a head on your chips | No; via TorchGeo | The notes name SatLAS NAIP Swin-V2-B as the closest free match for 20–50 cm tiles, Clay v1.5 (Apache-2.0), and Prithvi as 30 m scale†. Match the model to your imagery's scale |
| **Dumpsite detector** (Sun et al. 2023, Faster R-CNN based) | Draws boxes around dumps in 0.3–1 m imagery | No | Sensitivity 0.98 but precision 0.70†; its dataset is academic-use only |

**Labels beyond MOENV:** `waste_lab/catalog.json` lists other Taiwan sources you could use as negatives or extra positives: Taoyuan legal landfills (data.gov.tw 76762), Tainan dump sites (173490), contaminated sites (8817). The research notes also mention the Ministry of the Interior's land-use change points (國土利用監測): 132,254 points from 2008–2022, of which 4,748 were confirmed waste or soil-dumping violations†. That's a much larger label set, but the access path is unconfirmed.

**Things to know when choosing:**
- **Visibility first.** If buried, overgrown or cleared sites make up much of the list, no model will find them. Sort the records and review before training anything.
- **Balance and metrics.** Most image chips won't contain a dump. With a 2:1 negative split, a model that always says "no dump" scores about 67% accuracy† while finding nothing. Report precision, recall or F1 on the dump class.
- **Group by site.** Images of the same site from different dates stay in the same split.
- **Licenses decide your imagery.** Sentinel-2, Landsat and MOENV are free to use with credit. Google can't be used for training. NLSC and FORMOSAT need checking or an application.

---

## Path D — Can a satellite measure Taiwan's air pollution?

**The question:** the Sentinel-5P satellite carries a spectrometer (TROPOMI) that measures gases such as NO₂ and SO₂ from orbit. That's real chemical detection, with no lab. How well do its readings over Taiwan match what ground stations measure, and what can you learn from where they disagree?

**What's already here:**
- No code for this path yet. It's the newest and least scaffolded path, so you'd be building it.
- **Sentinel-5P** is free on Microsoft Planetary Computer (collection `sentinel-5p-l2-netcdf`, data from 2018-04-30). Products: `no2`, `so2`, `co`, `o3`, `hcho`, `ch4`, aerosol index and clouds. `waste_lab/taiwan.py` and `scripts/fetch_taiwan_s2.py` show the search-and-download pattern. The files are NetCDF, so you'll need `pip install xarray netCDF4`.
- **MOENV ground stations:** MOENV's open-data API serves air-quality data (e.g. dataset `aqx_p_432`, real-time AQI) and needs a free API key; keep the key in an environment variable, never in Git. Find the hourly historical dataset on MOENV's data portal. The research notes also list an hourly NMHC dataset (data.gov.tw 34829).
- **Low-cost sensors:** Taiwan's Civil IoT / AirBox network has thousands of cheap PM2.5 sensors. The API address in the research trail served a certificate for the wrong hostname, so find the current one on Civil IoT Taiwan's site.

**Ways in:**
- **Start with a map.** Average a month of NO₂ over Taiwan. Where are the hotspots: cities, ports, power plants, industrial zones? Does the pattern change with the season?
- **Compare with the ground.** For each MOENV station, take the satellite value over the station on each overpass day and the station's reading at the overpass hour. How well do they track each other? Then try predicting the station value from the satellite plus weather, and test on stations the model never saw.
- **Calibrate the cheap sensors.** Pair AirBox sensors with a nearby official station and correct the cheap readings. It's the same calibration-transfer problem the Raman repo studies, applied to air.
- **Use natural experiments.** Holidays, typhoon days and the early-2020 COVID period all changed emissions or weather. Can the satellite see the difference?

**Open questions you could chase:**
- Does satellite NO₂ match station NO₂ better in some seasons, regions or weather? Why?
- Which large emitters show up in long-term averages, and which don't?
- Are SO₂ or HCHO strong enough over Taiwan to measure, or mostly noise?
- How much of the gap between satellite and ground can weather (wind, rain, mixing height) explain?

**Watch out for:**
- **Column vs surface.** The satellite measures the gas in the whole column of air above a pixel; a station measures the air at ground level. They're related, not the same.
- **Pixels are kilometres wide.** The satellite sees cities and plants, not streets or single factories.
- **Clouds and quality flags.** Each product has a quality value per pixel. Read the product guide and filter before you average.
- **One overpass a day** (early afternoon) vs hourly stations. Compare like with like.
- **Split by station.** Test on stations the model never saw, or the score will flatter you.
- **Methane is the weak target here.** The research notes found no public satellite methane plume over Taiwan.

**A solid result looks like:** a pollution map, a satellite-vs-station comparison scored on held-out stations, and an honest paragraph on where the satellite agrees with the ground and where it doesn't.

### Models and data to choose from (Path D)

This path is mostly tabular data, the same kind of problem as the PetFinder and shelter work. There are no Taiwan numbers in the research notes, so finding what's been published is part of the work.

**Data**

| Data | What it measures | Access | Notes |
|---|---|---|---|
| **Sentinel-5P / TROPOMI** | Gas columns: NO₂, SO₂, CO, O₃, HCHO, CH₄; aerosol index | Free, no account (Planetary Computer) | Daily, early-afternoon overpass, km-scale pixels; NetCDF files |
| **Sentinel-3 aerosol** | Aerosol optical depth (haze/particles) | Free (Planetary Computer `sentinel-3-synergy-aod-l2-netcdf`, from 2020-04-16) | A possible route to PM2.5 |
| **MOENV stations** | Ground-level pollutants, hourly | Free API key | The ground truth for this path |
| **Civil IoT / AirBox** | PM2.5 from low-cost sensors, dense coverage | Check the current API | Noisier; good for calibration projects |
| **Weather** (e.g. ERA5 reanalysis, Taiwan's Central Weather Administration) | Wind, rain, temperature, mixing height | Free; access not checked here | Often explains a lot of the satellite-ground gap |

**Approaches and models**

| Approach | In this repo? | Notes |
|---|---|---|
| **Direct comparison** (correlation, scatter plots) | No code needed | Always do this first |
| **Linear regression** (satellite + weather → station) | scikit-learn is already a dependency | A strong, readable baseline |
| **Random forest / gradient boosting** | scikit-learn | The tabular workhorses from the shelter project |
| **Small neural network (MLP)** | PyTorch is already a dependency | Worth it only if it beats the trees on held-out stations |

**Things to know when choosing:**
- **This path detects chemicals in air, not waste.** It's a legitimate project in its own right, not a way to identify plastics.
- **Group by station and by time.** Neighbouring hours and days are correlated, just like neighbouring video frames.
- **Start simple.** If linear regression with weather gets close, that's a finding.

---

## Your own path

Good proposals tend to have:
- **Taiwan in it:** Taiwan data, or at least running a model on Taiwan images.
- **Something you can measure,** and a way to keep the test data separate (by flight, scene, site or date).
- **A first version you can run in a week** on free compute.

Some seeds:
- Compare the *material mix* your drone detector finds (bottle vs other garbage) with Taiwan's official cleanup statistics. According to the research notes, the Ocean Conservation Administration publishes monthly tonnage by item type on data.gov.tw (dataset 172912). Check that first.
- Try to get MDImageNet, a Taiwan coastal image set with 34 litter classes, and see what a richer label set changes.
- Build a small demo: a local Gradio app that runs your best model on an uploaded photo.

**Parked: chemistry of waste.** Identifying plastics from Raman or infrared spectra needs a spectrometer and a Taiwan reference library, and we have neither. Satellites can't do it at Sentinel-2's resolution and bands, and there are no published joint Raman + image models to borrow (the [raman-open-ml](https://github.com/JoulesSpace/raman-open-ml) repo, for example, is spectra-only). If you want chemistry without a lab, see Path D: a satellite spectrometer measuring gases in air. The spectra code (`configs/toy-spectra.json`, `prepare-spectra`) stays in case that changes. If you find an open Taiwan spectral dataset, that's worth raising.

---

## Ground rules (every path)

- **Split by group,** meaning flight, scene, site or date, never by frame or pixel. If neighbouring samples can end up in both train and test, your score is lying.
- **Don't touch the test set** until you've chosen your settings. Tune on validation.
- **Record each run:** config, seed, device, time taken and the exact numbers, each in its own `runs/` folder. If you can't reproduce it, it didn't happen.
- **Claim only what you measured.** A bottle box is a PET guess, not chemistry. A flagged satellite pixel is "something floating," not "plastic."
- **Respect data terms:** credit MARIDA (CC BY 4.0) and MOENV (Open Government Data License), cite HAIDA's thesis, and keep HAIDA images out of the repo.
- **Use AI coding tools freely,** but check what they write against the actual data and code, the same verify-then-write habit as in the capstone.

## New terms

| Term | What it means here |
|---|---|
| **Sentinel-2** | Free European Earth-observation satellites; each pixel covers 10–60 m of ground |
| **Band** | One color channel of a satellite image; Sentinel-2 has 13, from blue to shortwave infrared (SWIR) |
| **Reflectance** | The fraction of sunlight a surface reflects in a band (0–1); water is near 0 in the infrared |
| **FDI** | Floating Debris Index, a formula that makes floating material stand out in near-infrared |
| **NDWI / NDVI** | Simple band ratios that highlight water / vegetation |
| **Sun glint** | Sunlight bouncing off the sea toward the satellite, which brightens water in every band |
| **MARIDA** | Open benchmark of Sentinel-2 patches, labelled into 15 sea-surface classes (none from Taiwan) |
| **HAIDA** | Open drone dataset from National Taiwan Ocean University, with boxes around coastal trash |
| **Transfer learning** | Starting from a model trained on other data and adapting it to yours |
| **Domain shift** | Deployment data looks different from training data, so the model gets worse |
| **YOLO** | A family of fast object detectors that draw labelled boxes |
| **mAP@0.5** | The standard detector score; a box counts as correct if it overlaps the true box by at least 50% |
| **Group split** | Keeping everything from one flight/scene/site/station in a single split |
| **Sentinel-5P / TROPOMI** | A European satellite (and its spectrometer) that measures trace gases in the air |
| **Column** | The total amount of a gas in the air above a spot, from the ground to space. What Sentinel-5P measures |
| **NO₂ / SO₂** | Nitrogen dioxide (mainly from traffic and power plants) and sulfur dioxide (mainly from burning coal and oil) |

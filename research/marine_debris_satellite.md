# Deep-learning detection of floating marine debris from Sentinel-2 (models, datasets, weights, compute, free imagery) — as of 6 Nov 2026

Scope: marine / floating debris only. Land/landfill waste and Taiwan-specific data are excluded (one incidental Taiwan item noted at the end). Every numeric claim carries a source URL; items marked **[unverified]** come from search snippets or secondary citations I could not open.

Local artifacts used while researching (PDF text extracted with pypdf; not needed by the reader): `/private/tmp/claude-501/-Users-jonathansherman/58533252-3293-4335-b000-10d2fc983b1/scratchpad/pdftext/` (ResAttUNet, Mifdal 2021, Moorton survey, Martínez-Vicente 2019).

---

## KQ1. Verification of each paper the student named (dataset, bands, split, metrics, model size, hardware, code/weights)

### Takeaway
All five items exist and are described accurately in outline, but two of the student's framings need correction: Biermann's "~86 %" is not a held-out multi-class accuracy (it is the share of *suspected-plastic* pixels classified as plastic, from ~200 hand-picked training pixels), and ResAttUNet's headline "F1 0.95" is a *micro* F1 (≈ pixel accuracy) — its macro-F1 (0.77) and IoU (0.67) do **not** beat MARIDA's own Random-Forest baseline (0.79 / 0.69). arXiv 2307.02465 was published in *iScience* 26(12), Dec 2023, and its weights are on Torch Hub, but the 116 GB training data has been taken private.

### Cited Findings

**(a) Biermann, Clewley, Martinez-Vicente, Topouzelis — "Finding Plastic Patches in Coastal Waters using Optical Satellite Data", Scientific Reports 10:5364, published 23 Apr 2020 (correction 12 May 2020)**
- Method: Floating Debris Index FDI = R_NIR − R'_NIR, where R'_NIR is a baseline interpolated between Red-Edge-2 (B6, 740 nm, 20 m) and SWIR-1 (B11, 1610 nm, 20 m) scaled by a wavelength ratio ×10; plus NDVI. Classifier = scikit-learn GaussianNB on five features (FDI, NDVI, reflectance at 740, 833, 1610 nm) — [Europe PMC full text](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7181820/fullTextXML)
- Training pixels: plastics Durban 53, seaweed Barbados 48, timber British Columbia 60, spume Scotland 17, seawater 20; pumice (Tonga) appears in tables without a stated count. Mytilene PLP targets held out (9 usable pixels, all classified plastic). Input: S2 L1C → ACOLITE v20181210 DSF surface reflectance; 10 m — [same](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7181820/fullTextXML)
- "86 %" = fraction of *suspected* plastic pixels across Accra (23 px, 87 %), Da Nang (22 px, 77 %), Gulf Islands BC (26 px, 100 %), Scotland (23 px, 83 %) that the NB model labelled plastic; up to 3 % went to seawater and 11 % to foam; no precision/recall/per-class accuracy is reported in the text; no data/code availability statement — [same](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7181820/fullTextXML)
- Sub-pixel: PLP targets (5×5, 5×10, 10×10, 5×20 m) "were detected on subpixel scales of 30–55 %, as confirmed by drones"; no explicit minimum size stated — [same](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7181820/fullTextXML)
- Mifdal et al. (below) reconstructed Biermann's 195 labelled pixels and showed that none of the idealised class distributions (seafoam, pumice, plastic, seawater) align with 157,319 "in-the-wild" floating-object pixels — i.e., Biermann's NB does not transfer — [Mifdal 2021 PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf)

**(b) Mifdal, Longépé, Rußwurm — "Towards Detecting Floating Objects on a Global Scale with Learned Spatial Features Using Sentinel 2", ISPRS Annals V-3-2021, pp. 285–293, 17 Jun 2021, CC BY 4.0**
- Dataset ("FloatingObjects"): hand-drawn **line** labels in Google Earth Engine over coastal scenes; closed rings are filled as polygons. 15 scene/dates listed in the paper's table (Bay of Biscay 824 lines, Da Nang 51, Kolkata 15, Longxuyen 33, Panama 182, Rio 114, San Diego 15, Toledo 35, Accra 99, Lagos 30 + 188, Mandaluyong 13, Venice 97 + 67, Vungtau 79); 157,319 floating-object pixels; both L1C (13 bands) and L2A (12 bands) stored when available — [PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf)
- Model/training: U-Net on 12 S2 bands; 128×128 crops centred on line segments (random crop from 256); batch 80; 50 epochs; LR 0.001; weighted BCE; augmentation = flips, 90° rotations, spatial+spectral noise, random TOA/BOA mixing; hard-negative random patches; test = 480×480 tiles with 64-px overlap + TTA; six-fold **region-wise** cross-validation; hardware not stated — [PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf)
- Results (balanced pixel sample from test regions): U-Net acc 84.28 / F1 0.81 / κ 0.68 vs. Naïve Bayes on NDVI+FDI 60.81 / 0.53 / 0.21, RF 58.83 / 0.69 / 0.17, SVM 58.82 / 0.67 / 0.17 — [PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf)
- Failure cases shown: coastline waves (Accra), ships (Accra), cloud fringes (Da Nang); on PLP-2018 10×10 m targets only 2 of 6 fold-models produced any score — [PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf)
- Repo (archived read-only 6 Feb 2026): 12 scenes / 11 regions table in README; Apache-2.0; Torch Hub models `unet_seed0/1`, `manet_seed0/1` (2-fold); snapshot name `unet-posweight10-lr001-aug1.pth.tar`; data via Google-Drive `code/download.py`; companion paper Carmo, Mifdal, Rußwurm, OCEANS 2021 — [GitHub](https://github.com/ESA-PhiLab/floatingobjects)
- Environmental Data Science Book notebook "Detecting floating objects using deep learning and Sentinel-2 imagery" (Mifdal & Carmo), v2025.6.0, 20 Jun 2025, MIT — [Zenodo 15706997](https://zenodo.org/records/15706997); [GitHub](https://github.com/eds-book/b34facfa-cea8-48f5-89f6-f11ce00812a9)

**(c) Kikaki, Kakogeorgiou, Mikeli, Raitsos, Karantzalos — "MARIDA", PLOS ONE 17(1): e0262247, published 7 Jan 2022**
- Data: 63 S2 scenes 2015–2021, 11 countries; 1,381 patches 256×256; 837,357 annotated px; 11 Rayleigh-corrected bands (B9, B10 dropped) via ACOLITE at 10 m (20/60 m bands pixel-replicated); FDI/FAI used to aid annotation; per-pixel confidence (high/moderate/low) — [PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247)
- Split: scene-disjoint, ~50/25/25: train 694 patches (429,412 px), val 328 (213,102 px), test 359 (194,843 px) — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8740969/fullTextXML)
- Baselines evaluate 11 classes (Wakes, Cloud Shadows, Waves, Mixed Water merged into Marine Water) — [PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247)
- Table 4 (test set; IoU / PA / F1): **Marine Debris** — RF_SS 0.55/0.91/0.71; RF_SS+SI **0.67/0.92/0.80**; RF_SS+SI+GLCM 0.65/0.92/0.79; **U-Net 0.33/0.70/0.50**. **Class-average** — RF_SS 0.57/0.72/0.69; RF_SS+SI 0.59/0.73/0.70; **RF_SS+SI+GLCM 0.69/0.79/0.79**; U-Net 0.57/0.69/0.69 — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8740969/fullTextXML)
- U-Net: from scratch, 11 bands, 4 down/4 up blocks, 16 base channels, Adam, class-weighted CE, LR 2e-4 → 2e-5 after epoch 40, 44 epochs, batch 5, rot/flip aug; RF: 125 trees, depth 20, inverse-freq class weights, confidence weights 1 / 2/3 / 1/3; hardware: only "NVIDIA GPU" acknowledgement, no time — [PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247)
- Code MIT; U-Net and ResNet weights at `https://pithos.okeanos.grnet.gr/public/lxh8hL4zvuSKds2BdVnMd2` (format unstated; link liveness not tested); also Radiant MLHub `marida_v1` (MLHub shut down in 2024 — **[unverified]**) — [GitHub README](https://github.com/marine-debris/marine-debris.github.io)

**(d) arXiv 2307.02465 — Rußwurm, Jilla Venkatesa, Tuia, "Large-scale Detection of Marine Debris in Coastal Areas with Sentinel-2"; arXiv v1 5 Jul 2023; published *iScience* 26(12) 2023**
- Venue: arXiv page still says "in review" — [arXiv abs](https://arxiv.org/abs/2307.02465); repo README cites iScience 26(12) 2023 — [GitHub](https://github.com/marccoru/marinedebrisdetector)
- Training data combined: FloatingObjects (26 scenes, 3,297 line-annotated objects; 22 regions for training), MARIDA (63 scenes, 6,672 polygons: 1,882 debris / 2,447 water / 2,343 other; debris = positive, water = negative, other classes unused; re-downloaded as 12-band Sen2Cor L2A), S2Ships (ship patches as negatives), RefinedFloatingObjects (point labels; val: Lagos, New Orleans, Venice, Marmara; eval: Accra 2018-10-31, Durban 2019-04-24), PLP 2021/2022 (qualitative only) — [arXiv HTML](https://arxiv.org/html/2307.02465v1)
- Label refinement: FDI-Otsu thresholding + random-walker segmentation → 25 alternative masks per sample, one drawn at random per step; negatives from random background, S2Ships, and GEE smileCART false positives — [arXiv HTML](https://arxiv.org/html/2307.02465v1)
- Models: U-Net ≈31 M params; **UNet++ (ResNet encoder) ≈26 M params = released "Marine Debris Detector"**; 12 bands, 128×128 patches, LR 0.01, wd 1e-6, 100 epochs; selection by val AUROC; **1× RTX 3090, ≈8 h (UNet++) / 9 h (U-Net), ≈2.8 kg CO₂e per run**; calibrated thresholds (UNet++ 0.132/0.064/0.025) — [arXiv HTML](https://arxiv.org/html/2307.02465v1)
- Results (F1; RF-orig = MARIDA RF, U-Net-orig = Mifdal U-Net, then models trained on combined data): **MARIDA test** 0.288 / 0.701 / RF 0.708 / U-Net 0.741 / **UNet++ 0.749**; **Accra** 0.464 / 0.871 / 0.545 / 0.920 / 0.926 (UNet++ w/o refinement 0.948); **Durban** 0.105 / 0.497 / 0.708 / 0.756 / **0.837** — [arXiv HTML](https://arxiv.org/html/2307.02465v1)
- Weights: `torch.hub.load("marccoru/marinedebrisdetector","unetpp")` (also `"unet"`, `label_refinement=False`); `pip install marinedebrisdetector`; Google-Drive weights folder; **dataset (116 GB) made private — "costs of hosting got too high", email author** — [GitHub](https://github.com/marccoru/marinedebrisdetector); data sizes MARIDA 21/24 GB, FloatingObjects 18/37 GB, Refined 1.3/12 GB, PLP 38/41 MB, S2Ships 685 MB/2.6 GB — [doc/data.md](https://raw.githubusercontent.com/marccoru/marinedebrisdetector/main/doc/data.md)
- GEE comparison app — [marinedebrisexplorer](https://marcrusswurm.users.earthengine.app/view/marinedebrisexplorer)

**(e) ResAttUNet — Azhan Mohammed (GalaxEye Space), arXiv 2210.08506, 16 Oct 2022, preprint only**
- MARIDA, weighted CE (log class-frequency weights), 200 epochs, LR 1e-3 halved every 40 epochs; CBAM attention in every U-Net block + 3 residual blocks; Dice and focal variants were worse; hardware, patch size, bands, parameter count, split not stated — [PDF](https://arxiv.org/pdf/2210.08506)
- Table 2: UNet-MARIDA (re-run) macro-P 0.74 / macro-F1 0.69 / micro-F1 0.89 / IoU 0.57; **ResAttUNet macro-F1 0.77 / micro-F1 0.95 / weighted-F1 0.95 / IoU 0.67**; no per-class (Marine Debris) scores; conclusion text says IoU 0.65 vs table 0.67 — [PDF](https://arxiv.org/pdf/2210.08506); code [SADMA](https://github.com/sheikhazhanmohammed/SADMA)

### Inferences
- ResAttUNet's re-run "UNet-MARIDA" (IoU 0.57, macro-F1 0.69) equals MARIDA Table 4's U-Net class-average row, so its IoU column is the class-mean IoU; by that yardstick ResAttUNet (0.67 / 0.77) beats the U-Net baseline but is at or below MARIDA's RF_SS+SI+GLCM (0.69 / 0.79). The "0.95" the student may quote is micro-F1 dominated by water/cloud pixels.
- The iScience detector is the only one of the student's items that is a true "global" binary debris detector with calibrated thresholds, public Torch-Hub weights and a pip CLI; it is the natural baseline for a student project even though the training data are now private.

### Gaps
- MARIDA ResNet multi-label numbers are in S6 Table (not extracted); hardware/time for MARIDA and ResAttUNet are not published.
- I could not verify that the MARIDA grnet.gr weight links or the FloatingObjects Google-Drive link still resolve (the FloatingObjects repo is archived).

---

## KQ2. Which models perform best, and which results are legitimately comparable (same dataset / split / metric)?

### Takeaway
There is no single leaderboard: MARIDA (11-class, scene-disjoint split, per-class IoU/F1), MADOS (15-class, F1/mIoU/OA, 5-trial mean), the iScience binary protocol (debris-vs-water with calibrated thresholds, MARIDA-test + Accra + Durban), the GFM-Bench/PANGAEA foundation-model protocols (re-chipped 96×96 tiles, frozen encoders) and the 2026 Frontiers binary MARIDA+MADOS merge all use different label sets, chips and metrics. Within protocols: on MADOS the best published numbers are MambaMPD (F1 74.8 / mIoU 69.8, Sept 2026) and MariNeXt (76.0 / 64.3 as reported; 70.6 / 59.2 when reproduced); on MARIDA multi-class the RF baseline (MD F1 0.80, mean 0.79) is still the number to beat; on the iScience binary protocol UNet++ trained on merged data wins (MARIDA-test F1 0.749, Durban 0.837).

### Cited Findings
- **MARIDA protocol** (11 classes, 359 test patches): MD F1 RF_SS+SI 0.80 vs U-Net 0.50; mean F1 RF 0.79 vs U-Net 0.69 — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8740969/fullTextXML). ResAttUNet macro-F1 0.77 / IoU 0.67 on MARIDA (same data, split not stated) — [arXiv PDF](https://arxiv.org/pdf/2210.08506).
- **iScience binary protocol** cross-evaluates both prior models: MARIDA RF (orig.) and Mifdal U-Net (orig.) on MARIDA test F1 0.288 / 0.701; on Accra 0.464 / 0.871; on Durban 0.105 / 0.497; merged-data UNet++ 0.749 / 0.926 / 0.837. **Joint-training effect** (U-Net, orig. → combined data): Accra 0.871→0.920, Durban 0.497→0.756, MARIDA 0.701→0.741. Label refinement helped Durban (0.776→0.837) and MARIDA (0.710→0.749) but hurt Accra (0.948→0.926). Authors attribute gains "mainly to dataset design (extensive negative sampling, label refinement), not architecture" — [arXiv HTML](https://arxiv.org/html/2307.02465v1); [arXiv abs](https://arxiv.org/abs/2307.02465)
- Caveat inside iScience's own table: on MARIDA-test AUROC the merged-data RF (0.862) and Mifdal U-Net (0.764) exceed UNet++ (0.746) — [arXiv HTML](https://arxiv.org/html/2307.02465v1)
- **MADOS protocol** (official scene-level split, 5 trials): MariNeXt F1 76.0 / mIoU 64.3 / OA 89.1; U-Net 63.8 / 51.0; SegNeXt 60.6 / 49.2 / 86.6; MariNeXt "outperforms all baselines by at least 12 % in F1 and mIoU" — [ScienceDirect snippet via search](https://www.sciencedirect.com/science/article/pii/S0924271624000625) **[numbers from search snippet; page returns 403]**; project page confirms the ≥12 % claim — [marine-pollution.github.io](https://marine-pollution.github.io/)
- **MambaMPD** (arXiv 2609.15676, 14 Sep 2026; Chen, Han, Ren, Luo, Fu) Table 2 on MADOS: RF 56.6/43.9/67.1; RF++ 64.4/52.4/83.8; U-Net 63.8/51.0/82.9; SegNeXt 60.6/49.2/86.6; **MariNeXt (reproduced) 70.6/59.2/81.6**; OSDMamba 71.2/68.1/82.3; **MambaMPD 74.8/69.8/83.1** (F1/mIoU/OA); 37.06 M params, 204 GFLOPs at 1250×650, 15.2 FPS on RTX 4070 Ti; code CC0 — [arXiv HTML](https://arxiv.org/html/2609.15676)
- **PANGAEA** (arXiv 2412.04204, v2 30 Apr 2025) MADOS mIoU with frozen encoders + UPerNet (80 epochs, Adam 1e-4, batch 8): CROMA 67.55, GFM-Swin 64.71, RemoteCLIP 60.00, DOFA 59.58, SpectralGPT 57.99, Scale-MAE 57.32, SatlasNet 55.86, **U-Net from scratch 54.79**, SSL4EO-S12 MoCo 51.76, Prithvi 49.98, ViT-B/16 scratch 48.19; "GFMs trained with a wider spectral range perform best" on MADOS; removing per-band normalisation collapses Scale-MAE 57.3→16.8 — [arXiv HTML](https://arxiv.org/html/2412.04204). TerraMind-B 69.5 mIoU on MADOS per MambaMPD's Table 3 — [arXiv HTML](https://arxiv.org/html/2609.15676)
- **GFM-Bench** (Si et al., arXiv 2503.12843, Mar 2025) MARIDA re-chipped to 96×96 (train 5,622 / val 624 / test 6,183; val+test merged), full fine-tune with UPerNet, 10 epochs: SatMAE++ ViT-L 59.82 mIoU, LESS ViT-B 55.64, SatMAE ViT-B 54.33, SpectralGPT 44.72, CROMA 43.04, Scale-MAE 41.12 — [arXiv HTML](https://arxiv.org/html/2503.12843); [HF card](https://huggingface.co/datasets/GFM-Bench/MARIDA)
- **Frontiers in Marine Science 2026** (Bouchelaghem et al., 31 Mar 2026): binary U-Net (≈31 M) on merged MARIDA+MADOS (4,702 patches, 8,095 debris px; 16×16 chips with stride 8 upscaled to 128×128; own 70/15/15 split): F1 0.901 / IoU 0.820 single run; 0.881 ± 0.016 over five seeds; cross-dataset MADOS→MARIDA F1 0.890, MARIDA→MADOS 0.833; authors explicitly warn their comparison table "mixes protocols (multi-class vs binary, micro vs macro, within- vs cross-dataset)" and their own text is internally inconsistent (MariNeXt F1 0.76 in tables vs 0.891 in abstract; RF F1 0.70 vs 0.79) — [Frontiers](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1765021/full)
- **MAP-Mapper** (Booth, Ma, Karakuş, Sci. Rep. 13:6822, Apr 2023): MARIDA-trained; "-HP" variant 95 % precision on MD vs ~30 % for the MARIDA U-Net; "-Opt" 87–88 % precision–recall; introduces MDM density index; applied to Manila and Mumbai 2022 — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10133222/) **[abstract-level only]**
- Secondary-only claims (could not open): AquaSense on MADOS F1 0.91 / mIoU 0.84 (Kaviya & Bhavani 2025) **[unverified; ResearchGate listing only]** — [RG](https://www.researchgate.net/publication/392455527_Advanced_Deep_Learning_Framework_AquaSense_for_Comprehensive_Marine_Pollutant_Detection_Using_Sentinel-2_Multispectral_Data); ResUNext F1 0.84 on MARIDA (Rao 2025) **[unverified; cited by Frontiers]**; Research Square preprint claiming SegFormer-B2 mIoU 0.726 and MD F1 0.81 vs 0.51 for MariNeXt on MADOS **[unverified; page is JS-only]** — [Research Square](https://www.researchsquare.com/article/rs-11300519/v1)
- Survey: Moorton, Kurt, Woo, "State of the art applications of deep learning within tracking and detecting marine debris: A survey", arXiv 2403.18067, 26 Mar 2024 — lists no FloatingObjects/MADOS and reports no hardware for the S2 studies — [arXiv HTML](https://arxiv.org/html/2403.18067)

### Inferences
- Legitimately comparable groups: (i) MARIDA Table 4 vs ResAttUNet class-mean numbers (same data; ResAttUNet's split is presumed official); (ii) MADOS Table (MariNeXt/U-Net/SegNeXt) vs MambaMPD Table 2 (same split, but MariNeXt reproduces 5 F1 points lower than reported — flag when quoting); (iii) PANGAEA's frozen-encoder MADOS table (all under one protocol; not comparable to end-to-end models); (iv) iScience's own table (RF-orig, Mifdal U-Net, merged-data models on identical binary test sets). Everything else — Biermann's 86 %, ResAttUNet's 0.95 micro-F1, Frontiers' 0.90, AquaSense's 0.91, GFM-Bench mIoUs — is not comparable across papers.
- The student's "joint training" item (2307.02465) is correct: it did train one model on MARIDA + FloatingObjects (+ S2Ships + refined labels) and reports both MARIDA-trained RF and FloatingObjects-trained U-Net on MARIDA's test set; the reverse direction (MARIDA model on FloatingObjects test) is **not** reported.

### Gaps
- MADOS per-class Marine-Debris IoU/F1 for MariNeXt is in the paper's Table 4 image (ScienceDirect blocked); only the secondary 0.51 F1 figure exists.
- No paper reports Marine-Debris-class metrics for ResAttUNet.

---

## KQ3. How and on what to train — compute actually used / required

### Takeaway
Published training budgets range from trivial (MARIDA U-Net: 694 patches, batch 5, 44 epochs, 16-channel U-Net — minutes on a Colab T4) through moderate (Frontiers binary U-Net ≈31 M params, 3,291 128×128 patches, 100 epochs, **195 min on an RTX 4080 16 GB**) to heavy (iScience UNet++ ≈26 M, 116 GB merged data, **≈8 h on one RTX 3090**; MADOS MariNeXt on mmsegmentation). For one student on Colab, MARIDA or MADOS patch-level training of a small U-Net / UNet++ (or a frozen foundation encoder + UPerNet head, PANGAEA-style) is realistic; retraining the iScience detector is not (data are private and 116 GB).

### Cited Findings
- MARIDA U-Net: 11 bands, 256×256, 4+4 blocks, 16 base channels, Adam, weighted CE, LR 2e-4→2e-5 at epoch 40, 44 epochs, batch 5, rot/flip — [PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247); repo env: Python 3.7, PyTorch 1.7, CUDA 11.0; RF GLCM feature stacking alone takes ~110 min CPU — [GitHub README](https://github.com/marine-debris/marine-debris.github.io)
- Mifdal U-Net: 12 bands, 128×128, batch 80, 50 epochs, LR 1e-3, weighted BCE, region-wise 6-fold CV — [PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf)
- iScience: 12 bands, 128×128, LR 0.01, wd 1e-6, 100 epochs, batch 256 in the example command with 32 workers, PyTorch-Lightning + W&B; 1× RTX 3090; ≈8 h UNet++ / 9 h U-Net; ≈2.8 kg CO₂e — [arXiv HTML](https://arxiv.org/html/2307.02465v1); [doc/model_training.md](https://raw.githubusercontent.com/marccoru/marinedebrisdetector/main/doc/model_training.md)
- ResAttUNet: 200 epochs, LR 1e-3 halved every 40 epochs, weighted CE; no hardware — [PDF](https://arxiv.org/pdf/2210.08506)
- MADOS/MariNeXt: built on SegNeXt + mmsegmentation v0.24.1 (PyTorch 1.11, cu113, mmcv); VSCP sample-mixing augmentation; test-time augmentation with majority vote; 5 trained runs released; training hyper-parameters not in README — [GitHub](https://github.com/gkakogeorgiou/mados); [project page](https://marine-pollution.github.io/)
- Frontiers 2026: U-Net ≈31 M params, 11 bands, 128×128, AdamW LR 8e-4, wd 1.5e-4, batch 16, 100 epochs, 20-epoch warm-up + cosine, weighted BCE 0.5 + Dice 0.3 + focal 0.2, early-stop patience 15; **RTX 4080 16 GB, 195 min**; input-scale ablation 16→128 px raised val F1 0.778→0.895 — [Frontiers](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1765021/full)
- PANGAEA decoder-only recipe: frozen encoder, UPerNet on 4 intermediate layers, 80 epochs, Adam 1e-4, batch 8, wd 0.05, LR ×0.1 at 60 %/90 %; GPU not stated — [arXiv HTML](https://arxiv.org/html/2412.04204)
- GFM-Bench full fine-tune on MARIDA: 10 epochs, AdamW, effective batch 256, bf16, cosine; pre-training on 8× A100-40 GB (or 8× A6000, inconsistently stated) — [arXiv HTML](https://arxiv.org/html/2503.12843)
- MambaMPD: SGD 0.9, LR 1e-3 poly, ≤100 epochs, batch 4 or 8 (inconsistent), focal + Jaccard with deep supervision, ImageNet init, RTX 4070 Ti — [arXiv HTML](https://arxiv.org/html/2609.15676)
- Class imbalance is the central training problem: MARIDA MD = 3,399 px of 837,357 (0.41 %); MADOS MD = 4,696 px of ~1.48 M (0.32 %); merged ratio 1:288 on annotated px, 1:33,875 including background — [Frontiers](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1765021/full); [MADOS stats](https://marine-pollution.github.io/Report_Patches_stats.html)

### Inferences
- A Colab-scale project should (a) train on MARIDA/MADOS 11-band patches with class-weighted CE or BCE+Dice+focal, 128–256 px crops, ≤100 epochs; (b) reuse the iScience UNet++ (Torch Hub, 12-band Sen2Cor L2A input) for zero-shot or fine-tuning; or (c) attach a UPerNet head to a frozen TorchGeo/DOFA/CROMA encoder (PANGAEA recipe). Each of these fits a T4/L4 in well under the 195 min reported on an RTX 4080.
- Band/pre-processing mismatch matters: MARIDA/MADOS are ACOLITE Rayleigh-corrected 11-band; the iScience model expects 12-band Sen2Cor L2A (B10 removed); FloatingObjects mixes L1C/L2A. Pick one convention per model.

### Gaps
- No paper reports epoch-level timing for MARIDA/MADOS baselines; the only wall-clock figures are iScience (8–9 h, RTX 3090) and Frontiers (195 min, RTX 4080).

---

## KQ4. Inference speed per scene/patch and compute required

### Takeaway
Only three sources publish timings: the iScience detector processes a full Durban S2 scene (3122×3843 px, ~300 MB GeoTIFF) in **11 s on an RTX 3090, 15 s on a GTX 970, 11 min on a MacBook CPU**; the Frontiers U-Net runs **22 128-px tiles/s on an RTX 4080**; MambaMPD runs **15 FPS at 1250×650 on an RTX 4070 Ti (37 M params, 204 GFLOPs)**. A full 110×110 km S2 tile (~10,980² px) is ~8× the Durban scene, so order-of-minutes on a Colab GPU and ~1–2 h on CPU.

### Cited Findings
- marinedebrisdetector CLI timings (Durban scene): RTX 3090 11 s; GTX 970 15 s; MacBook Pro CPU 11 min; `--download-durban` ≈300 MB; default model UNet++ — [GitHub](https://github.com/marccoru/marinedebrisdetector); scene size 3122×3843 px; inference via overlapping 480-px patches, local maxima ≥3 px apart — [arXiv HTML](https://arxiv.org/html/2307.02465v1)
- Frontiers 2026: 22 tiles/s (128×128) on RTX 4080; U-Net ≈31 M params — [Frontiers](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1765021/full)
- MambaMPD: 37.06 M params, 204.29 GFLOPs at 1250×650, 15.19 FPS (100-pass mean, RTX 4070 Ti); baseline MambaU-Net 36.59 M / 164 GFLOPs / 17.2 FPS; SAM-OIL >600 M params — [arXiv HTML](https://arxiv.org/html/2609.15676)
- Mifdal U-Net test tiling: 480×480 tiles, 64-px overlap, plus flip/rotate TTA — [PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf)
- Model sizes: iScience U-Net ≈31 M, UNet++ ≈26 M — [arXiv HTML](https://arxiv.org/html/2307.02465v1); MARIDA U-Net 16 base channels (parameter count not stated) — [PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247)
- S2 tile ≈110×110 km (100 km grid step + overlap) — [CDSE S2 page](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html)

### Inferences
- A 110 km tile at 10 m is ~1.2×10⁸ px ≈ 8× Durban's 1.2×10⁷ px → roughly 1.5–2 min per tile on an RTX 3090-class GPU with UNet++, ~1.5 h on a laptop CPU. Colab T4 is slower than a 3090 (expect a few minutes per tile) — my estimate, not a published figure.
- MariNeXt uses TTA with majority voting over 5 runs, so its per-scene cost is several times a single U-Net pass — no figure published.

### Gaps
- No inference timings or FLOPs for MARIDA U-Net, ResAttUNet, MariNeXt, or MAP-Mapper.

---

## KQ5. Openly downloadable datasets (URLs, licenses, sizes, classes, splits) and datasets beyond the student's list

### Takeaway
Three Sentinel-2 benchmarks are fully open and small enough for Colab: **MARIDA** (Zenodo, CC BY 4.0, 1.2 GB zip / 4.38 GB, 1,381 256×256 patches, 15 classes, scene-disjoint split), **MADOS** (Zenodo, CC BY 4.0, 4.0 GB zip / 5.35 GB, 174 scenes, 2,803 240×240 crops, 15 classes incl. Oil Spill, official split), and **FloatingObjects** (GitHub/Google Drive, Apache-2.0, ~18 GB zipped / 37 GB, line labels, 12+ scenes). Beyond these: PLP 2019/2021 target campaigns (Zenodo, CC BY 4.0, 69 MB and 159.5 GB), the Cózar 2024 Mediterranean litter-windrow catalogue (Zenodo netCDF of 14,374 detections), Hugging-Face re-chips of MARIDA (marida-taco, GFM-Bench/MARIDA), NASA Marine Debris (PlanetScope 3 m, in TorchGeo), Ocean Scan (ESA in-situ database), and several drone/USV sets (FloW, AFO, AquaPlasticWaste) that are freshwater or SAR-oriented rather than open-sea litter.

### Cited Findings

**MARIDA**
- Zenodo 10.5281/zenodo.5151941, v1.0.0, published 1 Aug 2021 (modified 23 Jan 2022); MARIDA.zip 1.2 GB (md5 9bf32266…), 4.38 GB uncompressed; **CC BY 4.0**; 1,381 GeoTIFF patches 256×256 by date/tile; train/val/test lists included; software DOI 10.5281/zenodo.5152216 — [Zenodo](https://zenodo.org/records/5151941)
- 15 classes (mask values 1–15): Marine Debris, Dense Sargassum, Sparse Sargassum, Natural Organic Material, Ship, Clouds, Marine Water, Sediment-Laden Water, Foam, Turbid Water, Shallow Water, Waves, Cloud Shadows, Wakes, Mixed Water — [GitHub README](https://github.com/marine-debris/marine-debris.github.io)
- Pixels per class: MD 3,399 (1,625 high / 1,235 moderate / 539 low confidence); DenS 2,797; SpS 2,357; NatM 864; Ship 5,803; Cloud 117,400; MWater 129,159; SLWater 372,937; Foam 1,225; TWater 157,612; SWater 17,369; Waves 5,827; CloudS 11,728; Wakes 8,490; MixWater 410. Patches per class: MD 373 (multi-label counts sum > 1,381) — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8740969/fullTextXML)
- Split: train 694 / val 328 / test 359 patches; all patches of one scene-date in one split — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8740969/fullTextXML)
- File layout `data/patches/S2_DATE_TILE/*_CROP.tif`, `*_CROP_cl.tif`, `*_CROP_conf.tif`; `data/splits/{train,val,test}_X.txt`; `labels_mapping.txt`; extras `dataset.h5`, GLCM/indices stacks via grnet.gr links — [GitHub README](https://github.com/marine-debris/marine-debris.github.io)
- Project-site reports: overview map, categories/confidence EDA, spectral signatures, embeddings, patch statistics — [marine-debris.github.io](https://marine-debris.github.io/)

**MARIDA derivatives / mirrors**
- `isp-uv-es/marida-taco` (Univ. València): 15,381 tiles 96×96, 2.3 GB, 11 L2A bands on one 10 m grid, CC BY 4.0; splits train 6,246 / val 2,952 / test 6,183; **warning: not scene-disjoint — 12 of 63 scenes span splits (5,904 tiles, 38.4 %)**; `scene_id` column allows re-splitting; rasters not georeferenced — [HF](https://huggingface.co/datasets/isp-uv-es/marida-taco)
- `GFM-Bench/MARIDA` (and older `Ehzoahis/MARIDA`): 256→nine 96×96 chips; train 5,622 / val 624 / test 6,183 (original val+test merged = 678 patches); 11 bands; loading-script based — [HF](https://huggingface.co/datasets/GFM-Bench/MARIDA); [HF](https://huggingface.co/datasets/Ehzoahis/MARIDA)
- iScience re-download of MARIDA scenes as 12-band Sen2Cor L2A (21/24 GB) — now private — [doc/data.md](https://raw.githubusercontent.com/marccoru/marinedebrisdetector/main/doc/data.md)
- "MARIDA-SAR"-style derivative: Manesis, Mikeli, Kikaki, Kakogeorgiou, Karmas, Karantzalos, "Detecting Marine Pollutants Using Sentinel-1 SAR and Sentinel-2 Optical Imagery", ISPRS Annals XI-3-2026 pp. 187–192, 8 Jul 2026, CC BY 4.0: Sentinel-1 patches built from **MADOS** annotations (±1 day), classes oil spill / sea surface / look-alikes / ship / platform; MariNeXt F1-macro 92.7 vs SegNeXt 75.9 vs U-Net 70.6; "mapping marine debris in SAR is especially hard"; no data link — [ISPRS Annals](https://isprs-annals.copernicus.org/articles/XI-3-2026/187/2026/)

**FloatingObjects (ESA Φ-lab)**
- Apache-2.0; 12/13-band S2 scenes for Accra, Da Nang, Kolkata, Lagos, Long Xuyen, Mandaluyong, Panama, Rio de Janeiro, San Diego, Venice (×2), Vung Tau (+ Limassol example); Google-Drive download via `code/download.py`; COG-Explorer overviews; repo archived 6 Feb 2026 — [GitHub](https://github.com/ESA-PhiLab/floatingobjects)
- Labels are hand-drawn lines (rings filled); 157,319 labelled floating-object pixels; 15 scene-dates with per-scene line counts (e.g., Bay of Biscay 824) — [PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf)
- Mirror size 18 GB zipped / 37 GB unzipped; "26 globally distributed scenes; 3,297 annotated floating objects" in iScience's accounting — [doc/data.md](https://raw.githubusercontent.com/marccoru/marinedebrisdetector/main/doc/data.md); [arXiv HTML](https://arxiv.org/html/2307.02465v1)

**MADOS (Marine Debris and Oil Spill)**
- Zenodo 10.5281/zenodo.10664073, v1, 16 Feb 2024; MADOS.zip 4.0 GB (md5 1076b25d…), 5.35 GB uncompressed; **CC BY 4.0**; 174 scene folders; 2,803 crops 240×240; per-resolution band files (10 m: 492/560/665/833 nm; 20 m: 704/783/865/1614/2202 nm; 60 m: 443 nm) + RGB + ACOLITE turbidity; `splits/{train,val,test}_X.txt` — [Zenodo](https://zenodo.org/records/10664073)
- Classes (mask 0–15): 0 non-annotated, 1 Marine Debris, 2 Dense Sargassum, 3 Sparse Floating Algae, 4 Natural Organic Material, 5 Ship, 6 Oil Spill, 7 Marine Water, 8 Sediment-Laden Water, 9 Foam, 10 Turbid Water, 11 Shallow Water, 12 Waves & Wakes, 13 Oil Platform, 14 Jellyfish, 15 Sea snot — [Zenodo](https://zenodo.org/records/10664073)
- Per-class (train/val/test patches; total px): Marine Debris 250/98/124 (472 patches; **4,696 px**); Oil Spill 204/77/80 (234,568 px); Marine Water 1,360 patches (496,702 px); Foam 1,225 px; Jellyfish 12 patches; totals 2,018/897/1,036 patches, 1,481,155 px — [stats page](https://marine-pollution.github.io/Report_Patches_stats.html)
- Paper: Kikaki, Kakogeorgiou, Hoteit, Karantzalos, ISPRS J. Photogramm. Remote Sens. 210:39–54, Apr 2024; 174 scenes from 47 tiles, 2015–2022, ~1.5 M px — [project page](https://marine-pollution.github.io/); code MIT; MariNeXt weights (5 runs) on Google Drive — [GitHub](https://github.com/gkakogeorgiou/mados)
- Auto-downloadable through PANGAEA-bench (GPL-3.0) — [GitHub](https://github.com/VMarsocci/pangaea-bench); no TorchGeo or HF copy found — [search](https://docs.torchgeo.org/en/stable/api/datasets.html)

**Plastic Litter Project (PLP) targets — Univ. of the Aegean**
- PLP2019 dataset: Zenodo 10.5281/zenodo.3752719, 15 Apr 2020, CC BY 4.0, 68.9 MB rar: S2 L1C catalogue, ACOLITE-corrected netCDF subsets (sun-glint removed, land-masked), UAV photos, ESRI point vectors with **% plastic per S2 pixel** — [Zenodo](https://zenodo.org/records/3752719)
- PLP2021 dataset: Zenodo 10.5281/zenodo.7085112, 23 Sep 2022, CC BY 4.0, **159.5 GB** in 22 dated zips (Jun–Oct 2021, Gulf of Gera, Lesvos): S2 L1C + ACOLITE netCDF, UAS RGB/hyperspectral, orthophotos, ancillary log; ESA Discovery contract 4000131040 — [Zenodo](https://zenodo.org/records/7085112)
- Target scale: 2018 targets 10×10 m; 2019 5×5 m; 2021: 28 m HDPE mesh and 28 m wooden target; 2022: 5 m PVC and two 7 m HDPE meshes — [MDPI PLP-2019](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-02013/article_deploy/remotesensing-12-02013.pdf); [arXiv HTML](https://arxiv.org/html/2307.02465v1); PLP2022 1 % and 3 % abundance targets — [plp.aegean.gr](https://plp.aegean.gr/plastic-litter-project-2022/)
- Ocean Scan lists a PLP2021 entry under a different DOI (10.5281/zenodo.7380994) **[unverified which is canonical]** — [Ocean Scan](https://web-ocean-scan.vercel.app/dataset/8cef774c-afb9-46df-a8d3-4426712574a6)

**Litter windrows (Mediterranean)**
- Cózar et al., "Proof of concept for a new sensor to monitor marine litter from space", Nat. Commun. 15:4637, 14 Jun 2024: 288,166 S2 L1C images (~150 TB, 411 tiles, Jul 2015–Sep 2021) → 14,374 litter windrows (94.5 km²); detector = adapted IR spectral index + filament filter (>70 m) + manual supervision by six researchers (not ML); **data: netCDF of all detections + image clippings at Zenodo 10.5281/zenodo.11045944**; code "upon request" — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11178853/fullTextXML)

**Other satellite sets**
- NASA Marine Debris (TorchGeo): PlanetScope 3 m RGB, 707 images 256×256, object detection (1 class), Apache-2.0 — [TorchGeo](https://docs.torchgeo.org/en/stable/api/datasets/nasa-marine-debris.html)
- Sannigrahi et al. (arXiv 2106.03694, Jun 2021; later Marine Pollution Bulletin): 27 S2 scenes, in-situ targets Mytilene/Limassol, RF/SVM with kNDVI, ~99 % at Calabria/Beirut; no public dataset/code — [arXiv](https://arxiv.org/abs/2106.03694); [survey](https://arxiv.org/html/2403.18067)
- Ocean Scan (ESA Discovery; isardSAT, Lobelia, GFZ…): global database of in-situ litter observations paired with remote-sensing images; free with registration; web, API, mobile app; each campaign has a Zenodo DOI — [Ocean Scan](https://web-ocean-scan.vercel.app/)
- Spectral references: Hu 2022 "cautious note" on MSI spectral interpretation (sub-pixel <10 %) — [NOAA repository](https://repository.library.noaa.gov/view/noaa/68394); Martínez-Vicente et al. 2019 observation-requirements perspective (SNR 200:1 sufficient for ≥20 % Sargassum pixel cover) — local PDF text (Remote Sensing 2019 11:2443, **DOI unverified**)

**Drone / USV / aerial sea-surface litter (for pretraining; mostly not open sea)**
- FloW (Cheng et al., ICCV 2021): USV-view floating waste in inland waters; FloW-Img + FloW-RI (4,000 radar-image frames); GitHub ORCA-Uboat; licence not confirmed — [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Cheng_FloW_A_Dataset_and_Benchmark_for_Floating_Waste_Detection_in_ICCV_2021_paper.pdf)
- AFO – Aerial Floating Objects: 3,647 drone images, 39,991 objects, maritime SAR (people/objects), CC BY-NC-SA 3.0 — [Voxel51](https://docs.voxel51.com/dataset_zoo/datasets_hf/afo_aerial_floating_objects.html)
- AquaPlasticWaste 2026 (Mendeley): 5,000 UAV images, 30,166 polygon instances, Dal Lake (freshwater); licence not shown — [Mendeley](https://data.mendeley.com/datasets/zshjxgfmc3/1)
- 4TU "Floating Plastic Detection Dataset" (Sukno & Palunko 2024) — [4TU](https://data.4tu.nl/datasets/82306610-84ab-4aed-a759-5ec92567b7d5) **[contents unverified]**
- DEEP-PLAST (Black Sea, Water 2025): U-Net/U-Net++ on S2 + YOLOv5 on drones + drift model **[abstract only]** — [MDPI](https://doi.org/10.3390/w17223318)
- Survey's aerial sets (Jakovljevic 2020 river UAV, Maharjan 2022 river UAV, de Vries 2021 vessel GoPro) — none with stated public URLs — [survey](https://arxiv.org/html/2403.18067)

### Inferences
- MARIDA + MADOS together (~10 GB, both CC BY 4.0, same authors, same ACOLITE pipeline, overlapping class schemes) are the practical training corpus; Frontiers 2026 shows the merge is straightforward (4,702 patches, 8,095 debris px) and cross-dataset transfer costs 1–8 F1 points.
- HF re-chips are convenient but the TACO mirror leaks scenes across splits; use the official Zenodo splits for any reported number.

### Gaps
- No open Sentinel-2 litter dataset newer than MADOS (2024) with pixel labels was found for 2025–2026; the Cózar windrow catalogue has locations/sizes, not pixel masks.
- The MADOS-derived Sentinel-1 dataset (ISPRS Annals 2026) has no public download link.

---

## KQ6. Free imagery for inference, Sentinel-2 resolution/revisit limits, minimum detectable patch size

### Takeaway
All Sentinel-2 L1C/L2A imagery is free: Copernicus Data Space (OData/STAC/S3 + Sentinel Hub API 10,000 PU/month + openEO 10,000 credits/month, 12 TB/month transfer), Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`, 2017-03-28 → present, free for research/education), and Microsoft Planetary Computer (STAC `sentinel-2-l2a`, COGs from 2015-06-27, anonymous SAS tokens). Physical limits: 10 m pixels, 5-day revisit (2 satellites; 3 satellites Mar 2025–Mar 2026), ~20 % plastic cover of a 10×10 m pixel as the empirical detection floor (PLP, Cózar), 30–55 % sub-pixel for Biermann's targets, filaments >70 m long for windrows, and "~100 m² aggregations" per the Frontiers authors — individual items are invisible.

### Cited Findings
- **CDSE quotas (General user)**: 12 TB/30-day transfer (then 1 MB/s, 1 connection), 20 MB/s per connection, 4 concurrent; Sentinel Hub 10,000 requests + 10,000 processing units/month, 300/min; 50,000 COG requests/month; openEO 10,000 credits/month ("temporary boost"), 2 concurrent jobs; STAC/OData/S3 2,000 req/min (S3 only) — [CDSE Quotas](https://documentation.dataspace.copernicus.eu/Quotas.html); openEO credits — [CDSE](https://documentation.dataspace.copernicus.eu/APIs/openEO/credit_usage.html)
- CDSE products: L1C global from 2015; L2A (Sen2Cor, with SCL); Level-3 quarterly mosaics; tile ≈110×110 km; revisit 10 d (1 sat), 5 d (2 sats), 2–3 d mid-latitudes; **Sentinel-2C launched 5 Sep 2024, replaced 2A on 21 Jan 2025; 2A resumed 13 Mar 2025 in a new orbit → three satellites until 13 Mar 2026**; Collection-1 reprocessing to baseline ≥5.0 — [CDSE S2 page](https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html); launch — [Airbus](https://airbus.com/en/newsroom/press-releases/2024-09-airbus-built-sentinel-2c-satellite-successfully-launched); 2A extension — [European Spaceflight](https://europeanspaceflight.com/sentinel-2a-given-one-year-exceptional-and-temporary-extension/)
- **GEE**: `COPERNICUS/S2_SR_HARMONIZED` L2A, 2017-03-28 → 2026-11-05, 5-day revisit, UINT16 ×10,000, no B10, +1000 offset harmonised for baseline ≥04.00 (post 2022-01-25); QA60 empty 2022-01-25→2024-02-28; "free to use for research, education, and nonprofit use"; Colab example with `ee`/geemap — [GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED). MARIDA, FloatingObjects and iScience all located/labelled scenes in GEE — [Mifdal PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf); [doc/data.md](https://raw.githubusercontent.com/marccoru/marinedebrisdetector/main/doc/data.md)
- **Planetary Computer**: STAC collection `sentinel-2-l2a`, COGs, temporal extent from 2015-06-27, licence "proprietary" (Copernicus terms), providers ESA/Esri/Microsoft, assets B01–B12, SCL, AOT, WVP, visual — [STAC JSON](https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a); SAS token endpoint returned a read/list token with no credentials supplied (expiry ~1 h) — [SAS endpoint](https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a)
- Pre-processing used by the benchmarks: ACOLITE (MARIDA, MADOS, PLP, Biermann) vs Sen2Cor L2A (iScience) vs L1C/L2A mix (FloatingObjects) — [PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247); [arXiv HTML](https://arxiv.org/html/2307.02465v1); TOA vs BOA changed iScience's Durban detections (136 BOA vs 164 TOA debris; 609 vs 1,484 total objects) — [arXiv HTML](https://arxiv.org/html/2307.02465v1)
- **Minimum detectable size**: PLP unmixing detects aggregations "at least as low as 20 % coverage of a 10×10 m Sentinel-2 pixel" — [ESA exec. summary](https://nebula.esa.int/sites/default/files/neb_tec_study/1459/public/C4000131040ExS.pdf); Cózar 2024: "minimum detectable plastic surface coverage is close to 20 % of a pixel" for S2, filaments >70 m retained, proposed sensor targets 1 % — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11178853/fullTextXML); Biermann targets at 30–55 % sub-pixel — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7181820/fullTextXML); Papageorgiou 2024 (IJAEOG): 1/2/3 m² white HDPE targets — 3 m² detected in S2 and SuperDove "but with commission errors that render … practical application questionable" **[search snippet; DOAJ page 403]** — [DOAJ](https://doaj.org/article/a6add142085841e0b9bbc9a6d3b89c12); Frontiers: "only aggregations above about 100 m² are reliably detectable" — [Frontiers](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1765021/full); Mifdal: "only aggregations of floating objects are detectable with the coarse 10 m resolution"; 10×10 m PLP-2018 targets "barely visible" — [PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf); Hu 2021 (RSE 259:112414) theoretical ~0.3 % sub-pixel threshold at SNR 200 **[search snippet]** — [NOAA repo](https://repository.library.noaa.gov/view/noaa/68394)
- Typical patch extraction: MARIDA 256×256 crops per scene-date; MADOS 240×240; FloatingObjects/iScience 128×128 crops centred on labels + random negatives; full-scene inference with 480-px overlapping tiles — [Zenodo](https://zenodo.org/records/5151941); [Zenodo](https://zenodo.org/records/10664073); [arXiv HTML](https://arxiv.org/html/2307.02465v1)

### Inferences
- A student can run the iScience CLI on any CDSE/PC/GEE L2A scene for free; per-scene download is a few hundred MB (Durban example 300 MB) so a dozen scenes fit CDSE's quotas trivially.
- Because the detection floor is ~20 % of a pixel (~20 m² of plastic per 100 m² pixel) and revisit is 5 days, S2 only sees accumulation events (river plumes, windrows, fronts), which is why every benchmark is built around reported events at known hotspots.

### Gaps
- Sentinel-2A's post-March-2026 status (whether the 3-satellite constellation continued) was not confirmed.
- Commercial Sentinel Hub (sentinel-hub.com) trial terms were not checked; CDSE's Sentinel Hub API quota is the free path.

---

## KQ7. Remote-sensing foundation models with published Sentinel-2 weights usable as a shared trunk

### Takeaway
Many open S2 encoders exist and two benchmarks already test them on debris data: on MADOS (frozen encoder + UPerNet) CROMA (67.6 mIoU), GFM-Swin (64.7), TerraMind-B (69.5, per MambaMPD) and DOFA (59.6) beat a from-scratch U-Net (54.8), while Prithvi (HLS 30 m, 6 bands) does worst (50.0); on re-chipped MARIDA, SatMAE++ ViT-L (59.8 mIoU) leads. TorchGeo exposes dozens of S2 weights (ResNet/ViT/Swin MoCo/DINO/MAE/SoftCon/Satlas, DOFA, Copernicus-FM, CROMA, Panopticon) in one API; licences range from CC-BY-4.0/MIT/Apache-2.0 to ODC-BY and OpenRAIL.

### Cited Findings
- TorchGeo S2 weights (channels / licence): ResNet18/50 `SENTINEL2_ALL_MOCO`/`_DINO`/`_SOFTCON` (13, CC-BY-4.0), `_DECUR` (13, Apache-2.0), `_CLOSP`/`_GEOCLOSP` (13, OpenRAIL), `SENTINEL2_ALL_SECO_ECO` (12, MIT), Satlas `SENTINEL2_MI/SI_MS_SATLAS` (9, ODC-BY; also ResNet152, Swin-V2-T/B); ViT-S/B/L/H `SENTINEL2_ALL_MAE`/`_FGMAE` (13, CC-BY-4.0); ViT-S/B-14 DINOv2 `SOFTCON`; Tessera encoder (11 ch, CC0); FTW U-Nets (8 ch); sensor-agnostic Copernicus-FM, CROMA, DEO, DOFA, Panopticon, OlmoEarth; no MARIDA/MADOS dataset class — [TorchGeo models](https://docs.torchgeo.org/en/stable/api/models.html)
- DOFA (Xiong et al., arXiv 2403.15356, 2024): wavelength-conditioned ViT; weights HF `XShadow/DOFA` (ViT-B `DOFA_ViT_base_e100.pth`); MIT; `torchgeo.models.dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)`; takes `(x, wavelengths_in_µm)` — [GitHub](https://github.com/zhu-xlab/DOFA)
- Prithvi-EO-2.0 (IBM/NASA/JSC, arXiv 2412.02732, Dec 2024): ViT-MAE on HLS 30 m, 6 bands (B, G, R, NIR-narrow, SWIR1, SWIR2), 4.2 M samples; 300M/600M (+TL, tiny 5M, 100M-TL); Apache-2.0; fine-tune via TerraTorch — [HF](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
- Clay v1.5: ViT-MAE; Apache-2.0 code+weights; checkpoint `made-with-clay/Clay/v1.5/clay-v1.5.ckpt`; segmentation fine-tune examples — [Clay docs](https://clay-foundation.github.io/model/index.html)
- TerraMind-1.0 (IBM/ESA/FZJ, ICCV 2025, arXiv 2504.11171): S2L2A, S2L1C, S1GRD, S1RTC, DEM, RGB; 9 M TerraMesh samples; Apache-2.0; TerraTorch — [HF](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
- SpectralGPT (Hong et al., arXiv 2311.07113): trained on >1 M S2 images; checkpoints Zenodo 10.5281/zenodo.10533809 — [arXiv](https://arxiv.org/pdf/2311.07113) **[Zenodo DOI from search snippet]**
- PANGAEA MADOS mIoU (frozen): CROMA 67.55 > GFM-Swin 64.71 > RemoteCLIP 60.00 > DOFA 59.58 > SpectralGPT 57.99 > Scale-MAE 57.32 > SatlasNet 55.86 > U-Net scratch 54.79 > SSL4EO-MoCo 51.76 > Prithvi 49.98 > ViT-B scratch 48.19; at 10 % labels CROMA 32.44 vs U-Net 24.30 — [arXiv HTML](https://arxiv.org/html/2412.04204); TerraMind-B 69.5 — [MambaMPD](https://arxiv.org/html/2609.15676)
- GFM-Bench MARIDA (full FT): SatMAE++ ViT-L 59.82, LESS ViT-B 55.64, SatMAE ViT-B 54.33, SpectralGPT 44.72, CROMA 43.04, Scale-MAE 41.12 — [arXiv HTML](https://arxiv.org/html/2503.12843)
- Community example: ViT-S/16 MAE-SSL + UNet++ on MARIDA (GitHub Nakshu35) **[unvetted student repo]** — [GitHub](https://github.com/Nakshu35/Marine-Debris-Segmentation-Model)

### Inferences
- For a shared trunk across "chemical + trash" tasks, CROMA / DOFA / TerraMind / Copernicus-FM are the defensible choices: open licences, native 12–13-band S2 input, and already benchmarked on MADOS (which contains both oil spills and debris). Prithvi's 6-band 30 m HLS pre-training is a poor match for 10 m debris.
- Frozen-encoder scores (≈60–69 mIoU) are in the same range as MariNeXt (59–64 reproduced/reported), so a Colab project that trains only a UPerNet head is competitive and cheap.

### Gaps
- No publication yet fine-tunes DOFA/TerraMind end-to-end on MARIDA/MADOS with per-class Marine-Debris scores.

---

## KQ8. Known failure modes and why hotspot-trained models may not transfer (student caveat check)

### Takeaway
The student's caveat is half right: every benchmark was indeed built from reported events at known hotspots (Accra, Durban, Honduras/Guatemala river mouths, Manila, Mytilene targets), and all papers document look-alike confusions (Sargassum/algae, foam/whitecaps, ships and wakes, cloud fringes, coastlines, sunglint, turbid plumes); cross-dataset transfer costs 1–8 F1 points (Frontiers) and the original FloatingObjects U-Net collapsed on Durban (F1 0.497). But "non-targeted locations need commercial high-res imagery" is not what the literature says — the stated remedies are more negatives/diversity, label refinement, and calibrated thresholds; high-resolution (PlanetScope 3 m, drones) is used for *validation*, and even 3 m² targets produce commission errors in SuperDove.

### Cited Findings
- MARIDA confusions: NatM vs MD (brown/green debris at river mouths), Sparse Sargassum vs sparse debris, small vessels classified as MD, ship edges vs wakes, foam/whitecaps/sunglint/waves as bright look-alikes (U-Net labelled some Foam as MD), mixed-resolution band artefacts, thin clouds; uneven geography, sparse low-confidence labels — [PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247)
- FloatingObjects: coastline waves, ships, cloud fringes flagged; models "overfit early with accuracy stagnating on unseen validation and test regions"; domain shift between regions/acquisitions named as the key limitation — [PDF](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf)
- iScience: Mifdal U-Net produced 20,830 (BOA) / 33,665 (TOA) objects on Durban vs 609 / 1,484 for UNet++; ground truth 6,448 debris px of 11,997,846 (0.05 %); failures with dense cloud, haze on TOA, ships (Accra-2), Sargassum/algae response inherited from FDI-based labels; model under-estimates debris probability → thresholds 0.03–0.13 — [arXiv HTML](https://arxiv.org/html/2307.02465v1)
- Frontiers: residual FPs mostly ship wakes and whitecaps (FPR ≈0.3 %); foam largely rejected (mean conf 0.044); fixed τ=0.5 not universally optimal; "geographic-diversity claim rests on two cross-dataset directions" — [Frontiers](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1765021/full)
- Biermann: works best in clear, low-turbidity water; whitecap scenes excluded; cannot discriminate plastic types; only Mytilene has in-situ truth — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7181820/fullTextXML)
- Hu: mixed band resolutions + <10 % sub-pixel cover distort MSI spectra; pixel averaging and subtraction required — [NOAA repo](https://repository.library.noaa.gov/view/noaa/68394)
- Cózar: manual rejection of aircraft exhaust, cloud edges, ship trails — [Europe PMC XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11178853/fullTextXML)
- PANGAEA: MADOS is where per-band normalisation and spectral breadth matter most — [arXiv HTML](https://arxiv.org/html/2412.04204)

### Inferences
- Hotspot bias is a *label* bias (positives come from reported events), so the fix in the literature is negative mining over ordinary ocean (S2Ships, random background, proxy-classifier false positives), not a change of sensor.
- Any student evaluation should hold out whole scenes/regions (MARIDA/MADOS do; TACO mirror does not) and report Marine-Debris-class F1/IoU plus a false-positive rate on debris-free scenes.

### Gaps
- No study quantifies false-positive rates over large debris-free ocean areas for the released detectors (iScience gives Durban counts only).

---

## Incidental / out-of-scope note
- A Chinese-language slide deck about Taiwan marine plastic (student team, blockchain theme) surfaced in a search PDF; irrelevant here and not cited (URL not captured).

## Consolidated source list
- https://arxiv.org/abs/2307.02465 · https://arxiv.org/html/2307.02465v1 · https://github.com/marccoru/marinedebrisdetector · https://raw.githubusercontent.com/marccoru/marinedebrisdetector/main/doc/data.md · https://raw.githubusercontent.com/marccoru/marinedebrisdetector/main/doc/model_training.md · https://marcrusswurm.users.earthengine.app/view/marinedebrisexplorer
- https://arxiv.org/abs/2210.08506 · https://arxiv.org/pdf/2210.08506 · https://github.com/sheikhazhanmohammed/SADMA
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0262247 · https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8740969/fullTextXML · https://zenodo.org/records/5151941 · https://github.com/marine-debris/marine-debris.github.io · https://marine-debris.github.io/
- https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7181820/fullTextXML (Biermann 2020; DOI 10.1038/s41598-020-62298-z)
- https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/ · https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/isprs-annals-V-3-2021-285-2021.pdf · https://github.com/ESA-PhiLab/floatingobjects · https://zenodo.org/records/15706997 · https://github.com/eds-book/b34facfa-cea8-48f5-89f6-f11ce00812a9
- https://zenodo.org/records/10664073 · https://github.com/gkakogeorgiou/mados · https://marine-pollution.github.io/ · https://marine-pollution.github.io/Report_Patches_stats.html · https://www.sciencedirect.com/science/article/pii/S0924271624000625
- https://arxiv.org/html/2609.15676 (MambaMPD) · https://arxiv.org/html/2412.04204 (PANGAEA) · https://github.com/VMarsocci/pangaea-bench · https://arxiv.org/html/2503.12843 (GFM-Bench) · https://huggingface.co/datasets/GFM-Bench/MARIDA · https://huggingface.co/datasets/Ehzoahis/MARIDA · https://huggingface.co/datasets/isp-uv-es/marida-taco
- https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1765021/full · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10133222/ (MAP-Mapper) · https://arxiv.org/html/2403.18067 (survey) · https://arxiv.org/abs/2106.03694 (Sannigrahi) · https://isprs-annals.copernicus.org/articles/XI-3-2026/187/2026/
- https://zenodo.org/records/7085112 · https://zenodo.org/records/3752719 · https://plp.aegean.gr/plastic-litter-project-2022/ · https://nebula.esa.int/sites/default/files/neb_tec_study/1459/public/C4000131040ExS.pdf · https://mdpi-res.com/d_attachment/remotesensing/remotesensing-12-02013/article_deploy/remotesensing-12-02013.pdf · https://doaj.org/article/a6add142085841e0b9bbc9a6d3b89c12
- https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11178853/fullTextXML (Cózar 2024) · https://repository.library.noaa.gov/view/noaa/68394 (Hu 2022) · https://web-ocean-scan.vercel.app/ · https://docs.torchgeo.org/en/stable/api/datasets/nasa-marine-debris.html
- https://openaccess.thecvf.com/content/ICCV2021/papers/Cheng_FloW_A_Dataset_and_Benchmark_for_Floating_Waste_Detection_in_ICCV_2021_paper.pdf · https://docs.voxel51.com/dataset_zoo/datasets_hf/afo_aerial_floating_objects.html · https://data.mendeley.com/datasets/zshjxgfmc3/1 · https://data.4tu.nl/datasets/82306610-84ab-4aed-a759-5ec92567b7d5 · https://doi.org/10.3390/w17223318
- https://documentation.dataspace.copernicus.eu/Quotas.html · https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html · https://documentation.dataspace.copernicus.eu/APIs/openEO/credit_usage.html · https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED · https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a · https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-2-l2a · https://airbus.com/en/newsroom/press-releases/2024-09-airbus-built-sentinel-2c-satellite-successfully-launched · https://europeanspaceflight.com/sentinel-2a-given-one-year-exceptional-and-temporary-extension/
- https://docs.torchgeo.org/en/stable/api/models.html · https://github.com/zhu-xlab/DOFA · https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M · https://clay-foundation.github.io/model/index.html · https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base · https://arxiv.org/pdf/2311.07113
- Unverified/secondary: https://www.researchgate.net/publication/392455527_Advanced_Deep_Learning_Framework_AquaSense_for_Comprehensive_Marine_Pollutant_Detection_Using_Sentinel-2_Multispectral_Data · https://www.researchsquare.com/article/rs-11300519/v1 · https://github.com/Nakshu35/Marine-Debris-Segmentation-Model

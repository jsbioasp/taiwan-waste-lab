# Taiwan-based datasets, imagery and published work for (1) waste/marine-debris detection from satellite/aerial/drone imagery and (2) spectral chemical identification — status as of 2026-11-06

Scope note: Research done 2026-11-06 via web search (EN + 繁體中文) and page fetches. Every item is tagged **[DOWNLOADABLE NOW]**, **[REQUEST/APPLICATION]**, **[EXISTS PER PAPER/NEWS, NOT RELEASED]** or **[UNVERIFIED]**. Several Taiwanese sites blocked or timed out on fetch (MDPI, inside.com.tw, scidm.nchc.org.tw zh pages, mdimagenet.namr.gov.tw is JS-only, icctaiwan.org.tw DNS failed, ci.taiwan.gov.tw TLS cert expired); where I relied on search snippets instead of the page, it is marked.

## Key question 1 — What Taiwan ground truth could serve as evaluation labels for a model trained on MARIDA / FloatingObjects / AerialWaste (the way COA shelter data served the prior project)?

### Takeaway
There is no Taiwan pixel/polygon label set for Sentinel-2 floating debris; the usable Taiwan "ground truth" is (a) NGO coastal rapid-screening surveys that give per-10-km-segment debris density (bags per 100 m) for 121 stations (2018–19) and for 14 hotspots re-surveyed in 2022, plus a Tamsui-basin river survey at 500 m resolution (2020, 2023); (b) the Ocean Conservation Administration's monthly county-level cleanup tonnage statistics (CSV/JSON API); and (c) the Ministry of Environment's drone/AI coastal survey programme, which has 4-level coverage grades for hundreds of km of coast but is not released as data. All are coarse (segment-level, not object-level) and work best as evaluation/ranking targets rather than training labels.

### Cited Findings

**A1. Greenpeace Taiwan + Society of Wilderness (荒野保護協會, SOW) national coastal rapid screening (全臺海岸廢棄物快篩調查), 2018–2019 — [EXISTS; per-station data not published online]**
- Taiwan's ~1,210 km main-island coastline was divided into 121 stations (~10 km apart); surveys ran quarterly for one year (first round July 2018, second October 2018); volume at each station was converted into "large black garbage bag" counts and extrapolated island-wide; first round ≈12.66 million litres, second ≈10.95 million litres; 56% (round 1) / 49% (round 2) of debris sat on 10% of the coastline; the page publishes no raw per-station table or map — [Greenpeace Taiwan, 2019-01](https://www.greenpeace.org/taiwan/update/12782/第二次快篩結果，限塑淨灘有感/)
- Final one-year result: ≈12,272,000 litres / 646 tonnes on the coast; north coast (Keelung, New Taipei, Taoyuan) and southwest coast (Changhua, Yunlin, Chiayi, Tainan) most polluted; the NGOs called for a "海岸廢棄物污染標準" — [Greenpeace Taiwan press release 7747](https://www.greenpeace.org/taiwan/press/7747/)
- Method detail: ~11.9 km of coast actually walked (≈1% of coastline); each station's volume converted to 90 L bags; 1,583 bags measured → 158,300 bags extrapolated; five dirtiest stations: 新北市瑞芳區台2線, 嘉義縣布袋鎮好美寮, 臺南市青草崙, 雲林縣麥寮鄉六輕, 彰化縣大城鄉海堤 — [TechBang report](https://www.techbang.com/posts/60852-coastal-rapid-sieve-survey-found-that-nearly-160000-bags-of-marine-litter-were-accumulated-on-the-coast-of-taiwan-island)
- Partner organisations: Greenpeace, SOW, 台灣環境資訊協會, 慈心有機農業發展基金會, 黑潮海洋文教基金會, 海洋公民基金會, NTOU 海洋事務與資源管理研究所, 愛海的旅行 — [Greenpeace Taiwan](https://www.greenpeace.org/taiwan/update/12782/第二次快篩結果，限塑淨灘有感/)
- Background article on the method ("從淨灘到快篩") — [Greenpeace Taiwan](https://www.greenpeace.org/taiwan/update/1897/從淨灘到「快篩」：問診全臺的海岸垃圾/)

**A2. SOW 2022 hotspot re-survey (2022海廢快篩熱點複查行動), press release 2023-02-23 — [DOWNLOADABLE NOW as a 4-page PDF press release containing the per-site table]**
- SOW re-surveyed the 14 dirtiest 2018 segments, once per quarter (4 surveys), same method (90 L bag = 1 bag), sponsored by Taiwan Porsche; dirtiest five in 2022: 新北市瑞芳區, 雲林縣麥寮鄉, 桃園市觀音區, 苗栗縣通霄鎮, 屏東縣滿州鄉; four sites increased (麥寮, 觀音, 通霄, 滿州; 滿州 +13.35 bags) — [SOW press release PDF](https://www.sow.org.tw/sites/sow/files/xin_wen_gao__huang_ye_qi_dong_hai_fei_fu_cha__jian_shi_tai_wan_hai_fei__fa_xian_bu_fen_re_dian_bu_jian_fan_zeng_.pdf)
- Table 1 in that PDF (average total bags per 100 m, 2018 avg → 2022 avg): 新北市瑞芳區 144.8→119.5; 彰化縣大城鄉 73.3→34; 台南市安南區 68→19.75; 新北市瑞芳區(second segment) 63.5→15; 雲林縣麥寮鄉 62.5→71.25; 桃園市觀音區 58→61; 新北市金山區 55.5→25.5; 彰化縣線西鄉 49→38; 苗栗縣通霄鎮 43→50.75; 桃園市蘆竹區 40→32; 台南市七股區 39.8→27.5; 嘉義縣布袋鎮 34.8→14.625; 屏東縣滿州鄉 33.9→47.25; 嘉義縣東石鄉 not surveyable; totals 766.10→556.13 — [SOW press release PDF, p.3–4](https://www.sow.org.tw/sites/sow/files/xin_wen_gao__huang_ye_qi_dong_hai_fei_fu_cha__jian_shi_tai_wan_hai_fei__fa_xian_bu_fen_re_dian_bu_jian_fan_zeng_.pdf)
- The same PDF states the 2018 survey was "deeply noted" by the EPA and OCA and that the government launched the 向海致敬 policy the following year — [SOW press release PDF](https://www.sow.org.tw/sites/sow/files/xin_wen_gao__huang_ye_qi_dong_hai_fei_fu_cha__jian_shi_tai_wan_hai_fei__fa_xian_bu_fen_re_dian_bu_jian_fan_zeng_.pdf)

**A3. SOW river debris rapid screening (河川廢棄物快篩), Tamsui basin 2020 and 2023 — [EXISTS PER NEWS; raw segment data not confirmed online]**
- Method adapted from Japan; volunteers count by bicycle in 500 m segments using 14 L bags as unit; 2020 Tamsui/Keelung/Xindian/Dahan rivers: 12,987.5 bags (≈181,825 L); 2023 re-survey ≈ doubled; ~80% of waste in ~6% of riverbank segments; hotspots 關渡橋, 蘆洲, 汐止, 社子島右側; top types 2023: plastic bottles/cans 28.8%, cups/straws/tableware 24.1%, plastic bags 15.3%; article does not say data/maps are online — [TVBS, 2023-11-19](https://news.tvbs.com.tw/life/2309738)

**A4. Ocean Conservation Administration (海洋委員會海洋保育署, OCA) open data on data.gov.tw — [DOWNLOADABLE NOW]**
- 海洋廢棄物清理狀況統計 (Marine debris cleanup status statistics): monthly county-level CSV + JSON, ROC 108.06 → 115.06 (≈85 monthly files), fields = county, number of cleanup sites, cleanup count, participants, tonnes by source (海漂/海底/淨灘/船舶人員產出/岸上垃圾桶), tonnes by type (寶特瓶, 鐵罐, 鋁罐, 玻璃瓶, 廢紙, 竹木, 保麗龍, 廢漁具漁網, 無法分類), disposal (焚化/回收/掩埋); licence 政府資料開放授權條款第1版 (OGDL v1); free; published 2025-02-20; API pattern `https://iocean.oca.gov.tw/oca_datahub/WebService/GetData.ashx?id=<resource-id>` — [data.gov.tw/dataset/172912](https://data.gov.tw/dataset/172912)
- 海漂底廢棄物及微型塑膠分布調查與海廢監控建置計畫: only resource is the FY2024 (113年度) project report PDF; no CSV/shapefile; OGDL v1; published 2025-11-28 — [data.gov.tw/dataset/175841](https://data.gov.tw/dataset/175841); report PDF at [oca.gov.tw](https://www.oca.gov.tw/filedownload?file=research/202509260932470.pdf&filedisplay=113%E5%B9%B4%E5%BA%A6%E6%B5%B7%E6%BC%82%E5%BA%95%E5%BB%A2%E6%A3%84%E7%89%A9%E5%8F%8A%E5%BE%AE%E5%9E%8B%E5%A1%91%E8%86%A0%E5%88%86%E5%B8%83%E8%AA%BF%E6%9F%A5%E8%88%87%E6%B5%B7%E6%B4%8B%E5%BB%A2%E6%A3%84%E7%89%A9%E7%A0%94%E7%A9%B6%E5%88%86%E6%9E%90%E8%A8%88%E7%95%AB.pdf)
- 海洋廢棄物管理工作成果年報 (annual reports 2021–2023, PDF) — [data.gov.tw/dataset/175664](https://data.gov.tw/dataset/175664) (search snippet; page not fetched)
- OCA reports 2024 cleanup >20,439 t and 10,357 t through Q3 2025; 912.935 t of waste fishing nets recovered over five years — [UDN, 2025](https://udn.com/news/story/7327/9109032)

**A5. Ministry of Environment (環境部) cleanup registration platform — [REQUEST/LOGIN; no open dataset found]**
- The 海岸淨灘認養系統 / 海岸清理資訊平台 requires an account for adoption, cleanup applications and result reporting; URL ecolife2.moenv.gov.tw/BeachCleanup — [New Taipei EPD manual PDF](https://www.epd.ntpc.gov.tw/UploadFile/CoaAdo/20250806165126434406.pdf)
- MOE statistics: >10,000 cleanup events and ~170,000 participant-visits per year; top-10 cleanup-application hotspots led by Taoyuan (4 sites) — [UDN 9550630](https://udn.com/news/story/7266/9550630)

**A6. Ministry of Environment, Environmental Management Administration (環境管理署, EMA) drone + AI coastal debris surveys (向海致敬 programme) — [EXISTS PER NEWS/JOURNAL; imagery, labels and model NOT released]**
- 2020–2025: >178 aerial inspection flights photographing >400 km of coast; 2025: 30 flights, 73.35 km, 23 flights led to cleanup referrals; flight locations New Taipei, Keelung, Miaoli, Taichung, Chiayi, Tainan, Pingtung, Penghu; AI vs manual interpretation agreement >81%; "常駐型無人機自動化巡查" (resident automated drone patrol, ~10 days per hotspot) planned for 2H 2026; 向海致敬 removed ≈362,000 t of waste over ~6 years — [UDN 9550631](https://udn.com/news/story/7266/9550631)
- 2024 figures reported in news: 15 coasts inspected, 13.183 km² photographed, >582 t identified and cleared; AI/manual agreement >80%; ~2 km per flight; inspection time cut from ~1 week to as fast as 1 day — [Taiwan News zh 6066937](https://www.taiwannews.com.tw/zh/news/6066937) (search snippet; page not fetched); English coverage also cites "trained on more than 8,000 historical aerial photographs" — [TVBS English 2818373](https://news.tvbs.com.tw/english/2818373) **[UNVERIFIED: snippet only; page redirected]**
- 向海致敬 management rule: 168 monitoring points nationwide; dirty sites must be cleaned within a week and re-inspected — [Business Today, 2023-10-19](https://www.businesstoday.com.tw/article/category/183035/post/202310190027/) (search snippet)
- Technical description in 《工程》(Journal of the Chinese Institute of Engineers magazine) vol. 98 no. 4, Dec 2025, authors from EMA (副署長 劉瑞祥 et al.) and vendor 環輿科技股份有限公司 (總經理 管永愷 et al.) — [CIE PDF](https://www.cie.org.tw/var/file/0/1000/img/2/708585316.pdf). **Partially legible after text extraction (custom font encoding)**; fragments that were readable: target GSD ≤3 cm; four-stage workflow (planning → flight → stitching/interpretation → AI recognition → hotspot reporting via Ecolife platform → cleanup tracking); model evolved YOLOv4 → YOLOv9; training set grew from ~495 images (2022 model) to 14,489 images (2025/114年 model); agreement rose from 70% to 81–96%; 4-level coverage grading (<1%, 1–3%, 3–6%, >6% debris coverage of beach area); 300–400 photos per mission, ~4–5 h on CPU vs ~1.5 h on GPU; NGO 100 m rapid-screening used as cross-check against drone coverage. Treat these numbers as **[UNVERIFIED pending a clean read of the PDF]**.

**A7. SOW / ICC beach-cleanup citizen-science data — [EXISTS; aggregated in papers; raw download not confirmed]**
- SOW and academics analysed 541 cleanup records (2004–2016) in Marine Pollution Bulletin: 90.8% of items fully or partly plastic; top five = plastic bags, bottle caps, disposable tableware, fishing gear, straws — [CNA, 2018-10-03](https://www.cna.com.tw/news/ahel/201810030133.aspx)
- SOW has led volunteer ICC-protocol cleanups since 2008 and built the 愛海小旅行 marine-debris information platform in 2015 — [SOW newsletter 2016](https://www.sow.org.tw/sites/sow/files/newsletter_20161221.pdf)
- icctaiwan.org.tw could not be resolved (DNS failure) on 2026-11-06 **[UNVERIFIED availability]**

### Inferences
- The 2018 (121 stations) and 2022 (14 hotspots) SOW tables give a ranking target: a Sentinel-2 or drone model's per-segment debris score can be rank-correlated (Spearman) against bags/100 m, mirroring how the prior project evaluated against COA shelter data. The 2022 table is fully reproduced in the PDF above; the 2018 121-station table must be requested from SOW (contact in PDF: 徐筱珺 02-23071317#32).
- OCA monthly county tonnage is too coarse for pixel evaluation but can serve as a sanity check / time-series correlate for county-level aggregated detections.
- The EMA 4-level coverage grades for ~400 km of coast are the closest Taiwan analogue to AerialWaste-style site labels, but they are not public; a student could request a sample under the OGDL or via the vendor 環輿科技.

### Gaps
- No per-station CSV/GeoJSON of the 2018–19 121-station screening was found online (Greenpeace and SOW pages give totals and rankings only).
- No Taiwan Sentinel-2-resolution floating-debris annotation set exists in any source found.
- The EMA drone imagery (>14k labelled images per the CIE article) has no public release; licence and access path unknown.
- ICC Taiwan per-event item counts: site unreachable; whether downloadable is unknown.

## Key question 2 — Is there any Taiwan imagery + label dataset for waste that is actually downloadable?

### Takeaway
Yes — two: (1) NAMR's **MDImageNet** (國家海洋研究院 海廢影像資料集), released 2026-07-30 under CC BY 4.0 with >20,000 real coastal images and >42,000 YOLO bounding boxes in 34 ICC-based classes, though the download page is JavaScript-only and the exact download mechanics could not be confirmed; and (2) NTOU's **HAIDA** high-resolution aerial trash dataset (MIT licence, Google Drive links on GitHub). Both are ground/low-altitude imagery, not satellite.

### Cited Findings

**MDImageNet (NAMR, 國家海洋研究院) — [DOWNLOADABLE per licence; mechanics unverified]**
- Released 2026-07-30 as "全臺首套 AI-Ready「MDImageNet海廢影像資料集」": >20,000 real coastal images, >42,000 annotated debris objects, taxonomy based on ICC with an added "NAMR 三層級分類"; CC BY 4.0 for research, education and industry; guided by 海洋委員會, AWS provides AI/cloud support, ITRI's AIdea platform runs the competition; NAMR's marine data platform holds >30 billion records — [CTEE, 2026-07-30](https://www.ctee.com.tw/news/20260730701780-431204)
- Press-release repost: images are "real coastal photos from NAMR's multi-year coastal survey work"; capture platform (drone vs handheld) not stated; baseline model released by NAMR; quotes from an OAC deputy minister (romanised "Wu Hsin-hsiu" in the repost), NAMR director 陳璋玲 (Chen Chang-ling) and AWS Taiwan GM 王定愷 (Wang Ting-kai); no download URL in the release — [Yam News repost, 2026-07-30 18:30](https://n.yam.com/Article/20260730961262)
- Competition page details: >20,000 images with YOLO bounding boxes; 34 source classes (IDs 0–33) mapped to 20 scoring classes (IDs 0–19: 寶特瓶 plastic_bottle, 塑膠瓶蓋, 其他飲料瓶與容器, 非食物的瓶罐與容器, 外帶飲料杯, 吸管, 免洗餐具, 塑膠提袋, 食品包裝袋, 鐵鋁罐, 菸蒂, 打火機, 鋁箔包/利樂包, 玻璃瓶, 漁網與繩子, 漁業浮球, 釣魚用具, 針筒針頭, 牙刷, 其他); extra source classes include 塑膠蓋, 杯狀物, 發泡塑膠浮具, 菸盒, 紡織品, 網狀物, 人工碎片, 免洗食品容器, 軟式浮具, 保麗龍容器, 純鋁包裝, 漁籠及餌料; training data "released publicly July 2026"; reference site https://mdimagenet.namr.gov.tw/ ; AIdea topic https://www.aidea-web.tw/topic/08e3111b-2c90-45cb-86dd-d0fadd09ed8b ; preliminary round 2026-08-11→09-01, final 09-03→09-17; mAP@0.5 via COCO API; NetScore lightweight award; contact mdivrc@digitimes.com — [DIGITIMES competition page](https://www.digitimes.com.tw/seminar/mdivrc_2026/index.html)
- Registration closed 2026-08-10; >40 teams including 4 international; NAMR plans to expand scenes and push lightweight models to phone patrols, drone imagery and long-term monitoring — [DIGITIMES news 763524](https://www.digitimes.com.tw/tech/dt/n/shwnws.asp?CnlID=13&id=763524) (search snippet)
- https://mdimagenet.namr.gov.tw/ returned only "載入中" (JS app) to the fetcher on 2026-11-06 — **[UNVERIFIED whether open download or registration is required]**

**HAIDA / HAIDA2 (NTOU) — [DOWNLOADABLE NOW]**
- GitHub repo "HAIDA-Trash-Dataset-High-Resolution-Aerial-image" by 廖育賢 (LiaoSteve), 江昶勳, 張仲崴, 吳御齊; MIT licence; Google Drive links for HAIDA (https://drive.google.com/file/d/12ieYe9479Yee9eYADrGal6CYKxg2lwG_/view) and HAIDA2 (https://drive.google.com/file/d/12-wJ_Pi3YRQLfpAUQSgTHpdKzfJ2Xa_N/view); citation = Y.H. Liao & J.G. Juang, NTOU master thesis 2021 "Application of UAVs for Marine Trash Detection and Real-Time Monitoring System"; README does not state image count, classes, GSD or beach — [GitHub](https://github.com/LiaoSteve/HAIDA-Trash-Dataset-High-Resolution-Aerial-image)
- Companion paper: Liao & Juang, "Automatic Marine Debris Inspection", Aerospace 10(1):84, 2023, DOI 10.3390/aerospace10010084; proposes HAIDA + YOLOv4-based detector, compares YOLOv3/YOLOv4/Scaled-YOLOv4 with k-fold CV, UAV used for coastal detection — [NTOU scholars](https://scholars.ntou.edu.tw/handle/123456789/23732)

**Other Taiwan waste-image resources checked**
- NCHC 科學資料管理平台 (scidm.nchc.org.tw) search for 垃圾/廢棄物/海廢 returned no waste-image dataset (only seedling and agricultural sets) — [search result](https://scidm.nchc.org.tw/ja/dataset/smarter_05_0000_1_0000_20230811_0000_000000)
- AIdea previously hosted a SOW-data "海岸廢棄物快篩調查 / 臺灣海洋廢棄物預測" tabular prediction topic (not imagery) — [Taipei high-school AI competition PDF](https://www.fg.tp.edu.tw/wp-content/uploads/doc/tfglib/臺北市高中職學生大數據與AI應用競賽.pdf) **[UNVERIFIED; PDF not fetched]**
- ITRI used AI to assist SOW's rapid screening (2022) — [Business Today, 2022-07-21](https://www.businesstoday.com.tw/article/category/183027/post/202207210031/) (search snippet)

### Inferences
- MDImageNet is the obvious Taiwan fine-tuning/evaluation set for ground-level and low-altitude drone detection; its ICC taxonomy aligns with SOW/OCA item categories, so a model fine-tuned on it can be compared directly against ICC cleanup counts.
- Neither MDImageNet nor HAIDA is satellite-scale; a MARIDA/FloatingObjects model cannot be evaluated on them directly, but a student could use them for a drone-scale branch of the project.

### Gaps
- MDImageNet: capture platform, per-image geolocation, train/test split sizes and the actual download/registration flow are unconfirmed (site JS-only; inside.com.tw article returned 403).
- HAIDA: image count, classes and GSD not in README; the MDPI full text (which has them) returned 403.

## Key question 3 — Land waste / illegal dumping: Taiwan programmes, datasets and analogues (衛星影像 AI 非法棄置)

### Takeaway
Taiwan has no released illegal-dump imagery+label dataset. The government's satellite monitoring of dumping runs through the Ministry of the Interior's 國土利用監測 (national land-use change detection, SPOT imagery, executed by NCU CSRSR since 2018), whose "變異點" (variation points) include waste/soil-dumping classes and are shared with the environment ministry; The Reporter downloaded the raw variation-point file in 2022, so this is the best analogue to the prior project's "government open data as evaluation labels" pattern. EPA/MOE statements reference drones, satellite imagery and image recognition for dumping, but no accuracy figures, vendors or data releases were found.

### Cited Findings
- Executive Yuan 2021-06-17 "廢棄物管理精進策略": EPA to use 空拍機, 衛星影像, GPS and 影像辨識 with patrol volunteers to detect dumping early (context: renovation waste dumped on leased farmland) — [ey.gov.tw](https://www.ey.gov.tw/Page/448DE008087A1971/18174cc8-0a41-4464-8150-238289659dad)
- The Reporter (2022-08-23) data analysis: 營建署 has run satellite land-use monitoring since 1999; 132,254 variation points 2014–Jul 2022 across 64 types; for the three waste-related types (傾倒廢棄物、土 / 堆積土石 / 處理廢棄物) 4,748 confirmed violations; EPA case list 945 cases 2011–Aug 2022; 770 points within 3 km of 台61線; 營建署 began sharing satellite monitoring data with EPA in 2022; no AI mentioned ("系統比對運算"); The Reporter downloaded the raw 2008–Jul 2022 variation-point file from 營建署's 國土利用監測整合資訊網 — [The Reporter](https://www.twreporter.org/a/enterprise-wastes-black-market-satellite-images-find-abandoned-dirt-and-waste)
- 土地利用監測辦法 defines 變異點 as changed locations found by comparing multi-date aerial/remote-sensing imagery; central authority monitors ≥2×/yr (in practice 6 periods/yr on land, 2 for coast/sea); since 2018 (107年) 國土管理署 commissions NCU 太空及遙測研究中心; SPOT imagery is the main source — [NLMA PDFs](https://www.nlma.gov.tw/uploads/files/6b2e70563270a0736060d06074591166.pdf) and [NLMA](https://www.nlma.gov.tw/filesys/file/chinese/dept/rp5/12-041080328.pdf) (search snippets) **[UNVERIFIED page-level]**
- Earlier programme stats: 2012 — four quarterly detection periods, 1,152 variation points reported, average violation-discovery rate 38.8% — [NLMA PDF](https://www.nlma.gov.tw/uploads/files/31c56d713646fff390fcdb5cd3fdabe8.pdf) (search snippet)
- Audit (審計部) notes rising numbers of un-closed variation-point cases over 5 years — [cy.gov.tw](https://cybsbox.cy.gov.tw/CYBSBoxSSL/edoc/download/69429) (search snippet)
- Academic precedents (non-deep-learning): NCU thesis on Taoyuan illegal-dump site potential mapping using data mining + multi-temporal satellite spectral change (11,285 ha high-potential area, 9.4% of county) — [NCU IR](https://ir.lib.ncu.edu.tw/handle/987654321/3553); NTU "3S" study using FORMOSAT-2, CART spatial rules + NDVI change over six northern counties — [NTU scholars PDF](https://scholars.lib.ntu.edu.tw/bitstreams/101f6bce-cec8-4cdf-b9ed-ec92b0d74fee/download)
- Water Resources Agency 第四河川局 combined image recognition, LINE Bot and drones to patrol river dumping, waste water and gravel theft; old pixel-difference system produced false alarms from vehicles/animals; product being commercialised (date not shown) — [iThome 134401](https://ithome.com.tw/news/134401)
- Searches (EN + 繁中) for "環境部/環管署 衛星影像 AI 非法棄置", "環保局 無人機 AI 非法棄置", "衛星影像 深度學習 非法棄置 論文 台灣" returned only foreign programmes (PoliMi PERIVALLON/AerialWaste, Abu Dhabi EAD, Shizuoka) — e.g. [PoliMi](https://www.polimi.it/en/research/news/detail/ai-and-satellites-to-detect-illegal-landfills-with-90-accuracy), [Shizuoka Pref.](https://www.pref.shizuoka.jp/kurashikankyo/recycle/1049488/1072063.html)

### Inferences
- The 變異點 file (point/polygon, type-coded, dated, with verification outcome) is structurally the same kind of evaluation label the prior project used: an AerialWaste-trained classifier applied to NLSC orthophoto tiles or FORMOSAT chips at those coordinates could be scored on confirmed vs rejected dumping points.
- Farmland-factory (農地工廠) monitoring uses the same 國土利用監測 pipeline (search snippet mentions 工廠管理輔導法 comparisons), so the same data path covers that analogue.

### Gaps
- No data.gov.tw dataset ID for variation points was found; access appears to be via 國土利用監測整合資訊網 (URL not captured) — needs direct check.
- No Taiwan deep-learning illegal-dump paper, model accuracy or vendor name was found in either language.
- The EMA drone/AI programme (KQ1-A6) is coastal only; no equivalent inland dumping AI programme with public numbers surfaced.

## Key question 4 — Imagery over Taiwan: practical access path and cost (FORMOSAT, orthophotos, Sentinel-2, NCHC, Civil IoT, hyperspectral)

### Takeaway
Cheapest paths: Sentinel-2 (free, global) and NLSC's free WMTS orthophoto tiles (PHOTO2 and yearly PHOTO2014–2020 layers, EPSG:3857, no application per community reports, licence terms unclear). FORMOSAT-2 (2015–16) and FORMOSAT-5 (2018–2025) scenes over Taiwan are available free-of-charge to domestic academia in limited quantity via the NSTC data market (scidm.nchc.org.tw/organization/tasa) after an application/review; commercial FORMOSAT purchase goes through TASA distributors and fsimage.tasa.org.tw. FORMOSAT-8 (launched 2025-11-29, 1 m / 0.7 m super-res, red-edge band) has no published academic channel yet. Aerial Survey Branch (航測及遙測分署) orthophotos are sold/WMS-served to agencies; no open-download product was found. No Taiwan airborne hyperspectral open archive exists.

### Cited Findings

**FORMOSAT (福衛) via TASA (國家太空中心)**
- TASA purchase page says image requests/purchases go through TASA-authorised distributors; TASA 影像營運系統 (fsimage.tasa.org.tw) requires a 我的E政府 account for individuals, and institutions apply for an account after review — [TASA purchase page](https://www.tasa.org.tw/zh-TW/service-and-tech/formosat-images/purchase) and [fsimage.tasa.org.tw](https://fsimage.tasa.org.tw/home) (search snippets; purchase page body did not render on fetch)
- NSTC data market (scidm.nchc.org.tw/organization/tasa) offers part of the historical FS2 (2015–2016) and FS5 (2018–2025) Level 1A and Level 4 pan/multispectral products over Taiwan main island; "供研究使用…限期免費、限量提供" (free for research, limited period/quantity) — search-snippet summary citing [TASA in-progress page](https://www.tasa.org.tw/inprogress.php?c=20021501) **[UNVERIFIED page text]**
- data.gov.tw "福爾摩沙衛星影像圖資": publisher 國家科學及技術委員會; opens part of historical FS2 and 2018–2023 FS5 imagery; OGDL v1; free; resources are a CSV index (https://mas.nstc.gov.tw/OPENDATA/GetFile?format=csv&serialno=467&fileodr=2) and an operation manual PDF; published 2021-11-02, metadata updated 2026-07-06; contact 02-27378005 — [data.gov.tw/dataset/146051](https://data.gov.tw/dataset/146051); a search snippet of the same listing says applicants must state purpose and be approved by the space centre before imagery is provided **[REQUEST/APPLICATION]**
- NCHC scidm TASA organisation page exists (showcase "福爾摩沙衛星影像示意圖", 9,111 downloads, licence "其他") — [scidm showcase](https://scidm.nchc.org.tw/en/dataset/showcases/formosa-satellite-image2); organisation URL https://scidm.nchc.org.tw/en/organization/tasa
- 2018 price reference: a FORMOSAT-5 full-island mosaic was quoted at NT$200k–300k commercially — [Inside](https://www.inside.com.tw/article/12080-formosat-5-photo) (dated; indicative only)
- 2021: FS2/FS5 imagery became purchasable through the Aerial Survey Office (農航所) from 2022-01-01 in addition to the NSPO service platform — [Inside 26219](https://www.inside.com.tw/article/26219-buy-formosa-satellite-image)
- FORMOSAT-8 first satellite (齊柏林衛星) launched 2025-11-29; first images released 2026-02-11; native 1 m resolution, 0.7 m after super-resolution; adds red-edge band; formal imaging began Jan 2026; TASA signed an imagery agreement with 看見·齊柏林基金會 in June 2026 (institutional, not a public channel) — [TechNews 2026-10-15](https://technews.tw/2026/10/15/formosat-8-satellite-taiwan/), [Want Rich 2026-02-11](https://wantrich.chinatimes.com/news/20260211900554-420501), [CTEE 2026-06-10](https://www.ctee.com.tw/news/20260610701834-430104) (search snippets)
- FS5 specs: 720 km orbit, 2 m pan / 4 m MS — [NYCU PDF](https://ir.lib.nycu.edu.tw/bitstream/11536/143561/1/12530.pdf) (search snippet)

**National orthophotos — NLSC (內政部國土測繪中心) WMTS — [DOWNLOADABLE NOW as tiles; licence terms unclear]**
- GetCapabilities at https://wmts.nlsc.gov.tw/wmts lists orthophoto layers PHOTO2 正射影像圖(通用), PHOTO1 (92–94年), PHOTO2014…PHOTO2020 (103–109年正射影像圖), PHOTO_VA (蘭嶼 109年, PNG), PHOTO_MIX; all GoogleMapsCompatible / EPSG:3857, JPEG; GetTile via RESTful and KVP; no fee/constraint text in the first 100k characters of the XML — [NLSC WMTS GetCapabilities](https://wmts.nlsc.gov.tw/wmts?SERVICE=WMTS&REQUEST=GetCapabilities)
- Community reports: most 國土測繪圖資服務雲 data require an agency application, but the WMTS endpoints are "免申請服務 API" — [iThome 鐵人賽 2023](https://ithelp.ithome.com.tw/articles/10326699) (blog; not official)
- Licensing caution: OSM Taiwan community in 2017 treated NLSC map data as non-open-licensed — [talk-tw list](https://lists.openstreetmap.org/pipermail/talk-tw/2017-April/001333.html); NLSC sample-data programmes require attribution "內政部國土測繪中心" — [school sample-application PDF](https://www.whjhs.tp.edu.tw/uploads/1762739704218CXppWZ6b.pdf) (search snippet)
- NLSC cadastral WMTS/WMS datasets on data.gov.tw are OGDL v1, free — [data.gov.tw/dataset/17220](https://data.gov.tw/dataset/17220) (search snippet)

**航測及遙測分署 (Aerial Survey and Remote Sensing Branch, Forestry and Nature Conservation Agency, MOA; formerly 農航所) — [REQUEST/PURCHASE]**
- 2012 article: 航遙測圖資供應平台 (ATIS) lets the public locate/view/overlay the latest orthophotos; physical imagery is sold; WMS connection for government agencies (78+ agencies by July 2012); holds aerial photography back to 1976 — [MOA 農政與農情 244](https://www.moa.gov.tw/ws.php?id=2446322)
- Old platform URL http://ngis.afasi.gov.tw launched 2013-05-16; three application channels (in person, mail, online) — [Forestry Bureau file](https://www.forest.gov.tw/file.aspx?fno=36058) (search snippet)
- Output scale: 2023 (112年) 8,978 1:5,000 frames flown; 2022 5,291 orthophoto frames produced — [Forestry file](https://www.forest.gov.tw/File.aspx?fno=87936) and [MOA 2445931](https://www.moa.gov.tw/ws.php?id=2445931) (search snippets)
- No price list or 2024–2026 open-product page surfaced **[GAP]**

**Sentinel-2 — [DOWNLOADABLE NOW; general knowledge, not re-verified this session]**
- Sentinel-2 L2A over Taiwan is free via the Copernicus Data Space Ecosystem (https://dataspace.copernicus.eu/) and Google Earth Engine; MARIDA/FloatingObjects-style models apply directly. **[UNVERIFIED in this session — standard public access]**
- Hyperion (EO-1) archive 2001–2017, 30 m, is on GEE and includes some Taiwan scenes — [GEE catalog](https://developers.google.com/earth-engine/datasets/catalog/EO1_HYPERION?hl=zh-tw) (search snippet)

**NCHC / Taiwan AI data portals**
- scidm.nchc.org.tw hosts agricultural hyperspectral/SERS datasets from NCHU Big Data Center (see KQ5) and the TASA organisation; many are "需申請/Need authorization agreement" — [scidm dataset list](https://scidm.nchc.org.tw/dataset?tags=seed)
- Civil IoT Taiwan (ci.taiwan.gov.tw) fetch failed (expired TLS certificate on 2026-11-06); it is a sensor (air/water/quake) platform, not imagery **[UNVERIFIED]**

**Airborne hyperspectral**
- NCKU thesis used CASI-1500 airborne hyperspectral (<3.5 nm spectral, 25 cm–1.5 m spatial) to map 小花蔓澤蘭; data obtained per project, not open — [NCKU thesis](https://thesis.lib.ncku.edu.tw/thesis/detail/6f5cab1a02361149e14a03df22e8bacd)
- NCHU smart-agriculture centre and NARLabs TIRI discussed a hyperspectral resource-sharing platform (agri/forest/ocean/biomed) — meeting note, not a data portal — [NCHU PDF](https://research.nchu.edu.tw/upfile/web/15興大智慧農業研發中心辦理國研院儀科中心媒合會 共創資源共享交流平台.pdf)

### Inferences
- For a Colab-scale student project the realistic imagery stack is Sentinel-2 (coastal floating debris, 10 m) + NLSC WMTS orthophotos (sub-metre, for beach/dump chips at SOW hotspot or 變異點 coordinates). FORMOSAT-5 (2 m pan) is worth one application to scidm/TASA for a handful of coastal scenes but should not be on the critical path.
- NLSC tile use for an internal academic evaluation is low-risk, but publication/redistribution of tiles needs an explicit licence check with NLSC.

### Gaps
- Current (2026) FORMOSAT academic terms (quota, turnaround, whether FS8 is included) could not be read from TASA pages (empty on fetch).
- ATIS current URL, open products and price list not found.
- NLSC WMTS licence/attribution text not located in the capabilities XML (truncated) or an official page.

## Key question 5 — Are there Taiwan spectral (Raman / NIR / hyperspectral) datasets at all?

### Takeaway
Almost none that are open. The only Taiwan-origin spectral datasets found are (1) NCHU Big Data Center SERS/Raman datasets of pesticides and food dyes on the NCHC scidm platform (CSV wavenumber/intensity with supervised labels, train/valid/test split) — **application required**; (2) NCHU VNIR/NIR/hyperspectral rice-disease image datasets on scidm — **authorization agreement required**; (3) OCA's open CSV of FTIR polymer-identification *results* (match score + polymer name, no spectra) for 2021–2023 microplastic samples. No Taiwan Raman/NIR dataset on Zenodo, Figshare, Kaggle or GitHub was found, and no Taiwan remote-sensing hyperspectral dataset is open. After searching in both languages, I state explicitly: **no openly downloadable Taiwan Raman or NIR spectral library exists as of 2026-11-06.**

### Cited Findings
- 表面增強拉曼(SERS)數據集-北勢育苗場: pesticide SERS spectra from Beishi nursery, multiple concentrations and light sources, nanostructured chip enhancement; CSV rows = wavenumber, intensity; files X_train/X_valid/X_test/y_train/y_valid/y_test; organisation 國立中興大學大數據中心, maintainer 陳炫諭; "Application Required"; created/updated 2025-09-03; 0 downloads — [ark.nchc.org.tw mirror](https://ark.nchc.org.tw/en/dataset/smarter_smds_sers01_0_sers_20220902_spec_000002) (scidm original: https://scidm.nchc.org.tw/zh_TW/dataset/smarter_smds_sers01_0_sers_20220902_spec_000002, timed out)
- Related NCHC SERS/Raman sets (pesticides, 蘇丹紅 Sudan dyes; CSV wavenumber/intensity with labels; application required): [sers04 sads 20240711](https://scidm.nchc.org.tw/ja/dataset/smarter_smds_sers04_0_sads_20240711_spec_000004), [raman_1_pest_20240816](https://scidm.nchc.org.tw/zh_TW/dataset/smarter_smds_raman_1_pest_20240816_pest_000001) (search snippets; pages timed out) **[REQUEST/APPLICATION]**
- NCHU hyperspectral/NIR datasets on scidm (all "Need authorization agreement"): 水稻病害高光譜影像資料集 470–900 nm, imec Snapscan VNIR, >150 bands — [scidm best_wish96457](https://scidm.nchc.org.tw/en/dataset/best_wish96457); 水稻病害近紅外光影像資料集 900–1700 nm with XML annotations — [scidm nir123](https://scidm.nchc.org.tw/en/dataset/smarter_06_group3_0_seed_20230308_7a11_nir123); 未浸泡稻種病害 VNIR 400–1000 nm — [scidm bb150u](https://scidm.nchc.org.tw/en/dataset/smarter_06_group3_0_seed_20230314_7a11_bb150u); 作物病害線蟲高光譜預警模型 490–900 nm — [scidm hype](https://scidm.nchc.org.tw/zh_TW/dataset/smarter_06_2022_h_1_hype_20230315_h0d1_000001); mirrored at NCHU AI-Data — [aidata.nchu.edu.tw](https://aidata.nchu.edu.tw/smarter/en/dataset/?tags=seed)
- OCA 海域塑膠微粒及人造物傅立葉紅外線光譜定性調查: research-vessel surveys 2021–2023 (110–112年), two high-concentration sites each in east/west/north waters; fields = particle ID, cruise, station, lon/lat, water layer, 光譜比對分數 (match score), database material name, category 1/2, sea area, season; CSV https://iocean.oca.gov.tw/oca_datahub/WebService/GetData.ashx?id=f47b6377-7275-4695-85be-f3633929b980 and JSON …id=ba5cfd81-54c2-45eb-bbdd-267ba05c907c; OGDL v1; published 2024-11-18; raw spectra not described — [data.gov.tw/dataset/172479](https://data.gov.tw/dataset/172479) **[DOWNLOADABLE NOW — results, not spectra]**
- A scidm full-text search for 拉曼 returned 55 hits, of which only one title literally contained 拉曼 (a Forestry trail named 北得拉曼), i.e. no additional Raman datasets are indexed there — [scidm search](https://scidm.nchc.org.tw/zh_TW/dataset?q=拉曼)
- Time-resolved LC-Raman amino-acid spectra from "Lo, Hiramatsu lab, NCTU Taiwan" are listed on IEEE DataPort — [IEEE DataPort](https://ieee-dataport.org/documents/raman-spectroscopy-multiple) (search snippet) **[UNVERIFIED; licence/size unknown]**
- International aggregators that could be filtered for Taiwan affiliations: RamanBench (74 datasets, 325,668 spectra; arXiv 2605.02003, NeurIPS 2026) — [arXiv](https://arxiv.org/abs/2605.02003); raman-data PyPI package (93 datasets from Kaggle/HF/Zenodo/Figshare/Mendeley/GitHub) — [PyPI](https://pypi.org/project/raman-data/1.6.5/); Raman DL benchmark with code/weights (arXiv 2601.16107) — [arXiv](https://arxiv.org/pdf/2601.16107). None of the search results identified a Taiwan-origin member.
- NTUST thesis on Raman imaging to distinguish polystyrene vs melamine microplastics plans future ML; no data release — [NTUST ETD](https://etheses.lib.ntust.edu.tw/thesis/detail/61f65fa7e9d07e693e8cbc23e69e6467/?seq=3) (search snippet)
- Searches for Taiwan Raman/NIR on Zenodo/Figshare/GitHub (EN and 繁中, with NTU/NTHU/NCKU/NYCU terms) returned only non-Taiwan records (e.g. WHU-OHS China; Zenodo exosome Raman) — [Zenodo 7258035](https://zenodo.org/record/7258035), [Zenodo 7011379](https://zenodo.org/record/7011379)

**Instruments / vendors for collecting new Taiwan spectra**
- 台灣超微光學 (OtO Photonics): Taiwanese spectrometer maker, ~18 years old, >60% domestic market share, customised products competing with Ocean Optics — [CIO Taiwan](https://www.cio.com.tw/instrument-equipment-taiwans-new-blue-sea/) (does not confirm handheld Raman/NIR line)
- Metrohm Taiwan sells handheld Raman and NIR instruments (Swiss vendor with local site) — [Metrohm zh_tw](https://www.metrohm.com/zh_tw/products/raman-spectroscopy.html)
- NTU 光電工坊 (Photonic Workshop) builds custom laser/Raman spectroscopy systems (lab, not handheld) — [NTU CCMS PDF](https://www.ntu-ccms.ntu.edu.tw/photonic_workshop/001/Upload/1625/relfile/111048/222488/dda76e5e-2aa7-4f05-bdc9-e0c010fd4cba.pdf)
- No Taiwanese manufacturer of handheld Raman/NIR units was confirmed **[GAP]**

### Inferences
- For task (2) the student should plan on foreign open spectra (RRUFF, RamanSPy bundles, RamanBench) for training and, if a Taiwan angle is required, (a) apply for the NCHU SERS pesticide sets via scidm (CSV, already split — minimal preprocessing), or (b) use the OCA FTIR results table as a Taiwan class-distribution prior / evaluation of polymer-class frequency, not as spectra.
- Collecting a small Taiwan Raman set of beach-litter polymers (PET, PE, PP, PS, nylon from the MDImageNet classes) with a borrowed handheld unit would be the only way to get Taiwan-origin spectra for evaluation.

### Gaps
- Application terms, licence and sizes of the NCHU SERS datasets (pages timed out; mirror shows "Application Required" and 0 downloads).
- Whether OCA retains raw FTIR spectra (could be requested from the listed contact "Weng Li-nan", 07-3382057 ext. 262118).
- No Taiwan Raman-ML paper with released code/data was found in either language.

## Key question 6 — Published Taiwan studies on marine/river/beach litter detection, landfill detection and Raman-ML, with released data/code

### Takeaway
Taiwan's deep-learning litter work is almost entirely UAV/ground imagery with YOLO (NTOU, NCU, NCKU); only NTOU's HAIDA has public data/code. No Taiwan Sentinel-2 marine-debris paper, no Taiwan deep-learning landfill/illegal-dump paper, and no Taiwan Raman-ML paper with released data were found.

### Cited Findings
- NTOU (Liao & Juang 2023, Aerospace 10(1):84, DOI 10.3390/aerospace10010084): HAIDA dataset + YOLOv4 detector; YOLOv3/v4/Scaled-v4 comparison with k-fold CV; UAV coastal deployment; code/data on GitHub (MIT) — [NTOU scholars](https://scholars.ntou.edu.tw/handle/123456789/23732); [GitHub](https://github.com/LiaoSteve/HAIDA-Trash-Dataset-High-Resolution-Aerial-image)
- NCU (Drones 6(12):401, 2022): bottle marine debris on a sandy Taoyuan beach; UAV at 12 flight heights giving 0.12–1.54 cm/pixel; YOLOv2; augmentation + background removal raised training mean IoU to 0.81 — [MDPI Drones](https://www.mdpi.com/2504-446X/6/12/401) (search snippet; MDPI returned 403), [NCU scholars listing](https://scholars.ncu.edu.tw/en/publications/detection-of-bottle-marine-debris-using-unmanned-aerial-vehicles-/) (DNS failed) **[data availability unknown]**
- NCKU master thesis 2024 (洪昱翔, advisor 董東璟, 水利及海洋工程學系) "海洋廢棄物辨識AI模型之研發與測試": YOLOv7/YOLOv8, SAM-based cut-and-paste synthetic augmentation onto own beach photos; toothbrush 99.5%, PET bottle 96.5% recognition; full text public, image data not stated as public — [NCKU ETD](https://thesis.lib.ncku.edu.tw/thesis/detail/f8724a49b360c453ff2a52b6c69e9ed0)
- NTOU (Machines 11(9):876, 2023): SIFT stitching of multi-UAV images + YOLO variants to identify riverbank garbage (Keelung area) — [MDPI Machines](https://www.mdpi.com/2075-1702/11/9/876) (search snippet); follow-up on stitching optimisation — [IJOCTA](https://accscience.com/journal/IJOCTA/articles/online_first/8192) (search snippet)
- NTOU project "Traceability of Marine Debris and Design of Detection System" (deep learning, pattern recognition, GNSS; not satellite) — [NTOU CRIS](https://scholars.ntou.edu.tw/cris/project/pj07758?locale=en)
- NTOU 2019 ISPRS Archives paper on UAV marine-debris detection (Taiwan) — [ISPRS Archives XLII-3/W8](https://isprs-archives.copernicus.org/articles/XLII-3-W8/55/2019/) (search listing only) **[UNVERIFIED content]**
- SOW + academics, Marine Pollution Bulletin 2018 (541 cleanups 2004–2016) — [CNA](https://www.cna.com.tw/news/ahel/201810030133.aspx)
- Government operational AI: EMA drone programme YOLOv4→YOLOv9 with vendor 環輿科技 (see KQ1-A6) — [CIE 工程 Dec 2025 PDF](https://www.cie.org.tw/var/file/0/1000/img/2/708585316.pdf) (partially legible)
- Reference non-Taiwan satellite work the student already cites, for completeness: Sentinel-2 marine debris detector with released weights (Rußwurm et al. 2023) — [arXiv 2307.02465](https://arxiv.org/html/2307.02465); FloatingObjects (ISPRS Annals 2021) — [ISPRS Annals V-3-2021](https://isprs-annals.copernicus.org/articles/V-3-2021/285/2021/)
- 2019 Presidential Hackathon team 海波浪 (marine debris topic) presentation exists — [PDF](https://presidential-hackathon.taiwan.gov.tw/history/2019/files/7.第七組_海波浪_複選會議簡報.pdf) **[not read]**; 2022 海湧工作室 entry — [PDF](https://presidential-hackathon.taiwan.gov.tw/outlink/history/2022/upload/files/20220510海湧工作室.pdf) **[not read]**

### Inferences
- A defensible student design that reuses the prior project's pattern: train/fine-tune on MARIDA + FloatingObjects (Sentinel-2) → run over Taiwan coastal Sentinel-2 tiles → aggregate detections per SOW 10-km segment → compare to SOW 2018/2022 bags-per-100-m rankings; in parallel, fine-tune a YOLO on MDImageNet/HAIDA for a drone-scale demo. For land dumping, apply an AerialWaste-trained classifier to NLSC orthophoto chips at 國土利用監測 waste-type variation points.

### Gaps
- No Taiwan satellite-based coastal litter study found (EN/繁中; NDLTD not directly searchable here).
- No Taiwan river-CCTV litter AI paper found; only the WRA 第四河川局 news item.
- No Taiwan Raman-ML paper with code/data found.
- HAIDA image count/classes and NCU dataset availability remain unverified because MDPI blocked fetches.

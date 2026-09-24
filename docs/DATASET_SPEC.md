# TRUEWATCH — dataset specification

Problem statement **SIH26187** · Phase 1 · design and audit only. No implementation in this
document.

**Authority.** `SIH26187_TRUEWATCH_final-1.pdf` is the source of truth for every committed
capability and target. `docs/MEASUREMENTS.md` is the source of truth for every number described
as measured. `docs/PHASE_MINUS1_SCOPE.md` fixes scope. `docs/ARCHITECTURE_V2.md` fixes the
runtime this data feeds. Where a dataset fact could not be verified from a primary source it is
recorded in **§10 Open questions** and is never stated as fact elsewhere in this file.

**Source change, 2026-09.** KAIST Multispectral Pedestrian is **replaced by Teledyne FLIR ADAS
v2**. The Hugging Face KAIST mirror carries images for `set00` and `set05` only and no annotation
file at all, and the official KAIST label archives sit behind dead links, so no labelled KAIST
corpus can be built. FLIR ADAS v2 supplies what KAIST was chosen for (visible and thermal frames,
day and night, people) and adds vehicles. All three enabled sources are read from third-party
Kaggle mirrors of the original releases (§7). `02_convert_kaist.py` stays in the repository,
collision-safe, with the source `enabled: false`. Where this document still describes KAIST it
says so explicitly.

**What this phase decides.** One detector — YOLO11-s — is the only model in the stack that is
fine-tuned (`ARCHITECTURE_V2` §1.3, slide 3). Everything else ships pretrained. The two headline
numbers the deck carries, *person mAP@50 0.85 day / 0.75 infrared* and *under 5 false alerts per
camera per day* (slide 5, both TARGETS per `PHASE_MINUS1_SCOPE` §5.2), are decided by the corpus
assembled here far more than by anything Phase 2 does with it. A detector trained on clean
daylight road scenes collapses on infrared, on far-field objects below the measured 19 px cliff,
and on the empty frames that make up most of a border camera's day.

---

## 1. Label schema

### 1.1 The unified taxonomy

Five classes. The list is fixed by capability 1 (person) and capability 2 (*"truck, car,
two-wheeler, cart"*, slide 2, `PHASE_MINUS1_SCOPE` §4 row 2).

| id | name | definition |
|---|---|---|
| 0 | `person` | One human body, standing, walking, running, sitting or lying, including a human riding or pushing any vehicle. The box bounds the human only where the annotation allows it, and the human-plus-machine where the source annotates them jointly (see §1.6). |
| 1 | `two_wheeler` | A two- or three-wheeled road vehicle: motorcycle, scooter, moped, bicycle, and the three-wheeled autorickshaw. Box bounds the machine, not its rider. |
| 2 | `car` | A passenger road vehicle up to and including a van or minibus: car, jeep, SUV, van, and the small-bus body an Indian road scene calls a tempo or minibus. |
| 3 | `truck` | A goods or heavy passenger vehicle on four or more wheels: lorry, tipper, tanker, full-size bus, caravan and trailer. |
| 4 | `cart` | A non-motorised load-carrying vehicle: animal-drawn cart, hand cart, rickshaw pulled or pedalled as a load carrier. **Conditional — see §1.5.** |

The names are the strings `ARCHITECTURE_V2` §4.2 already freezes on the wire:
`'person' | 'truck' | 'car' | 'two_wheeler' | 'cart'`. The schema must not drift from that list,
because `backend/src/services/ingest.service.js` maps those exact strings.

### 1.2 IDD Detection — all 15 native classes mapped

IDD Detection ships Pascal-VOC XML annotations over 46,588 images
(`MEASUREMENTS.md`, 31,569 / 10,225 / 4,794; the three list files of the Kaggle mirror
`vinayak21574/idd-detection` hold exactly these counts, verified 2026-09). Layout:
`Annotations/<subset>/<drive>/<frame>.xml` beside `JPEGImages/<subset>/<drive>/<frame>.jpg`, with
`train.txt` / `val.txt` / `test.txt` naming `<subset>/<drive>/<frame>`; the six subsets are
`frontFar`, `frontNear`, `highquality_16k`, `rearNear`, `sideLeft`, `sideRight`. Frame names
repeat across drives (2,694 of them), so a frame is identified by its full path, never its stem.
The mirror also carries editor swap files (`.001542_r.xml.swp`); only `*.xml` is read and
everything else under `Annotations/` is counted. Its detection taxonomy has 15 classes. Every one is
accounted for below; none is left to a default.

| # | IDD native class | maps to | why |
|---|---|---|---|
| 1 | `car` | **2 `car`** | direct |
| 2 | `bus` | **3 `truck`** | merged. A bus is a heavy multi-axle vehicle and behaves like a truck for every rule in `edge/rules/`: same size band, same lane occupancy, same fence-crossing footprint. The deck's four vehicle classes contain no bus, so the honest choice is to merge rather than to invent a sixth class the PDF does not commit to. |
| 3 | `truck` | **3 `truck`** | direct |
| 4 | `motorcycle` | **1 `two_wheeler`** | direct |
| 5 | `bicycle` | **1 `two_wheeler`** | merged. Slide 2 says "two-wheeler", not "motorcycle"; a bicycle is a two-wheeler. Separating them would create a class the wire format cannot express. |
| 6 | `autorickshaw` | **1 `two_wheeler`** | merged. It is three-wheeled, so the name is imperfect, but it is a small, light, single-driver vehicle whose size and speed band match a motorcycle far better than a car. See §1.5 — it is explicitly **not** the cart class. |
| 7 | `person` | **0 `person`** | direct |
| 8 | `rider` | **0 `person`** | merged. IDD annotates the human on a two-wheeler separately from the machine; both boxes are kept, the human as `person`, the machine as `two_wheeler`. This is the behaviour capability 1 needs — a rider is a person to be detected and tracked. |
| 9 | `animal` | **DROPPED** | No class in the deck. Kept out of the label set but **not** discarded from the image: frames containing animals go into the hard-negative pool (§5), because a cow on a border road at night is exactly the false positive the 5-alerts-per-camera target must survive. |
| 10 | `traffic light` | **DROPPED** | Static infrastructure. Not a capability. |
| 11 | `traffic sign` | **DROPPED** | Static infrastructure. Not a capability. |
| 12 | `caravan` | **3 `truck`** | merged. Large towed/self-propelled body, truck size band. Rare in IDD; merging avoids a class with a handful of instances. |
| 13 | `trailer` | **3 `truck`** | merged, same reasoning as `caravan`. |
| 14 | `train` | **DROPPED** | Rail, not road. No border-post camera in scope watches a track (`PHASE_MINUS1_SCOPE` §4). |
| 15 | `vehicle fallback` | **CONDITIONAL — see §1.5** | IDD's explicit catch-all for vehicles outside its other fourteen classes: tractors, construction vehicles, animal carts and hand carts all land here. It is the only place in any of the three sources where a cart can be. |

Dropped classes never become background silently. `01_convert_idd.py` counts every dropped
instance by native class name and prints the tally, so the drop is auditable.

### 1.3 Teledyne FLIR ADAS v2 — every occurring class mapped

Kaggle mirror `samdazel/teledyne-flir-adas-thermal-dataset-v2`, original layout: six
directories, each `<subset>/coco.json` plus `<subset>/data/*.jpg`, frames named
`video-<videoId>-frame-<n>-<hash>.jpg` (file names unique across the release).

| subset | modality | images | official split |
|---|---|---|---|
| `images_rgb_train` | visible | 10,318 | train |
| `images_rgb_val` | visible | 1,085 | val |
| `images_thermal_train` | LWIR | 10,742 | train |
| `images_thermal_val` | LWIR | 1,144 | val |
| `video_rgb_test` | visible | 3,749 | test |
| `video_thermal_test` | LWIR | 3,749 | test |

Counts read from the mirror's `coco.json` files, 2026-09. RGB and thermal frames carry
different videoIds and are not pixel-registered; thermal frames are 640×512, RGB frames range
from 1024×768 to 2048×1536. Train RGB frames are tagged day / night / dawn-dusk in
`extra_info.hours`, which the converter keeps as `lighting`.

Each `coco.json` declares 80 COCO-style category names; 16 actually occur. Every occurring name
is mapped or dropped below; a name that occurs and is in neither column fails the converter.

| FLIR native class | maps to | why |
|---|---|---|
| `person` | **0 `person`** | direct |
| `bike`, `motor`, `scooter` | **1 `two_wheeler`** | bicycle, motorcycle and scooter are all two-wheelers in the §1.1 sense. `scooter` is rare (15-41 instances per subset); its exact meaning (kick scooter or motor scooter) is **OQ-14** and does not change the mapping. |
| `car` | **2 `car`** | direct |
| `truck`, `bus` | **3 `truck`** | merged, the same reasoning as IDD `bus` (§1.2) |
| `other vehicle` | **DROPPED, as an unlabelled object** | FLIR's catch-all for vehicles outside the list (trailers, construction and farm machines). Merging it into `truck` corrupts the truck size prior exactly as IDD `vehicle fallback` would (§1.5). Its boxes are kept on the record: a frame holding one is never a negative, and no negative tile may touch one. 698-1,647 instances per subset. |
| `light`, `sign`, `hydrant` | **DROPPED** | static infrastructure |
| `train` | **DROPPED, unlabelled object** | rail, 5-9 instances |
| `dog`, `deer` | **DROPPED, unlabelled object** | animals; an animal-only frame is an `animal` hard negative (§5) |
| `skateboard`, `stroller` | **DROPPED, unlabelled object** | the person using one is annotated separately as `person` |

Every dropped instance is counted by native name in `reports/convert_flir.json`.

### 1.3a KAIST Multispectral Pedestrian (disabled)

Kept for reference and for `02_convert_kaist.py`, which remains runnable. KAIST annotates
pedestrians only, in its own text format, with a small tag vocabulary. Frame names
(`I00000.jpg`) restart in every set and video; the converter indexes annotations by
set/video/frame and stages frames as `set_video_frame`, so no two videos can collide.

| KAIST native tag | maps to | why |
|---|---|---|
| `person` | **0 `person`** | direct |
| `people` | **IGNORE REGION** | A group box around several unseparated humans. Training a single `person` box on a crowd teaches the detector to merge people, which destroys tracking and therefore capabilities 1, 5 and 6. The box becomes an ignore region: no positive label, and the frame is **not** usable as a negative. |
| `cyclist` | **0 `person`** | merged. The KAIST box encloses the human on the bicycle. The human dominates the box and the dataset carries no separate bicycle box to emit as class 1. Recorded as a known taxonomy impurity in §7 biases. |
| `person?` | **IGNORE REGION** | KAIST's own uncertainty marker. Treating an uncertain box as a positive injects label noise into the exact class the 0.85 / 0.75 targets are measured on. |

**Ignore-region handling.** YOLO has no native ignore semantics. Two options exist: drop the whole
frame, or keep the frame and zero-fill the ignore boxes. The decision is **keep and zero-fill** for
`people`, **drop the frame** when a `person?` box overlaps no other annotation. Zero-filling a crowd
region removes the ambiguous pixels without losing the clean `person` boxes elsewhere in the same
frame, and the fill is recorded per frame in the conversion manifest.

### 1.4 LLVIP — all native classes mapped

Kaggle mirror `afradhossain/llvip-dataset`, identical in layout to the upstream `LLVIP.zip`:
`Annotations/<id>.xml`, `visible/{train,test}/<id>.jpg`, `infrared/{train,test}/<id>.jpg`,
12,025 train and 3,463 test pairs (the upstream README's figures, confirmed against the files).

| LLVIP native class | maps to | why |
|---|---|---|
| `person` (the only annotated class) | **0 `person`** | direct |

LLVIP annotates pedestrians in strictly registered visible/infrared pairs. Nothing is dropped.
Because it is single-class, LLVIP frames containing vehicles carry **unlabelled vehicles**. This is
a real hazard: an unlabelled car is a teaching signal that cars are background. Mitigation in §3.4.

### 1.5 The cart problem, stated honestly

**None of the three sources has a `cart` class.** IDD's `autorickshaw` is a motorised three-wheeled
passenger vehicle and is *not* a cart; calling it one to fill the slot would be dishonest and would
make capability 2's confusion matrix meaningless. IDD's `vehicle fallback` is a heterogeneous
catch-all that *contains* animal carts and hand carts among tractors and construction vehicles, but
the label does not tell you which is which.

The decision, with a numeric rule so it cannot be fudged later:

1. `01_convert_idd.py` extracts every `vehicle fallback` instance to a review sheet with its crop
   path, source image and box.
2. A manual pass — budgeted at 45 minutes, the only manual labelling in this phase — marks each
   crop as `cart` or `not_cart`. Only `cart` becomes class 4. Everything marked `not_cart` is
   **dropped**, not merged into `truck`, because a tractor in the truck class corrupts the truck
   size prior.
3. **Gate:** if the hand-verified cart count is **>= 300 instances**, class 4 ships and its AP is
   reported with the caveat "trained on N hand-verified instances". If it is **< 300**, class 4 is
   **withdrawn from the shipped schema**, capability 2 ships as three vehicle classes, and the
   withdrawal is written into `docs/DATASET_CARD.md` and reported to the panel as a scope
   reduction rather than hidden behind a class with 40 examples and an AP of 0.03.

The schema keeps id 4 reserved either way, so the wire format in `ARCHITECTURE_V2` §4.2 does not
change under either outcome.

**As implemented.** `01_convert_idd.py` writes `processed/cart_review/vehicle_fallback.csv`
(`uid,image,xmin,ymin,xmax,ymax,verdict`, every row `UNVERIFIED`) and carries any verdict already
written over to the next run, so a re-run can never erase a review. `09_build_yolo_ds.py`
evaluates the gate: at >= 300 rows marked `cart` it adds those boxes as class 4; below that it
writes no class-4 label, records the decision in `manifests/cart_gate.json`, lists `cart` in
`manifests/class_exceptions.txt` (read by gate G11) and states the withdrawal in `data.yaml`,
which keeps `nc: 5` and the name `cart` at id 4. **Current state: the review has not been done,
so the Kaggle build withdraws class 4.** `vehicle fallback` boxes stay recorded as unlabelled
objects, which keeps those frames out of the negative pool. This closes `PHASE_MINUS1_SCOPE` open question 1 with a decision
procedure rather than a guess.

### 1.6 Box convention

- Format: Ultralytics YOLO txt, one row per object, `class_id cx cy w h`, all four geometry values
  normalised to `[0, 1]` against the image the label sits beside.
- Rider and machine are two boxes, never one, wherever the source separates them (IDD). Where the
  source does not (KAIST `cyclist`), the joint box is labelled `person`.
- Boxes are clipped to the image, not dropped, when the source extends past the edge. A box whose
  clipped area falls below **16 px²** is dropped and counted.

---

## 2. Split policy

Three splits plus one gate set. The policy exists to make two specific failures structurally
impossible: frame-level leakage between splits, and a validation set that does not contain the
night-time data the 0.75 IR target is measured on.

### 2.1 The rule that prevents leakage

**Splitting happens at the sequence level and nowhere else.** Every image entering the corpus is
assigned a `sequence_key` at conversion time, before any split logic runs. The split file maps
`sequence_key -> split`. No code path assigns a split to an image directly.

- FLIR: `sequence_key = "flir/<videoId>"` — the video the frame was cut from.
- LLVIP: `sequence_key = "llvip/<scene_prefix>"` — the first two digits of the id; see §2.3.
- IDD: `sequence_key = "idd/<subset>/<drive_folder>"` — IDD's images are organised under
  per-drive folders, so a drive is the sequence.
- KAIST (disabled): `sequence_key = "kaist/<setNN>/<VNNN>"`.

Because the mapping is `sequence -> split` and never `image -> split`, two frames from one video
cannot land in different splits. `06_split.py` asserts this by grouping the final assignment back
by `sequence_key` and failing if any key resolves to more than one split (§9, gate G3).

### 2.2 FLIR ADAS v2 — the official partition, asserted

| split | FLIR subsets |
|---|---|
| train | `images_rgb_train`, `images_thermal_train` |
| val | `images_rgb_val`, `images_thermal_val` |
| **test (held out)** | `video_rgb_test`, `video_thermal_test`, every **3rd** frame of each video |

`06_split.py` asserts (`assert_flir_videos_disjoint`) that no videoId appears in two official
splits. On the 2026-09 mirror none does (132 + 17 + 8 RGB and 133 + 17 + 8 thermal videos, no
overlap between any two subsets). If one ever does, every frame of that video moves to the
split holding most of its frames (ties: test, then val, then train), and the move is logged
and counted, never silent. The video test set is 8 videos at a high frame rate; keeping every
third frame of each (`splits.yaml flir.test_decimation`) removes near-identical consecutive
frames that would otherwise make test 20% of the corpus. Decimation happens inside one sealed
split and cannot leak; the decimated frames are listed in `manifests/balance_dropped.txt`.

The pairing of an RGB video with its thermal twin is not recorded in the mirror, so the
assertion can only check videoIds within a modality family. That the official partition keeps
twins together is assumed, not verified: **OQ-13**.

### 2.2a KAIST — explicit set assignment (disabled)

KAIST is captured as twelve sets, alternating day and night campus/road/downtown captures.

| split | KAIST sets | lighting |
|---|---|---|
| train | `set00`, `set01`, `set02`, `set03`, `set04`, `set05` | three day, three night |
| val | `set06`, `set09` | one day, one night |
| **test (held out)** | `set07`, `set08`, `set10`, `set11` | two day, two night |

The day/night composition of each set is recorded as **OQ-1** — the assignment above assumes the
conventional set00–02 day / set03–05 night / set06–08 day / set09–11 night structure, and
`02_convert_kaist.py` verifies it at runtime from the LWIR/visible statistics rather than trusting
the assumption, failing loudly if a set does not match its declared lighting.

`MEASUREMENTS.md` §1, §2 and §3 were all produced on `set00/V007`, which is in **train**. That is
deliberate and harmless: every one of those numbers came from a **pretrained, not fine-tuned**
model, so no fine-tuned figure is ever reported on data the fine-tune saw. This is stated in the
dataset card so nobody later mistakes it for a leak.

### 2.3 LLVIP and IDD

- **LLVIP** ships an official train/test partition. The official test partition becomes part of our
  **held-out test** untouched. Our **val** is carved out of the official train partition by scene
  key, never by random frame: scenes are taken in a seeded order until they hold **25%** of the
  official-train frames. OQ-3 is resolved: the two-digit id prefix partitions the corpus into 19
  train scenes (`01`-`18`, `25`) and 7 test scenes (`19`-`24`, `26`), none shared. The
  perceptual-hash fallback (Hamming <= 6 pseudo-sequences) remains in code for a layout without
  usable prefixes.
- **IDD** ships train/val/test, but the public IDD test partition has no released labels. So: IDD
  train -> our **train**; IDD val is split by drive folder, frame-weighted, **60%** into our
  **val** and **40%** into our **test**; IDD test is not used at all. The official lists are
  drive-disjoint (438 train, 119 val, 69 test drives, no drive in two lists).

### 2.4 Target proportions

| split | share of labelled images | purpose |
|---|---|---|
| train | 70% ± 3% | fine-tuning |
| val | 15% ± 3% | epoch selection, early stopping, every intermediate report |
| test | 15% ± 3% | **sealed until Phase 11** |

G14 counts **untiled** images: a far-field tile is a derived view of a train frame, and counting
it would make the proportions a function of the tile budget.

Every split must satisfy, independently: **visible : infrared between 55:45 and 70:30**. A val set
that is 90% daylight cannot measure a 0.75 IR target, and a training set that is half infrared
starves the 0.85 day target. The ratio is checked as gate G8.

### 2.5 The sealed test set

`test` is written once by `06_split.py` and then not read by any script in Phases 2–10. Enforcement
is procedural plus mechanical: `09_build_yolo_ds.py` emits `data.yaml` whose `val:` key points at
the val split and which contains **no `test:` key at all**, so an Ultralytics run in `training/`
cannot address it by accident. The test manifest lives in `datasets/manifests/split_test.txt` with
its sha256 recorded, so Phase 11 can prove the set was not edited after sealing.

### 2.6 The hard set

A **regression gate set of 280 images** (< 300 by design), drawn only from **val**, never from test,
selected by four objective criteria (heights at the 640 px detector input, the scale of
`MEASUREMENTS.md` §3; intensity spread as the raw LWIR standard deviation recorded by
`04_ir_to_3ch.py`; any shortfall is filled from val in a deterministic order):

| bucket | count | selection rule |
|---|---|---|
| tiny objects | 80 | frames whose median person box height is **<= 19 px** — the measured cliff, `MEASUREMENTS.md` §3 |
| heavy infrared | 80 | LWIR frames in the lowest decile of intensity variance (the washed-out thermal frames) |
| crowds | 60 | frames with **>= 8** person instances |
| occlusion | 60 | frames where **>= 30%** of person boxes have IoU > 0.3 with another person box |

The hard set is run as a pass/fail regression in every later phase that touches the detector. Its
member list is frozen in `datasets/manifests/hard_set.txt`; a Kaggle build writes it into its own
output (`manifests/hard_set.txt`), because that is what the training notebook mounts.

### 2.7 Modality balancing — the rule

The real source sizes put the raw corpus above the §2.4 band: IDD alone is 41.8k visible-only
labelled frames against ~31k infrared frames in total (FLIR 15.6k, LLVIP 15.5k), so train would
sit near **0.72** visible and val near **0.72**. The rule, applied by `06_split.py` after the
source assignment and before anything else:

1. Per split, if the visible share exceeds its target (`balance.target_visible_share`: train
   **0.63**, val and test **0.65**), whole **visible-only sequences** of the sources in
   `balance.subsample_sources` (IDD) are dropped in a seeded order until it does not.
2. Infrared is **never** dropped. Paired sources (LLVIP, FLIR) are never thinned, so the
   visible/infrared structure they carry is kept intact. If a split is **below** the band,
   nothing is dropped and gate G8 reports it.
3. Every dropped image is listed, with its split, in `manifests/balance_dropped.txt`.

Train's target is lower than val/test's because §3.5 then adds ~10% negatives to train and
those are predominantly visible (infrared empty frames are scarce); the far-field tiles are
budgeted at 35% LWIR. Projected from the real counts: IDD keeps ~14.3k of 31.6k train frames,
~3.6k of 6.1k val frames and ~4.0k of 4.1k test frames; the final train share lands near
**0.66**, val and test near **0.65**; untiled proportions near **0.70 / 0.14 / 0.16**. The cost
is stated plainly: roughly half of IDD, the only Indian-road and the most vehicle-rich source,
is not used. That is the price of the §2.4 band, and the band exists because a val set that is
mostly daylight cannot measure the 0.75 infrared target.

---

## 3. Imbalance and failure analysis

### 3.1 Class frequency

**Prediction.** `person` dominates. LLVIP is a pedestrian-only corpus (~31k frames); IDD and FLIR
are the vehicle sources, FLIR mostly `car` (~70k instances per modality in train). Expected instance
share, order of magnitude: `person` 55–70%, `car` 12–20%, `two_wheeler` 10–18%, `truck` 4–8%,
`cart` well under 1%.

**Mitigation.** Not oversampling — oversampling a rare class in a detector mostly multiplies its
backgrounds. Instead:
1. Class-balanced **image** sampling: images containing `truck` or `cart` enter the epoch list
   twice, capped so no image is seen more than twice per epoch.
2. Per-class AP is reported separately at every checkpoint. A single mAP number hides a dead class.
3. The §1.5 gate: a class that cannot reach 300 instances is withdrawn rather than shipped broken.
4. Gate G11 fails the build if any **shipped** class holds < 1% of instances without a written
   decision recorded in the dataset card.

### 3.2 Object size

**Prediction.** Bimodal. IDD is dashcam-framed, so its vehicles are large; LLVIP pedestrians
are mid-field with a median person height near 77 px (`MEASUREMENTS.md` §3). The far-field tail
below 19 px — the band that actually matters at a border post at 150 m — is **under-represented in
all three sources**, because none of them is a pole-mounted long-range camera.

**Mitigation: SAHI tiling of the far-field strip.** This is §3.3.

### 3.3 Small-object strategy, justified against the measured cliff

`MEASUREMENTS.md` §3 is the authority: recall is 100% at 38 px and 27 px, **90% at 19 px, 80% at
14 px, 49% at 9 px**. The cliff is between 19 px and 9 px.

The arithmetic that sets the tile size:

- A 1920×1080 frame fed whole to a 640 px detector is downscaled by **3.0×**. A person standing at
  the far field who occupies **27 px** in the native frame arrives at the detector as **9 px** —
  the 49% recall band. The same person is *detectable* at 27 px and *nearly lost* at 9 px, and the
  only difference is the downscale.
- Tiling the far-field strip into **640×640 tiles at native resolution** removes the downscale
  entirely. That 27 px person arrives as 27 px — the **100% recall** band. A 19 px person arrives
  as 19 px (90%) instead of 6 px (below anything measured).

**Parameters.**

| parameter | value | justification |
|---|---|---|
| tile size | **640 × 640 px** | matches the detector's native input, so the tile is fed at 1:1 with no resampling — the whole point of the exercise |
| overlap ratio | **0.20** (128 px) | an object is only guaranteed whole in some tile if the overlap exceeds its largest dimension. Far-field targets at a border post are the small tail: a person at 19–27 px, a truck at well under 128 px. 128 px covers every far-field object with margin. Larger overlap costs tiles quadratically for no recall. |
| strip definition | the **upper 40%** of frame height by default, per-camera overridable | the far field of a pole-mounted camera is the upper band. Tiling the whole frame triples inference cost to re-detect objects that are already large enough. |
| tiles per frame | 3 across × 1 down at 1920×1080 with 0.2 overlap on a 432 px strip | 3 extra forward passes, not 9 |
| minimum box area kept in a tile | **30%** of the original box area | a box sliced by a tile edge is kept only if most of it survives, otherwise the model learns to fire on body fragments |

**Tile budget, as implemented.** Every train frame at least 640 px on both sides yields candidate
windows; a window is kept when it holds at least one box surviving the 30% rule. From those,
`_lib.plan_object_tiles` takes **6,000** (`composition.tiles.max_tiles`, gate G13 needs 2,000),
**35% LWIR** (LLVIP's 1280×1024 infrared frames are the only LWIR large enough; FLIR thermal is
640×512), windows holding an object of <= 27 native px first. The plan is deterministic for a
seed, and 07 sizes the negative pool against the same plan.

The same tiling is applied at **training** time (as extra training images, §8 of the pipeline) and
at **inference** time (`edge/pipeline/tiling.py`, `ARCHITECTURE_V2` §5, Phase 6). Training on
untiled data and inferring on tiles is a train/test mismatch; this spec refuses it.

### 3.4 Day/night ratio and infrared handling

**Prediction.** IDD is daylight-only. FLIR RGB train is ~60% day, ~35% night; FLIR thermal is
day and night. LLVIP is predominantly night. Combined, the raw corpus lands near **70:30**
visible:infrared, on the edge of the §2.4 band, so §2.7 balances it by rule and G8 asserts the
result.

**How LWIR becomes a 3-channel tensor.** Slide 4 commits to *IR replicated to 3 channels*. Exactly:

1. Read the LWIR frame as single-channel 8-bit (`cv2.IMREAD_UNCHANGED`, then a 16→8-bit linear
   rescale if the source is 16-bit, recording which it was).
2. Apply CLAHE, clip limit 2.0, tile grid 8×8. Thermal frames are low-contrast; the detector's
   pretrained filters expect structure.
3. **Replicate the single channel across all three** — `np.repeat(ch[..., None], 3, axis=2)` — so
   B = G = R. No false colour. A thermal colormap invents chromatic structure the sensor never
   measured, and `MEASUREMENTS.md` §1 records what false-colour thermal costs: **AP@50 = 0.133**,
   precision 0.364, recall 0.093. That number is the reason this pipeline replicates instead of
   colourising.
4. Write the 3-channel result into the dataset. The conversion runs at write time
   (`_lib.load_for_output`, used by 08 for tiles and 09 for frames); no 3-channel intermediate is
   stored, because ~31k lossless 3-channel frames would take ~36 GB, more than the whole Kaggle
   output budget. The final file is JPEG q92 like every other image: with R = G = B the encoder's
   chroma is exactly neutral and the three channels decode bit-identical (checked). Gate G10
   verifies the written files: 3 channels **and** B = G = R.

**One model or two heads — the decision.** **One model, one head, trained on mixed visible and
infrared.** Five lines of defence:

1. The edge runs **one** ONNX graph (`ARCHITECTURE_V2` §1.1); two heads means two models resident
   in a free-CPU-tier Space, doubling the memory the demo cannot spare.
2. A two-head design needs a day/night classifier to route frames, and that router's errors become
   detector errors at exactly the dusk hours a border post cares about most.
3. `MEASUREMENTS.md` §1 measured **89% IR recovery from a COCO-pretrained visible-only model** —
   the shared representation already transfers, so the two domains are not as disjoint as they look.
4. One model sees 100% of the corpus; two heads each see roughly half, and the smaller of the two
   halves is the infrared one carrying the harder 0.75 target.
5. Kaggle gives 30 GPU-hours a week for the whole project. One fine-tune fits; two do not.

### 3.5 Empty frames — the single biggest lever on the false-alert target

**Prediction.** The three sources are all object-centric: annotators pointed cameras at scenes with
people and vehicles in them. Expected images-with-zero-objects in the raw corpus: **under 3%**.

A real border camera is the opposite. Most of its day is an empty road, moving foliage, rain, and
headlight glare. A detector that has never been shown "nothing is here" has no way to represent it,
and every one of those frames becomes a low-confidence false positive. **Slide 5's "under 5 false
alerts per camera per day" is a precision target, and precision on a 24-hour stream is set by how
the model behaves on the 23 hours that contain nothing.**

**The number: 10% of training images must contain zero objects.** This follows the Ultralytics
guidance for background images and is bounded on both sides — below roughly 5% the effect on
false-positive rate is negligible; above roughly 15% the model starts under-detecting and the
recall side of the 0.85 target suffers. 10% sits in the middle of that band, is checked by gate G10
at ±2%, and is a tunable the Phase 2 ablation can move once there is a measured false-positive
curve to move it against.

Negatives appear in **train only**. Val and test keep their natural composition, so val metrics stay
comparable to published numbers on these datasets.

**As implemented.** "Training images" includes the far-field tiles, so `07_negatives.py` sizes the
pool against train positives **plus** the tile plan (the same deterministic plan `08` cuts). Train
frames with no objects that are not selected are excluded from the build and counted, so the ratio
is a decision, not an accident of the sources. Whole empty frames are scarce (FLIR train holds
~1.1k, IDD few), so when they run short the pool is filled with **empty far-field tiles**: 640×640
native crops of the upper strip that no annotated box of any native class touches, dropped
classes included, at most two per parent frame. That is exactly what the edge tiler hands the
detector on an empty approach road. LLVIP is never a negative source (unlabelled vehicles).

---

## 4. Augmentation policy

Every entry is justified against a failure mode of a **fixed, pole-mounted, outdoor border camera**.

### 4.1 Always on

| augmentation | parameters | failure mode it addresses |
|---|---|---|
| Mosaic | `mosaic = 1.0`, **disabled for the final 10 epochs** | four images per sample multiplies small-object context and scale diversity. Disabled at the end because mosaic distorts the object-size statistics the model should finish calibrated on. |
| Random scale | `scale = 0.5` (0.5×–1.5×) | one camera sees the same person at 9 px and 200 px depending on range. Scale jitter is the cheapest proxy for range. |
| Translate | `translate = 0.1` | objects appear anywhere in frame; a fixed camera still has wind sway and mount drift. |
| Horizontal flip | `fliplr = 0.5` | a person may walk either way across the frame. Direction is a *rule* input, not an appearance input, so flipping costs nothing. |
| HSV jitter | `hsv_h = 0.015`, `hsv_s = 0.7`, `hsv_v = 0.4` | sun angle, haze, dust and auto-exposure move colour and brightness hour to hour. Hue is kept tight because large hue shifts are meaningless on a replicated-grey IR frame. |
| Random downscale–upscale | probability 0.25, factor 2–4×, `INTER_AREA` down then `INTER_LINEAR` up | directly synthesises the 19/14/9 px degradation `MEASUREMENTS.md` §3 measured, from the abundant large-object frames. This is the cheapest way to manufacture far-field examples the sources do not contain. |
| JPEG recompression | probability 0.3, quality 40–90 | the stack ingests **RTSP H.264/H.265** (`ARCHITECTURE_V2` §1). Every real frame arrives with compression artefacts; clean PNG training data does not. |
| Motion blur | probability 0.15, kernel 3–9 px, random angle | moving subjects at low shutter speed at dusk. |

### 4.2 Infrared only

| augmentation | parameters | failure mode |
|---|---|---|
| CLAHE | clip 1.5–3.0, grid 8×8, probability 0.5 | thermal contrast varies with ambient temperature; a July night and a January night are different sensors in effect. |
| Gaussian noise | σ 2–8 (8-bit) | uncooled microbolometers are noisy; FLIR and LLVIP are cleaner than a deployed camera. |
| Brightness / contrast jitter | brightness ±0.25, contrast ±0.25 | thermal auto-gain hunts when a hot vehicle enters frame and washes the whole scene. |
| Thermal wash-out simulation | probability 0.2, compress dynamic range to 40–70% of full | reproduces the washed-out mid-afternoon thermal frame where ground and body temperature converge — the hardest IR case, and the one the hard set (§2.6) gates on. |

Polarity inversion (white-hot ↔ black-hot) is **deliberately excluded** and recorded as **OQ-6**:
it is a real deployment variable, but neither FLIR nor LLVIP documents its polarity convention per
sequence, so inverting would train the model on a condition that cannot be verified against a
source.

### 4.3 Forbidden

| forbidden | why |
|---|---|
| Vertical flip (`flipud`) | a pole-mounted camera never sees the sky below the ground. An upside-down truck is not a case that exists; the capacity spent learning it is capacity taken from the 9–19 px band that does. |
| Rotation beyond ±10° | the camera is bolted to a pole. Roll is bounded by installation error and wind, not by arbitrary rotation. Large rotation also rotates axis-aligned boxes into boxes that no longer bound their object, injecting label noise. |
| Strong perspective warp (> 0.001 in Ultralytics terms) | perspective is **fixed per camera** and is exactly what the Phase 6 ground-plane homography exploits to convert pixel thresholds into metres (`PHASE_MINUS1_SCOPE` §4 row 5). Randomising it during training teaches the model to ignore the very cue the rule engine depends on. |
| Mixup / cutmix on detection | blending two scenes produces ghost objects at partial alpha. On a precision-critical task whose headline number is a false-alert count, translucent phantom people are the worst possible training signal. |
| Channel shuffle / false-colour mapping | breaks the B = G = R invariant that §3.4 establishes for infrared, and `MEASUREMENTS.md` §1 already measured what false-colour thermal does: AP@50 0.133. |
| Random erase over > 25% of a box | at that size it is indistinguishable from teaching the model that a partly visible person is background — the opposite of what occlusion handling needs. |

---

## 5. Hard-negative mining plan

The 10% negative pool from §3.5 must be made of the *right* negatives — the scenes that actually
produce false alerts on a border camera, not random empty crops.

**Six target categories, each with a free source.**

| category | why it causes false alerts | free source |
|---|---|---|
| Empty road, no actors | the baseline case; 23 of 24 hours | IDD frames whose XML contains zero in-schema objects after the §1.2 mapping. Free, already downloaded, correctly licensed for research use. |
| Moving foliage, wind | branch motion fires the motion channel (`edge/pipeline/motion.py`) and produces a candidate the appearance channel must reject | IDD roadside frames with zero objects; plus **DAVIS 2016/2017** (free, research licence) sequences of vegetation. |
| Headlight glare, night blooming | a blown-out headlight is person-sized and person-bright on an 8-bit frame | FLIR RGB **night** frames with zero in-schema objects (LLVIP is excluded: unlabelled vehicles). Free, already in the corpus. |
| Rain streaks, fog, haze | streaks are thin vertical structures; fog collapses contrast | the **Rain100 / RainCityscapes**-style free academic sets, and IDD's own monsoon-weather frames (IDD is explicitly an unstructured-Indian-road set and includes adverse weather). |
| Infrared hot spots | a sun-warmed rock, an engine block, a transformer — all person-temperature on LWIR | FLIR thermal frames with zero in-schema objects, filtered to those containing a high-intensity blob between 100 and 2,000 px² (the person-sized band) that is *not* annotated. This is the highest-value negative in the whole pool and it costs nothing. |
| Animals | IDD class 9, dropped from the label set in §1.2 but kept as an image | IDD frames whose only objects were `animal`. Already downloaded. Zero extra cost. |

**How they enter training.** A negative is an image file with an **empty (zero-byte) label file
beside it**. That is the Ultralytics convention for a background image and needs no special code
path. `07_negatives.py` builds the pool, tags each entry with its category in
`manifests/negatives.json`, and samples to hit exactly the §3.5 ratio with the per-category quota
below, so the pool cannot be 90% empty road.

| category | share of the negative pool |
|---|---|
| empty road | 30% |
| IR hot spots | 20% |
| headlight glare / night | 20% |
| foliage | 15% |
| weather | 10% |
| animals | 5% |

Everything above is already-downloaded data or a free academic download. **No paid source, no
scraping of a site whose terms forbid it, no manual collection budget.**

---

## 6. Synthetic Nepali plate set

**Why this exists.** `MEASUREMENTS.md` §4 records that PP-OCRv5 Devanagari read a *synthetic* plate
correctly at 0.989 / 0.990, but read **about half the characters** on a real Nepali plate. Slide 5's
**85% plate recognition is a TARGET**, and `MEASUREMENTS.md` states plainly that fine-tuning is
required to reach it. This section specifies the corpus for that fine-tune. **Target: >= 20,000
samples.**

### 6.1 Plate format

The format below is the **zone-based format**, and it is the one this spec builds against because
it is the one `MEASUREMENTS.md` §4 already demonstrates end to end:

> बा १२ प १२३४ — zone `बा` (Bagmati), lot number `१२`, vehicle-class letter `प`, serial `१२३४`

Layout: **two lines**, embossed. Line 1 carries the zone letter(s) and the lot number; line 2
carries the vehicle-class letter and the four-digit serial. Aspect ratio, exact kerning and the
embossing depth are **OQ-7**.

**Character inventory.**

- Devanagari digits, all ten, certain: `० १ २ ३ ४ ५ ६ ७ ८ ९`
- Zone letters — the fourteen administrative zones of the pre-2015 structure, which the plate
  series is built on: `मे से को सा ज ना बा ग धौ लु रा भे से म`. The precise orthography of several
  of these (including whether a zone is rendered with one akshara or two) is **OQ-8**; only `बा` is
  confirmed, by `MEASUREMENTS.md` §4.
- Vehicle-class letters: `प` is confirmed by `MEASUREMENTS.md` §4. The remainder of the class
  inventory (the letters distinguishing private, public, government, tourist and heavy vehicles) is
  **OQ-9**.

**Colour series.** Nepali plates are colour-coded by ownership class — private, public/commercial,
government, tourist and diplomatic series exist and differ in background and text colour. The exact
colour-to-class assignment is **OQ-10** and is **not invented here**. The generator reads the
colour table from `datasets/config/`-adjacent YAML inside `plates/`, so a corrected table changes a
config file and regenerates the corpus without touching code.

**Everything marked OQ-7 through OQ-10 is a configuration value, never a hard-coded constant.** That
is the whole design response to not being certain: be wrong in a file that takes 30 seconds to fix.

### 6.2 Rendering

- **Font: Noto Sans Devanagari**, SIL Open Font License 1.1 — free, redistributable, and it covers
  the full Devanagari block including the conjuncts a zone letter may need. The binary is **not
  committed**; `plates/fonts/README.md` gives the fetch command.
- Render with PIL's FreeType binding at **4× the target resolution**, then downsample with
  `INTER_AREA`. Rendering small and upscaling produces aliasing that the degradation stage would
  then amplify into artefacts no real camera produces.
- Emboss simulation: render the glyph layer twice with a 1–2 px offset — a dark copy and a light
  copy — under the flat colour layer. This is what makes a synthetic plate look pressed rather than
  printed, and printed-looking plates are the reason a model trained on naive synthetic data fails
  on real ones.
- Plate furniture: border rule, corner radius, and rivet/bolt marks at the mounting points, each
  jittered.

### 6.3 Degradation pipeline

Applied in this order, each stage independently sampled. The ranges are chosen so the *hardest*
samples are roughly as bad as the check-post frame in `MEASUREMENTS.md` §4, and no worse — a corpus
of illegible plates teaches nothing.

| stage | probability | parameter range |
|---|---|---|
| perspective warp | 0.9 | yaw ±35°, pitch ±25°, roll ±8° |
| scale to target | 1.0 | final plate width 60–260 px (a plate at a barrier versus a plate at 40 m) |
| motion blur | 0.45 | kernel 3–15 px, angle 0–180° |
| defocus blur | 0.30 | Gaussian σ 0.5–2.5 |
| low light | 0.40 | gamma 1.4–3.0, brightness ×0.25–0.8 |
| infrared wash-out | 0.15 | convert to single channel, compress to 35–70% dynamic range, replicate to 3 channels per §3.4 |
| rain | 0.15 | 40–200 streaks, length 8–30 px, opacity 0.1–0.45 |
| dirt / mud occlusion | 0.35 | 1–6 blobs, each covering 2–12% of plate area, never more than 25% total |
| specular glare | 0.25 | one elliptical highlight, 10–35% of plate area, additive |
| sensor noise | 0.60 | Gaussian σ 2–12, plus salt-and-pepper at 0.0005–0.003 |
| JPEG artefacts | 0.85 | quality 25–85 |

Composition onto a background is optional and off by default: the OCR recogniser is fine-tuned on
**cropped plate strips**, so the detection context is not needed. A 500-sample composited subset is
produced anyway, for the end-to-end demo script capability 4 requires.

### 6.4 Label format

PaddleOCR **recognition** format — one line per sample, tab-separated, UTF-8, no BOM:

```
out/images/000001.jpg<TAB>बा१२प१२३४
```

- `plates/labels/rec_gt_train.txt` and `plates/labels/rec_gt_val.txt`, **90 / 10**, split by
  *serial number* so no serial appears in both files.
- `plates/labels/charset.txt` — one character per line, the exact inventory from §6.1. The charset
  file is generated from the config, never hand-typed, so it cannot drift from what was rendered.
- The ground-truth string is the plate's characters with the inter-field spaces removed, since the
  recogniser reads a cropped strip and spacing is a layout property, not a character.

### 6.5 Real-plate validation — the blunt version

`MEASUREMENTS.md` §4 already states the problem: the real Nepali plate photo used for the
half-characters observation is **copyrighted and unusable**. That is not a solvable problem inside
this phase's budget and this spec does not pretend otherwise.

What is attempted, in order:

1. **Openly-licensed photographs.** Wikimedia Commons and Openverse carry CC-BY / CC0 street
   photography from Nepal. Target: **>= 30 plate crops**, each with its source URL, licence and
   author recorded in `plates/labels/real_eval_sources.csv`, attribution honoured. Crops are
   derivative works; CC-BY permits it with attribution, CC-BY-NC does **not** fit a project that
   may be demonstrated commercially, so NC-licensed images are excluded.
2. **Own photography.** Any plate photographed by the team is usable outright. Zero at the time of
   writing.

**If fewer than 30 usable crops are obtained, the report says exactly this and nothing more:**
"Plate recognition accuracy is reported on synthetic data only. The 85% figure from slide 5 remains
a target; no real-plate evaluation was possible within the licensing constraints, and the only
real-plate observation available — that the pretrained model reads about half the characters — is
recorded in `MEASUREMENTS.md` §4." **A synthetic accuracy number is never presented as a real-plate
accuracy number.** This is the single most tempting dishonesty available in this project and it is
ruled out here in writing.

---

## 7. Dataset card

Written in full to `docs/DATASET_CARD.md` in Phase 1B. Summary:

| source | size | licence | redistribution |
|---|---|---|---|
| **IDD Detection** | 46,588 images (31,569 / 10,225 / 4,794), IIIT Hyderabad; read from the third-party Kaggle mirror `vinayak21574/idd-detection` | Non-commercial research use, behind registration and a click-through agreement at `idd.insaan.iiit.ac.in`. **The exact clause text is OQ-4** — it is read and quoted at fetch time, not paraphrased from memory. | **Assume NO.** `fetch_idd.py` already refuses to bypass the gate; nothing derived from IDD is re-hosted. |
| **LLVIP** | 15,488 visible/infrared pairs, 12,025 train / 3,463 test (verified), third-party Kaggle mirror `afradhossain/llvip-dataset` of the BUPT release | Research use; the upstream repository states its own terms. **OQ-5.** | **Assume NO.** |
| **Teledyne FLIR ADAS v2** | 15,152 RGB + 15,635 thermal frames (verified), third-party Kaggle mirror `samdazel/teledyne-flir-adas-thermal-dataset-v2` | Distributed free by Teledyne FLIR after a registration form; the mirror labels it "other". The exact terms are **OQ-12**. | **Assume NO.** |
| ~~KAIST Multispectral Pedestrian~~ | disabled: no reachable annotations | **OQ-5.** | **Assume NO.** |
| **Synthetic Nepali plates** | >= 20,000, generated here | Ours. Font is OFL-1.1, which permits the rendered output freely. | **Yes** — this is the only part of the corpus that can be published. |

**Consequence, stated up front:** the combined corpus **cannot be re-hosted** as a Kaggle Dataset or
a Hugging Face dataset repo. §8 is designed around that constraint, not against it.

**Known biases**, carried into the card verbatim:
- Geographic: IDD is Indian urban/semi-urban roads; FLIR ADAS is North American and European
  driving; LLVIP is Chinese street scenes. **No source is an Indo-Nepal border post.** Every number this corpus
  produces is a public-dataset number, exactly as slide 5's status line already says.
- Viewpoint: IDD and FLIR are vehicle-mounted; LLVIP is an elevated street camera. **None is pole-mounted at
  4–6 m looking down a 150 m approach.** §3.3 tiling and §4.1 downscale augmentation are partial
  compensations, not a fix.
- Taxonomy impurity: FLIR `other vehicle` and IDD `vehicle fallback` are unlabelled vehicles
  inside labelled frames (§1.3, §1.5); FLIR `scooter` semantics are OQ-14.
- Class skew toward `person`; `cart` conditional on the §1.5 gate.
- Spectral: infrared is LWIR only. No NIR, no active illumination.

**Intended use:** fine-tuning one YOLO11-s detector for the five classes in §1.1, for a
proof-of-concept border-surveillance prototype.
**Out of scope:** operational deployment; any identification of individuals; any claim of field
validation; re-distribution of the source datasets; commercial use of anything derived from IDD.

---

## 8. Storage and transfer

**As implemented (2026-09): the corpus is built on Kaggle.** The three mirrors are attached to a
CPU notebook (`datasets/notebooks/kaggle_build.ipynb`) as read-only inputs; `00_fetch.py`
records where each is mounted (it never copies or extracts), intermediate state lives in `/tmp`,
and only the finished dataset is written to `/kaggle/working/truewatch_ds` (≤ 20 GB output
limit; projected 13-16 GB). Every image is JPEG q92; untiled frames are capped at **1280 px**
on the long side (2× the 640 training size, headroom for the 0.5-1.5× scale augmentation, and
IDD 1920×1080 / FLIR 1800×1600 frames drop from ~0.5-1.2 MB to ~0.2-0.3 MB); tiles are cut at
native resolution before the cap and never resized. `09_build_yolo_ds.py` projects the output
size after the first ~2% of images and stops above 19.5 GB. The build directory is attached to
the training notebook directly; it carries `data.yaml` with a relative `path`, the manifests
(hard set, splits, negatives, cart gate) and the reports.

The plan below is the original local-build design, kept for the local path.

**The constraint from §7: source data cannot be re-hosted.** So the pipeline never uploads source
imagery anywhere — it uploads **scripts and manifests**, and rebuilds the corpus on the machine
that needs it.

| step | where | size | note |
|---|---|---|---|
| raw KAIST | local `var/datasets/kaist` | ~15–20 GB full; ~1 GB with `--pattern` subset | fetched by `huggingface_hub`, resumable, cached |
| raw LLVIP | local `var/datasets/llvip` | ~4–6 GB | same |
| raw IDD | local `var/datasets/idd` | ~18–22 GB | **manual download**, licence gate respected |
| converted labels + 3-channel IR PNG | local `datasets/processed` | ~25–35 GB | the expensive intermediate |
| far-field tiles | local `datasets/processed/tiles` | +6–10 GB | 3 tiles per far-field frame |
| **final YOLO dataset** | local `datasets/processed/yolo` | **~12–18 GB** after JPEG re-encode at q90 and 1280 px long-side cap | this is the only thing training needs |
| Kaggle Dataset | Kaggle | must be **< 20 GB** per dataset (free tier: 20 GB per private dataset, 100 GB total) | fits, with the cap above |

**How it moves, free:**
1. Local Mac: fetch → convert → dedupe → split → negatives → tile → build. CPU only, no CUDA.
2. `09_build_yolo_ds.py` emits the final tree plus `data.yaml` plus a sha256 manifest.
3. Upload to a **private** Kaggle Dataset with the Kaggle CLI (`kaggle datasets create -p`). Private
   satisfies the no-redistribution constraint in §7: it is a personal working copy, not publication.
4. Kaggle notebook attaches the dataset read-only at `/kaggle/input/<slug>`. No copy, no download
   inside the notebook, no GPU time spent on I/O.
5. Plates go to their own small Kaggle Dataset (~1.5 GB) because they regenerate independently and
   are the only publishable part.

**Staying inside the limit.** Three levers, applied in order until under 18 GB: JPEG q90 instead of
PNG for visible frames (IR stays PNG — lossless matters on a low-contrast single channel
replicated three ways); long side capped at 1280 px for **untiled** images only, never for tiles,
which is the entire point of tiles; and if still over, drop the lowest-value IDD drives by scene
duplication score from `05_dedupe.py` rather than by random sampling.

**Cached vs streamed.** Raw sources are cached locally and never re-downloaded (`00_fetch.py` is
resumable and count-verified). The final dataset is materialised, not streamed — Kaggle's
`/kaggle/input` is already a mounted read; streaming from the Hub inside a training loop wastes the
GPU budget on network waits.

---

## 9. Validation gates

`10_validate.py` runs all of these, prints a PASS/FAIL table, and **exits non-zero if any fails**.
Phase 2 does not start until it exits 0.

| # | gate | check | pass threshold |
|---|---|---|---|
| G1 | No duplicate images across splits | 64-bit perceptual hash of every image; compare across splits | **0** cross-split pairs at Hamming distance **<= 4** |
| G2 | No near-duplicates within a split | same hash, within-split | **< 0.5%** of images have a within-split duplicate at distance <= 2 |
| G3 | No leaked sequences | group final assignment by `sequence_key` | **every** key resolves to exactly **1** split; count of multi-split keys must be **0** |
| G4 | Label values in range | every `cx cy w h` parsed | **100%** in `[0.0, 1.0]`; **0** violations |
| G5 | No zero-area box | `w * h` in normalised units | **0** boxes with `w <= 0.0005` or `h <= 0.0005` |
| G6 | Class ids in schema | every first field | **100%** in `{0,1,2,3,4}`; **0** out-of-schema ids |
| G7 | Image/label parity | every image has a readable, parseable label file (empty counts as readable) and every label has an image | **0** orphans on either side; counts equal |
| G8 | Day/night ratio | visible:infrared per split over every built image, tiles and negatives included | each split between **55:45 and 70:30** |
| G9 | Negative ratio | zero-byte label files in **train** | **10% ± 2%** (i.e. 8.0–12.0%) |
| G10 | IR frames are genuinely 3-channel | the **written** files: `cv2.imread(p).shape` and B = G = R on a random **500**-image IR sample | **100%** report `(_, _, 3)` with identical channels; **0** failures |
| G11 | Class histogram present and no silent dead class | per-class instance counts written | histogram file exists; **every shipped class >= 1.0%** of instances, or the class name appears in the documented-exception list |
| G12 | Box-size histogram present with a small-object tail | height histogram in px **at the 640 px detector input** over 8 bins | file exists **and >= 5%** of person boxes fall below **19 px** — the corpus must actually contain the hard band it claims to train for |
| G13 | Far-field tiles present | object tiles in train (empty negative tiles not counted) | **>= 2,000** tiles, each 640×640 exactly |
| G14 | Split proportions | untiled image counts | train **70% ± 3**, val **15% ± 3**, test **15% ± 3** |
| G15 | Hard set frozen and disjoint from test | membership of `hard_set.txt` against the built index | exactly **280** entries, **0** in test, **all** in the built val split |
| G16 | Determinism | re-run `06_split.py --seed 42` | sha256 of `splits.json` **identical** to the recorded value |
| G17 | Plate corpus size and label validity | count of generated samples and label lines | **>= 20,000** samples; label lines == image count; **0** label strings containing a character outside `charset.txt` |
| G18 | No data staged for commit | scan the working tree against `.gitignore` patterns | **0** files matching `*.jpg *.jpeg *.png *.mp4 *.zip *.tar *.pt *.onnx` under `datasets/` outside `docs/images/` |
| G19 | **Manual** 100-image visual spot check | 100 random train images rendered with boxes drawn to `reports/spotcheck/` | a human opens them and records **PASS/FAIL in `reports/spotcheck/VERDICT.txt`**; the gate reads that file and fails if it is missing or says FAIL |

Nineteen gates, every one with a number. `10_validate.py` validates the **built** dataset
(`<dataset>/index/split.jsonl` and the written files), so it runs wherever the build is mounted.
G17 may be declared out of scope for a detection-only build with `--skip-plates`: it then prints
**SKIP**, is recorded as skipped, and is never counted as a pass. G19 is deliberately not automatable — a human looking at
100 images with boxes drawn catches coordinate-convention errors that no assertion catches, and the
gate is written so the build cannot pass by skipping it.

---

## 10. Open questions

Nothing below is asserted as fact anywhere else in this document.

- **OQ-1 — KAIST set lighting composition.** Moot while KAIST is disabled. The §2.2a assignment assumes set00–02 day, set03–05
  night, set06–08 day, set09–11 night. Verified at runtime by `02_convert_kaist.py`, which fails if
  a set's measured statistics contradict its declared lighting rather than proceeding on the
  assumption.
- **OQ-2 — IDD directory layout. RESOLVED 2026-09**: per-image VOC XML in a parallel tree plus
  split lists (§1.2), verified on the Kaggle mirror.
- **OQ-3 — LLVIP scene grouping. RESOLVED 2026-09**: the two-digit id prefix is the scene (§2.3).
  The pHash fallback stays in code.
- **OQ-4 — IDD licence clause text.** Read and quoted at fetch time into the dataset card. Not
  paraphrased from memory here.
- **OQ-5 — LLVIP and KAIST licence and exact size.** The Hugging Face dataset cards are read at
  fetch time and their licence strings copied verbatim into `sources.yaml` and the dataset card.
  Redistribution is assumed **forbidden** until a card says otherwise.
- **OQ-6 — infrared polarity convention** per sequence in FLIR and LLVIP. Polarity inversion stays
  out of the augmentation list until this is known.
- **OQ-7 — Nepali plate geometry**: exact aspect ratio, line split, kerning, embossing depth.
- **OQ-8 — zone-letter orthography**: the exact akshara for each of the fourteen zones. Only `बा` is
  confirmed, by `MEASUREMENTS.md` §4.
- **OQ-9 — vehicle-class letter inventory**. Only `प` is confirmed, by `MEASUREMENTS.md` §4.
- **OQ-10 — plate colour series to ownership class mapping.**
- **OQ-11 — the cart outcome.** Resolved by running the §1.5 procedure, not by discussion. Until
  the review sheet is filled in, every build withdraws class 4.
- **OQ-12 — Teledyne FLIR ADAS v2 licence.** Teledyne FLIR distributes the dataset free after a
  registration form; its public pages checked 2026-09 do not state the terms, and the Kaggle
  mirror labels the licence "other" without text. Redistribution is assumed forbidden.
- **OQ-13 — FLIR RGB/thermal video pairing.** The mirror does not carry the RGB-to-thermal video
  map, so the split assertion (§2.2) checks videoIds within each modality only. That the
  official partition keeps an RGB video and its thermal twin in the same split is assumed.
- **OQ-14 — FLIR `scooter`.** Whether it is a kick scooter or a motor scooter; either way it is
  class 1 (§1.3). 15-41 instances per subset.

OQ-7 through OQ-10 are all configuration values in `plates/`, so correcting any of them is a YAML
edit and a regeneration, not a code change.

---

## 11. Time budget

Eleven hours total, as briefed.

| block | hours | what |
|---|---|---|
| Day 1 | 1.0 | fetch KAIST + LLVIP (background), start the IDD manual registration and download |
| Day 1 | 1.5 | conversion scripts run and their dropped-class tallies read |
| Day 1 | 0.75 | the §1.5 cart review pass |
| Day 1 | 1.25 | dedupe + split + verify no leakage |
| Day 1 | 1.5 | negatives pool and its per-category quota |
| Day 2 | 1.5 | far-field tiling and the final build |
| Day 2 | 1.0 | plate generation and eyeballing 20 of them |
| Day 2 | 1.0 | gates, stats, and the G19 manual spot check |
| Day 2 | 1.5 | Kaggle upload and the notebook-side read test |

Fetch time is wall-clock, not attention, and overlaps everything above it.

---

## SELF-CHECK

- [x] **Every native class of IDD, FLIR, LLVIP (and the disabled KAIST) is mapped or explicitly dropped.** IDD: all 15
      enumerated in §1.2 — 8 mapped, 4 dropped (`animal`, `traffic light`, `traffic sign`, `train`),
      2 merged into `truck` (`caravan`, `trailer`), 1 conditional (`vehicle fallback`, §1.5).
      FLIR (replaces KAIST, 2026-09): all 16 occurring names in §1.3 — 8 mapped (`person`,
      `bike`, `motor`, `scooter`, `car`, `truck`, `bus`, plus `rider` if it ever occurs), 9
      dropped and counted. KAIST (disabled): 4 tags in §1.3a. LLVIP: 1 class, mapped. None
      unaccounted.
- [x] **Split policy makes sequence-level leakage structurally impossible.** Mechanism (§2.1): a
      `sequence_key` is assigned at conversion time and the split file maps `sequence_key -> split`;
      no code path maps an image to a split. Leakage would require one key to hold two splits, which
      gate G3 asserts is impossible with a threshold of 0.
- [x] **Negative/background ratio is a number tied to the false-alert target.** §3.5: **10% of
      training images**, bounded 5–15%, gated at ±2% by G9, tied to slide 5's under-5-false-alerts
      target because precision on a 24-hour stream is set by behaviour on the empty 23 hours.
- [x] **SAHI parameters justified against 19/14/9 px.** §3.3: 640×640 tiles at native resolution,
      0.20 overlap, upper 40% strip. A 1920-wide frame downscales 3.0× to 640, so a 27 px person
      becomes 9 px (49% recall, `MEASUREMENTS.md` §3); tiling at native resolution keeps it at 27 px
      (100%). Overlap 128 px exceeds every far-field object dimension.
- [x] **IR 3-channel handling matches slide 4.** §3.4 step 3: `np.repeat` of the single LWIR channel
      to B = G = R, no false colour, justified by the measured AP@50 0.133 on false-colour thermal.
      Verified by gate G10 at 100% of a 500-image sample.
- [x] **Every licence claim is sourced or marked OPEN QUESTION.** IDD gate → OQ-4; FLIR → OQ-12; LLVIP and KAIST →
      OQ-5; redistribution assumed forbidden by default in §7; OFL-1.1 for Noto Sans Devanagari is
      the one licence stated outright, and it is the font's published licence.
- [x] **>= 12 validation gates with numeric thresholds.** §9 lists **19**, each with a number.
- [x] **Nothing requires a paid service.** Hugging Face Hub (free), IDD registration (free), Kaggle
      free tier (30 GPU-h/week, 20 GB private dataset), Noto Sans Devanagari (OFL), Wikimedia
      Commons / Openverse (CC), local Apple Silicon CPU. No paid storage, no paid API.
- [x] **Watermark scan.** Every forbidden assistant-tool token and attribution phrase on the brief's
      list was searched for case-insensitively across this file. Count for each of the eight:
      **0, 0, 0, 0, 0, 0, 0, 0.** The scan command and its output are recorded in the commit
      message trailer-free body rather than reproduced here, because writing the tokens out in
      order to claim they are absent would itself put them in the file.

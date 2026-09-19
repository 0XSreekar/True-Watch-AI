# TRUEWATCH — Phase -1 Scope Lock

Problem Statement SIH26187 · AI-Based Intelligent Video Analytics Platform for Border
Surveillance using existing CCTV Infrastructure · Theme Blockchain & Cybersecurity ·
PS Category Software · Team ID 130977 · Team Name Future Bytes ·
Organization Ministry of Home Affairs; Sashastra Seema Bal (SSB), Police II Division. *(Slide 1)*

**Authority.** `SIH26187_TRUEWATCH_final-1.pdf` is the single source of truth for this
document. `MEASUREMENTS.md` is the single source of truth for every number described as
measured. Nothing else is admissible. Where a statement made during scoping was not
supported by those two files it is recorded in **Open Questions** rather than asserted.

**Purpose.** This file freezes what TRUEWATCH is, what each of the eight capabilities means
concretely, what is already measured, what is only a target, and what is not being built in
the six days available. No later phase may add a requirement that is not in this file.

---

## 1. One-paragraph idea

The Indo-Nepal border is an open border. Crossing it is legal, and thousands of people do so
every day. That is why the useful problem is not spotting movement — every camera already
sees movement all day — but **deciding which few movements out of thousands a jawan should
actually review**. Slide 2 states this directly: *"The Indo-Nepal border is open, so thousands
cross legally every day. The problem is not spotting them; it is deciding which few movements
a jawan should review."* TRUEWATCH is software that attaches to the cameras already installed
at border out posts and does that selection. Two independent channels watch every frame — one
judges appearance, the other judges motion — and an alert is raised only when both agree.
A vision-language model then writes, in plain words, why that particular movement was
selected, so the operator acts on a sentence instead of re-watching footage. Every clip is
hashed and chained at the moment it is recorded, so the record stands up later as evidence.
No new hardware goes on the pole.

---

## 2. The problem, restated — with slide citations

| Fact | Slide | Exact basis |
|---|---|---|
| The Indo-Nepal border is open; crossing is legal; thousands cross every day | 2 | Hook line, quoted in §1 above |
| The problem is selection, not detection | 2 | *"it is deciding which few movements a jawan should review"* |
| SSB operates **734 border out posts** | 4, 6 | Slide 4: *"Designed to scale to SSB's 734 border outposts"*. Slide 6 ref 8 (The Tribune, 17 Aug 2025) is given as the source for the 734 figure |
| **308** of those out posts have **no road connectivity** | 4, 5, 6 | Slide 4: *"the 308 posts with no road"*. Slide 5: *"308 posts with no road: reached by a board, not a truck"*. Slide 6 ref 8 |
| Cameras are **already installed**; no new hardware is proposed | 3, 4 | Slide 3: *"Existing IP camera — no new hardware on the pole"*. Slide 4 Operational: *"It reads the ONVIF and RTSP feeds already installed"*. Slide 4 Economic: *"No new hardware tender, no camera replacement, no vendor lock-in"* |
| SSB's area of responsibility: **Indo-Nepal 1,751 km**, **Indo-Bhutan 699 km** | 6 | Ref 9, ssb.nic.in |
| Nepali plates are Devanagari; **Bhutan** plates are Latin alphanumeric (BP-N-ANNNN) and are already covered by standard ANPR | 2, 4 | Slide 2 innovation bullet 1; slide 4 *"Bhutan posts"* note |
| One control room officer currently faces 8 cameras x 24 h = **192 camera-hours a day** | 5 | *"Control room officer: 8 cameras x 24 h = 192 camera-hours a day; target under 40 alerts"* |

### Two claims that were asserted during scoping but are NOT in the PDF

1. **"2,450 km."** This figure does not appear anywhere in the deck. The deck gives
   1,751 km (Indo-Nepal) and 699 km (Indo-Bhutan) on slide 6. Those sum to 2,450 km, but the
   deck never states the total, and the two borders are treated differently by the solution
   (Devanagari ANPR applies only on the Indo-Nepal side — slide 4). **Use "1,751 km Indo-Nepal
   and 699 km Indo-Bhutan, slide 6" and not "2,450 km".** The PDF wins.
2. **"Every piece of intelligence still comes from a human watching."** Not stated in the deck
   in that form. The supported statements are the slide 2 hook (*"No lamps. Nobody is
   watching."*), slide 5 *"Operational: ranks alerts, so less manual watching"*, and slide 5
   *"Investigating officer: searches archived footage instead of scrubbing it."* Use those.

---

## 3. The eight capabilities — concrete definitions

All eight are taken verbatim from slide 2, *"How it addresses the problem: all eight
capabilities asked for in the problem statement"*. Model/method comes from slide 2 and slide 3.
None added, none dropped, none renamed.

Acceptance criteria below are written to be executable in six days, with free tools, on the
public datasets that `MEASUREMENTS.md` records as verified available (KAIST Multispectral
Pedestrian, LLVIP, IDD Detection). Each criterion is a script that runs and emits a number —
**not** a claim that a number will reach a particular value.

| # | Capability (slide 2) | Model / method per PDF | INPUT | OUTPUT | Testable acceptance criterion (6 days, free) |
|---|---|---|---|---|---|
| 1 | Human detection and tracking | YOLO11-s with ByteTrack, day and IR (slide 2); fine-tuning planned on IDD, LLVIP, KAIST (slide 3) | Decoded frames from one stream, visible or LWIR | Person boxes with confidence, plus a persistent track id per person | A committed script runs YOLO11-s + ByteTrack over a held-out KAIST clip of >= 100 frames and writes a CSV of tracks with id, frame span, dwell and net displacement, reproducing the format of `MEASUREMENTS.md` §2. It prints person AP@50 against KAIST annotations as an observed number. **Person mAP@50 0.85 day / 0.75 IR is a TARGET (slide 5), not something this criterion asserts.** |
| 2 | Vehicle detection and classification | YOLO11-s: truck, car, two-wheeler, cart (slide 2) | Decoded frames | Vehicle boxes with one of the four class labels | A script reports per-class AP@50 over a fixed IDD Detection subset (>= 200 images) and writes the numbers to `MEASUREMENTS.md`. **"cart" is not a COCO class and the pretrained model cannot emit it — see Open Question 1.** The criterion passes when the four-class confusion matrix exists, whatever it says. |
| 3 | Face detection | YuNet, close range at gates and check posts, pretrained (slides 2, 3) | Cropped frames from gate/check-post cameras only | Face boxes; face data stays on the post, 30-day retention, unsealing is access-controlled (slide 4) | A script runs YuNet over a fixed close-range image set, reports detection count and mean confidence, and demonstrates that face crops are blurred in any output leaving the post, with unblurring behind an explicit access step. **No face accuracy target exists in the PDF; do not invent one.** |
| 4 | ANPR | PP-OCRv5 Devanagari, Nepali plates on Indo-Nepal posts, pretrained (slides 2, 3) | Cropped plate region | Plate string in Devanagari, with per-line confidence | A committed, rerunnable script reproduces the `MEASUREMENTS.md` §4 result: Devanagari OCR reads the synthetic plate correctly at confidence 0.989 / 0.990 while the Latin model returns `"9 9238"` (wrong). Extended to a >= 20-plate synthetic set with read accuracy reported. **85% recognition is a TARGET (slide 5) and `MEASUREMENTS.md` §4 states fine-tuning on real Nepali plates is required to reach it.** |
| 5 | Virtual fence intrusion | Track crosses a line the operator draws (slide 2); thresholds in metres via a ground-plane homography so they transfer between cameras (slide 4) | Track polylines plus an operator-drawn line | A crossing event naming the line and the track id | A script reproduces `MEASUREMENTS.md` §2 exactly: fence at x = 320, 10 tracks evaluated, tracks 4, 6 and 9 fire `fence crossed`. Output is byte-comparable to the table in that file. **"Under 2% missed fence crossings" is a TARGET (slide 5).** |
| 6 | Suspicious activity | Rules on tracks plus a per-camera baseline (slide 2); named rules, baseline learnt from the camera itself (slide 4) | Track set plus the baseline learnt from that camera | Alert objects, each naming the rule that fired | A script reproduces `MEASUREMENTS.md` §2: baseline learnt as majority flow left-to-right (6 with, 4 against); rules `wrong direction`, `fence crossed`, `loitering` (dwell >= 3 s AND net displacement < 90 px); result 10 tracks, 6 alerts, 4 normal; track 6 fires `fence crossed + loitering` on 70 px net versus 163 px path. Every alert names its rule. **Slide 4 already concedes this is a rule-sanity check, not a labelled evaluation. Keep that wording.** |
| 7 | Night-time movement | IR fine-tuning; motion channel weighted higher (slide 2); IR replicated to 3 channels, fine-tune on LLVIP and KAIST (slide 4) | Paired visible / LWIR frames | Detections plus a motion-channel weight that rises after dark | A script reproduces `MEASUREMENTS.md` §1 on the same KAIST set: 9 visible person detections, 8 under LWIR, 89% IR recovery, mean confidence 0.59 -> 0.54; and dense Farneback object-to-background flow contrast 1.9x visible versus 3.4x LWIR over 100 paired frames. The fusion weight must demonstrably change as a function of that contrast. **The false-colour thermal result AP@50 = 0.133 (precision 0.364, recall 0.093 at conf 0.50) must be reported alongside — it is the weakest measured number and it is measured.** |
| 8 | Real-time alerts and event logging | Detector ~30 ms, explanation ~3 s, hash-chained log (slide 2); SHA-256 at capture, daily Merkle root signed at sector HQ (slide 3) | Confirmed alert plus its clip | An append-only chained log entry, and an alert delivered to the console | A `verify` command walks the chain and exits 0 on an intact chain; when one clip is altered by a single byte the same command exits non-zero and names the first broken block. A day folds into one Merkle root. **~30 ms is a TARGET and slide 3 defines it precisely as "the YOLO11-s INT8 inference budget at 640 px, not end-to-end latency" — never quote it as end-to-end. ~3 s for the explanation is likewise a TARGET.** |

---

## 4. The four innovation claims — and the artefact that proves each

Taken verbatim from slide 2, *"Innovation and uniqueness of the solution"*.

| Claim (slide 2) | What it asserts | Artefact in this repo that proves it to a judge |
|---|---|---|
| **(a) Reads Nepali number plates.** LPR stacks such as Frigate provide Latin and Chinese recognition models, but not Devanagari. | Off-the-shelf ANPR cannot read the plates at Indo-Nepal posts; this stack can. | A single script, committed and rerunnable, that takes the same plate image and prints **both** outputs side by side: Latin OCR -> `"9 9238"` (wrong), PP-OCRv5 Devanagari -> the correct Devanagari string at 0.989 / 0.990. The comparison is the proof — one model alone proves nothing. Backing citation: slide 6 ref 11, `docs.frigate.video`, Frigate NVR reads Latin and Chinese, not Devanagari. Slide 6 ref 4, `paddleocr.ai`, Nepali listed among supported languages. Plus the console showing a Devanagari plate on an alert card. |
| **(b) Two channels built on different principles, so they do not fail on the same frame.** | Independence, not redundancy: appearance and motion fail for different reasons. | A run log over a fixed clip recording, per frame, the appearance verdict, the motion verdict and the fused verdict, plus a count of frames where exactly one channel fired. If that count is zero the claim is unproven and must be withdrawn. The measured backing for *why* motion carries the night is `MEASUREMENTS.md` §1: object-to-background flow contrast 1.9x visible versus 3.4x LWIR. **Do not reuse the retracted wording "optical flow is unaffected by IR" — `MEASUREMENTS.md` §1 explicitly records that as wrong and removed. Whole-frame flow on LWIR is only 32% of visible.** |
| **(c) Every alert carries a written reason, so the operator acts instead of reviewing footage.** | The output is a sentence, not a bounding box. | An alert record whose stored payload contains a plain-language sentence generated for that specific event, alongside the structured detection fields — and the same sentence rendered on the `AlertQueue` card in `/console`. The existing card already renders `alert.reason` at the `conf` stage, so the proof is that the sentence is produced by the model rather than drawn from fixtures. Model per slide 3: Moondream 2, pretrained. |
| **(d) Evidence is chained and signed as it is recorded, not afterwards.** | Ordering matters: hashing at capture, not at export, is what makes the record append-only. | The `verify` command from capability 8, plus a timestamp comparison showing the SHA-256 is written in the same operation as the clip, not in a later pass. Judge-facing surface: `EvidenceChainPanel` in `/console` walking the blocks and ending at `CHAIN INTACT — n / n VERIFIED`. Legal framing available on slide 6 ref 13, Bharatiya Sakshya Adhiniyam 2023 Section 63. |

---

## 5. Measured vs target

### 5.1 MEASURED — quoted exactly from `MEASUREMENTS.md`

These may be presented as measured. Each already exists; nothing here is pending.

**§1 Night and infrared** — KAIST Multispectral Pedestrian, set00/V007, paired visible + LWIR.
YOLO11-s pretrained on COCO, **no fine-tuning**:
- Person detections: **9 visible, 8 infrared (LWIR)**
- Mean confidence: **0.59 visible, 0.54 LWIR**
- IR recovery: **89% of visible**
- On false-colour thermal imagery, separate public set, 61 images, 43 persons:
  **AP@50 = 0.133**, **precision 0.364**, **recall 0.093 at conf 0.50**
- Dense Farneback optical flow over 100 paired frames, inside detected boxes vs background:
  visible **5.17 px** object / **2.74 px** background = **1.9x**;
  LWIR **2.73 px** object / **0.81 px** background = **3.4x**
- Retraction on record: whole-frame flow on LWIR is **32%** of visible; the claim
  "optical flow is unaffected by IR" was wrong and has been removed.

**§2 Suspicious activity without a dataset** — 100 consecutive KAIST visible frames,
YOLO11-s + ByteTrack:
- **14 tracks** formed, **10 lasting 8+ frames**
- Baseline learnt from the camera itself: majority flow left to right, **6 with, 4 against**
- Rules: fence at **x = 320**; wrong direction against the learnt baseline; loitering
  (**dwell >= 3 s and net displacement < 90 px**)
- Result: **10 tracks evaluated, 6 alerts, 4 normal**, no labelled training data
- Track 6: **70 px net, 163 px path**, fired `fence crossed + loitering`

**§3 Objects too small to detect** — 23 KAIST visible frames, 51 person detections at native
resolution, median person height **77 px**, conf 0.35:

| Person height | Detections | Recall vs native |
|---|---|---|
| 77 px | 51 | 100% |
| 54 px | 51 | 100% |
| 38 px | 51 | 100% |
| 27 px | 52 | 102% |
| 19 px | 46 | **90%** |
| 14 px | 41 | **80%** |
| 9 px | 25 | **49%** |

Cliff is between 19 px and 9 px. This is the measured basis for tiled far-field inference and
for grading cameras by DORI. (Slide 4 quotes the 90 / 80 / 49 figures; slide 6 ref 12 gives
the DORI bands from IEC 62676-4 — detect 25, observe 62, recognise 125, identify 250 px/m.)

**§4 Nepali plate reading** — synthetic plate rendered in-house, composited onto a public
vehicle photo at a check-post barrier, with blur, dimming and sensor noise:
- Latin OCR read **`"9 9238"` — wrong**
- PP-OCRv5 Devanagari read the plate **correctly at confidence 0.989 / 0.990**
- Caveat recorded in the same file: on a real, copyrighted, unusable photo of a Nepali plate
  the pretrained Devanagari model read **about half** the characters correctly and the Latin
  model returned nothing usable. **Do not quote the real-plate figure without that context.**

**Datasets verified available** — KAIST Multispectral Pedestrian (Hugging Face,
`richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset`, **23,210 files**, paired
visible/lwir); LLVIP (Hugging Face, `jsonhash/LLVIP`); IDD Detection (idd.insaan.iiit.ac.in,
**46,588 images**, 31,569 / 10,225 / 4,794).

### 5.2 TARGET — never to be presented as measured

Slide 5, box headed **POST-FINE-TUNING TARGETS**:
- Person mAP@50 of **0.85 day** and **0.75 IR** — TARGET
- Under **5 false alerts per camera per day** — TARGET
- Detection range **150 m at 25 px per metre on target** — TARGET
- Nepali plate recognition rate **85% or better** — TARGET
- Under **2% missed fence crossings** — TARGET

Slide 2 / slide 3 performance figures, all unmeasured on hardware:
- Detector **~30 ms** — TARGET, and slide 3 defines it as the YOLO11-s INT8 inference budget
  at 640 px, **not end-to-end latency**
- Explanation **~3 s**, asynchronous — TARGET
- Slide 5 control-room figure **target under 40 alerts** per day — TARGET

Slide 3, box headed **Design estimates, not yet measured on hardware**:
- **4 x 1080p at 8 to 10 FPS** with the VLM on alerts only — DESIGN ESTIMATE
- **2 streams** if face and OCR run continuously — DESIGN ESTIMATE

Slide 5 status line, quoted: *"prototype with preliminary validation on public datasets.
Next: border-specific labelled set, Jetson benchmarking, field evaluation."*
Slide 4 concession, quoted: *"Preliminary rule-sanity checks on sample tracks; not a labelled
evaluation."* Both of these sentences ship with the demo. They are the reason the measured
numbers are credible.

---

## 6. Out of scope for these six days — and the substitute

Blunt: each of the following is not achievable in six days at zero cost, and each will be
named as out of scope on the day of the demo rather than implied to exist.

| Out of scope | Why | What is demonstrated instead |
|---|---|---|
| **Real Jetson Orin Nano benchmarking** | The board is not owned. Slide 3 already labels the throughput figures *"Design estimates, not yet measured on hardware"* and slide 5 lists Jetson benchmarking under *"Next"*. | Inference timed on the Apple Silicon development machine and on a free Colab/Kaggle GPU, reported as **two host-labelled numbers with the host named**, plus the slide 3 estimate quoted as an estimate. No number is attributed to a Jetson. |
| **Real RTSP camera ingest at a border post** | No access to an SSB post or to a camera on a pole. | A recorded video file served as an RTSP stream by a local loopback server, consumed through the same ONVIF/RTSP-shaped ingest path the real deployment would use. The ingest code is identical; only the source differs. Stated aloud in the demo. |
| **TensorRT INT8 quantisation** | TensorRT targets NVIDIA hardware not present; the ~30 ms figure is defined against an INT8 budget that cannot be measured here. | FP16/FP32 inference on available hardware, with the INT8 budget quoted from slide 3 as a **TARGET**. The C++/TensorRT entry on slide 3 stays a stated technology, not a shipped artefact. |
| **734-post scale testing** | No infrastructure, no cost budget, no time. | Six cameras across six sectors in the console (`RXL-01`, `JGB-03`, `PNT-07`, `SNL-01`, `GLG-05`, `PHU-02`), and an architecture statement that each post runs one independent board — so scale is replication, not coordination. Slide 4 already frames it as *"Designed to scale to"*, which is a design claim and stays one. |
| **Real Nepali plate dataset collection** | Requires field access; real plate photos found online are copyrighted and `MEASUREMENTS.md` §4 records them as unusable. | The synthetic plate pipeline already measured in §4, extended to a >= 20-plate synthetic set with degradation (blur, dimming, sensor noise), reported as **synthetic**. The honest statement is the one already in `MEASUREMENTS.md`: fine-tuning on real Nepali plates is required to reach the 85% TARGET. |
| **Field evaluation / operational trial** | Requires MHA and SSB participation. Slide 5 lists it under *"Next"*. | The demo in §7, on public-dataset footage, plus the §5.1 measured table. The word "trial" is not used. |
| **Border-specific labelled dataset** | Does not exist publicly; labelling it is weeks of work. | Public datasets KAIST, LLVIP, IDD, all verified available in `MEASUREMENTS.md`. Fine-tuning is run on those and reported as such. |
| **A labelled evaluation of "suspicious activity"** | Slide 4 already concedes no dataset and no agreed definition exists. | The rule-sanity table from `MEASUREMENTS.md` §2, presented with slide 4's own wording: *"not a labelled evaluation."* |
| **Solar-powered deployment at the 308 road-less posts** | Hardware claim, unverifiable without the board. | Slide 4's economic argument quoted as a design rationale (*"The low-power board can run on solar at the 308 posts with no road"*), not demonstrated. |
| **Daily Merkle root signed at sector HQ, mirrored off-box** | Two-party infrastructure. | Single-node SHA-256 chaining at capture plus a daily root computed locally, with the `verify` command and the tamper test. The signing-and-mirroring step is described, not run. |

---

## 7. Demo definition — the five-minute story

Constrained to routes and panels that **already exist** in the repository, verified against
the source tree:

- Routes in `frontend/src/App.jsx`: `/`, `/login`, `/signup`, `/console` (plus `*` -> `/`).
- Panels rendered by `frontend/src/routes/ConsolePage.jsx`: `ConsoleRail`, `ConsoleTopBar`,
  `LiveWall`, `SectorMap`, `FootageSearch`, `AnalyticsPanel` (left column);
  `AlertQueue`, `EvidenceChainPanel` (right column).

**Two constraints discovered in the code, which the demo must respect:**
- `ConsoleRail` items are inert. Its own comment reads *"Items are inert until their modules
  are built."* **Do not click the rail during the demo.** Everything below is reachable by
  scrolling `/console`, which renders all six panels on one page.
- There is **no face-detection panel and no virtual-fence drawing panel** in the console.
  Capability 3 and capability 5 must surface through other existing panels — see the map below.

### Click-by-click

| t | Action | Route / panel | Capabilities and claims shown |
|---|---|---|---|
| 0:00–0:35 | Land on `/`. Read the hook aloud: open border, thousands cross legally, the problem is selection. Scroll past the capabilities and channels sections. | `/` (`Hero`, `ProblemsSection`, `CapabilitiesSection`, `ChannelsSection`) | Frames §1. Sets up claim (b). |
| 0:35–0:55 | `/login`, sign in as the operator. | `/login` | Roles exist; face unmasking is supervisor-gated (slide 4 privacy note). |
| 0:55–1:35 | Land on `/console`. `LiveWall` shows six cameras across six sectors, IR tiles among them, DORI bands drawn under each tile. Expand one IR tile. | `/console` → `LiveWall` | **Cap 1** (person boxes + tracks), **Cap 2** (vehicle classes), **Cap 7** (IR tile), and the §5.1 §3 small-object / DORI result. |
| 1:35–2:05 | A provisional alert arrives in `AlertQueue`: badge `PROVISIONAL · ~30 ms`, two chips appear in sequence — `APPEARANCE ✓` then `MOTION ✓`. Say aloud: the alert only advances because both fired. | `/console` → `AlertQueue` | **Claim (b)**. **Cap 8** (real-time alert). State ~30 ms is the slide 3 INT8 budget, a TARGET. |
| 2:05–2:40 | The card advances to `CONFIRMED · ~3 s` and the written reason types out, e.g. *"Loaded vehicle moving north at 02:41, outside sanctioned hours, on a route with no recorded night traffic."* Confidence bar fills. | `/console` → `AlertQueue` | **Claim (c)**. **Cap 6** (the reason names the rule). ~3 s is a TARGET. |
| 2:40–3:00 | Point to a second card carrying a Devanagari plate (`बा १२ च ४५६७`) under the camera id. Then show the side-by-side comparison artefact: Latin OCR `"9 9238"` wrong, Devanagari correct at 0.989 / 0.990. | `/console` → `AlertQueue` card subtitle, plus the committed comparison script output | **Cap 4**, **Claim (a)**. Say "synthetic plate" out loud. |
| 3:00–3:20 | Click `ACT · SEAL EVIDENCE`. The card turns to `SEALED · EVIDENCE BLOCK` with its hash. | `/console` → `AlertQueue` | **Cap 8**, start of **Claim (d)**. |
| 3:20–3:50 | In `EvidenceChainPanel`, click `VERIFY CHAIN`. Blocks tick through to `CHAIN INTACT — n / n VERIFIED`. Then run the tamper test in a terminal: alter one byte, `verify` exits non-zero and names the broken block. | `/console` → `EvidenceChainPanel` + terminal | **Claim (d)** completed. **Cap 8**. Cite slide 6 ref 13 if the judge asks about admissibility. |
| 3:50–4:15 | In `SectorMap`, click `Jogbani Ridge` — a post flagged `road: false` and `alert: true`. The wall filters to that sector. | `/console` → `SectorMap` | **Cap 5** context (which post, which line) and the 308-posts-without-road point from slide 4. |
| 4:15–4:40 | In `FootageSearch`, type *"group of more than four people after midnight, Jogbani"*, press SEARCH. Progress runs, result cards return with chips `group / 6 persons / no lamps`. | `/console` → `FootageSearch` | **Cap 3** and **Cap 5** surface here as searchable attributes (faces at gates, fence-line events), and slide 5's *"searches archived footage instead of scrubbing it."* SigLIP per slide 3. |
| 4:40–5:00 | `AnalyticsPanel`: per-camera alert counts, day 38 / night 62 split, channel agreement trending 91 → 98. Close on the honest line: *"prototype with preliminary validation on public datasets. Next: border-specific labelled set, Jetson benchmarking, field evaluation."* | `/console` → `AnalyticsPanel` | **Cap 6**, **Cap 7** (night-weighted), **Cap 8** (event log), **Claim (b)** quantified. Slide 5 status line, quoted verbatim. |

### Coverage check for the demo

| Item | Where in the five minutes |
|---|---|
| Cap 1 Human detection and tracking | 0:55 `LiveWall` |
| Cap 2 Vehicle detection and classification | 0:55 `LiveWall`; 2:40 plate card |
| Cap 3 Face detection | 4:15 `FootageSearch` attributes — **weakest surface, see Open Question 2** |
| Cap 4 ANPR | 2:40 `AlertQueue` + comparison script |
| Cap 5 Virtual fence intrusion | 3:50 `SectorMap`; 2:05 alert reason text — **no drawing UI exists, see Open Question 2** |
| Cap 6 Suspicious activity | 2:05 alert reason; 4:40 `AnalyticsPanel` |
| Cap 7 Night-time movement | 0:55 IR tile; 4:40 day/night split |
| Cap 8 Real-time alerts and event logging | 1:35, 3:00, 3:20 |
| Claim (a) Devanagari plates | 2:40 |
| Claim (b) Two independent channels | 1:35 chips; 4:40 agreement chart |
| Claim (c) Written reason on every alert | 2:05 |
| Claim (d) Chained and signed at capture | 3:00 + 3:20 |

---

## 8. Risk register

| # | Risk | Likelihood | Impact | Exact fallback |
|---|---|---|---|---|
| 1 | **Ultralytics YOLO11 is AGPL-3.0.** A public repository that imports it inherits AGPL obligations, including source availability for network use. The repo `github.com/0XSreekar/True-Watch-AI` is public. | High | High — a licence problem on the artefact a judge is pointed at | Decide before any YOLO code is committed. Option A: license the whole repo AGPL-3.0 and add the notice — simplest, costs nothing, and is the default assumption unless told otherwise. Option B: isolate all Ultralytics-importing code in a separate AGPL-licensed repo or submodule and keep the console repo untouched. Option C: swap the detector for a permissively licensed one, which invalidates every YOLO11-s number in `MEASUREMENTS.md` and is therefore the last resort. **This is not a PDF fact — verify the current licence text at source before deciding.** |
| 2 | **Fine-tuning does not reach the slide 5 targets** (0.85 day / 0.75 IR person mAP@50) in the free GPU hours available | High | Medium | Ship the measured numbers from `MEASUREMENTS.md` §1 and present the slide 5 figures as TARGET, exactly as slide 5 itself labels them. The deck is already honest here; do not make it less so under pressure. |
| 3 | **Free GPU quota exhausted mid-training** (Kaggle/Colab) | Medium | Medium | Checkpoint every epoch to a persistent location; keep the pretrained-only results as the shipping baseline so a failed run costs nothing that is already promised. §5.1 is entirely pretrained-only and survives this. |
| 4 | **Moondream 2 explanation latency or quality is unusable** on free-tier hosting | Medium | High — claim (c) depends on it | Generate the reason from a constrained template populated by the rule engine and the detection fields, and say plainly that the vision-language model is the next step. Claim (c) stays true — the alert still carries a written reason — but the artefact must be described accurately, not as VLM output. |
| 5 | **Free hosting cold-starts or times out during the live demo** (Hugging Face Spaces / Render free tier) | High | High | Run the entire demo from `localhost` on the development machine, with hosted deployment as a link the judge may open afterwards. Pre-warm the hosted instance anyway. Never depend on a network round-trip during the five minutes. |
| 6 | **RTSP loopback ingest does not work reliably** on Apple Silicon | Medium | Medium | Fall back to direct file ingest through the same decode path and state that the RTSP source is substituted, per §6. The ingest interface stays unchanged, so the substitution is a one-line change and visibly so. |
| 7 | **PP-OCRv5 Devanagari fails on anything but the single measured synthetic plate** | Medium | Medium — claim (a) narrows | Claim (a) is a *comparison* claim, and the comparison holds even at low absolute accuracy: Latin OCR returns a wrong string, Devanagari returns a correct one. Report read accuracy on the 20-plate synthetic set as-is and carry the `MEASUREMENTS.md` §4 real-plate caveat verbatim. |
| 8 | **Six days is not enough to wire all eight capabilities end-to-end**, leaving some visible only as fixtures in the console | High | High — a judge who finds mock data behind a panel discounts everything | Rank the capabilities now and build downward: 1, 7, 6, 5, 8 first (all five are already backed by measured results in `MEASUREMENTS.md`), then 4, then 2, then 3. Any capability still on fixtures at the end is **labelled in the UI as sample data** — `backend/src/data/mockData.js` already carries the header *"Everything in this file is sample data."* Keep that honesty visible rather than hiding it. |

---

## Open questions for Sreekar

1. **Capability 2 lists "cart" as a vehicle class.** YOLO11-s pretrained on COCO has no cart
   class. Slide 3 says fine-tuning is planned on IDD. Does IDD Detection contain a cart class
   usable for this, or does "cart" need its own labelled data? If neither, capability 2 ships
   as three classes and the deck's fourth class needs a note.
2. **Two capabilities have no console surface.** There is no face-detection panel and no
   virtual-fence drawing UI in `frontend/src/components/console/`. `CONSOLE_NAV` lists
   "Virtual fences" but `ConsoleRail` marks its items inert. Options: (i) surface both only
   through `AlertQueue` reason text and `FootageSearch`, as §7 currently does; (ii) build a
   minimal fence-drawing overlay on `LiveWall` — roughly half a day; (iii) accept the gap and
   say so. **§7 assumes option (i).** Confirm before Phase 0.
3. **Which border does the demo claim?** Slide 4 says Devanagari applies on the Indo-Nepal
   side and Bhutan plates are Latin (BP-N-ANNNN), already covered by standard ANPR. The mock
   camera list includes `PHU-02 Phuentsholing Line — Post 9`, which is Indo-Bhutan. Confirm
   the demo does not show a Devanagari plate on a Bhutan camera.
4. **Repository licence.** See risk 1. A decision is needed before the first commit that
   imports Ultralytics.
5. **Where do the fine-tuning artefacts live?** Free GPU notebooks produce weights that exceed
   normal Git limits. Confirm whether weights go to a Hugging Face model repo (free) and are
   referenced by id, rather than committed.
6. **The `192 camera-hours` and `under 40 alerts` figures** on slide 5 are stated per control
   room officer with 8 cameras. The console ships 6 cameras. Confirm whether the demo restates
   the slide figure or the console figure — they must not be conflated.

---

## Second-opinion run

A second-opinion pass against an independent model, and a diff of the two outputs, is a manual
step. It has not been performed inside this session. Run it before merging, and treat any
contradiction as an addition to Open Questions rather than a silent edit to this file.

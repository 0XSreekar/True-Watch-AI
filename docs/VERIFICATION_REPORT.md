# Verification report

Status of every checkable claim, measured on 2026-09-29/30 on the integration branch. Claim wording follows the
build plan's quotation of the submitted slides (the slide PDF is not in this repository). A verdict is PASS only when
a command or test below produced it on this machine (Apple M5 MacBook Air, 16 GB, macOS). Targets are never reported
as measurements.

Verdicts: **PASS**, **PARTIAL**, **FAIL** (target not met), **NOT MEASURED** (possible, not done yet),
**NOT TESTABLE HERE** (needs hardware or field data this project does not have).

## 1. Claims

| Slide | Claim | Where | How verified | Verdict | Evidence |
|---|---|---|---|---|---|
| 2 | Human detection and tracking | `edge/pipeline/appearance.py`, `tracking.py` | `tests/test_tracking.py`; end-to-end run (section 3) | PASS | 10 fused events on the day sample, tracks stable across the run |
| 2 | Vehicle classification: truck, car, two-wheeler, cart | detector classes, `training/results/METRICS.md` | per-class eval | PARTIAL | truck/car/two-wheeler trained and evaluated; **cart has no training data and is dropped at inference** (`fabf183`) |
| 2 | Face detection (YuNet), close range | `edge/face/detect.py`, `face/stage.py` | `tests/test_face_stage.py`, `test_face_privacy.py` | PARTIAL | wired behind `FACE_ENABLED` (default off, personal data); YuNet 1.4-2.7 ms per head crop; the sample clips contain no close-range faces, so detection quality is unit-tested only |
| 2 | ANPR, Nepali Devanagari | `edge/anpr/` | `anpr.evaluate` on held-out real photographs | FAIL vs 85% | see section 2 |
| 2 | Virtual fence intrusion | `edge/rules/fence.py` | `tests/test_rules.py` (segment intersection) | PASS (unit) | miss rate over scripted crossings NOT MEASURED |
| 2 | Suspicious activity: named rules + per-camera baseline | `edge/rules/` | `python -m tools.reproduce_rule_sanity` | PASS | MEASUREMENTS.md s2 table reproduces exactly: 10 tracks, 6 alerts, 4 normal; track 6 = fence + loitering |
| 2 | Night-time operation (IR) | detector IR eval, `pipeline/fusion.py` night weights | METRICS.md; `test_fusion.py` | PASS | IR person AP50 0.845 (target 0.75) |
| 2 | Real-time alerts and event logging | `pipeline/service.py`, `events.py`, backend ingest + SSE | end-to-end run; `backend/tests/stream.test.js` | PASS | 10 events -> 10 alerts, 30/30 messages delivered, SSE created + reason patch tested |
| 2 | (a) Devanagari plates read where Latin ANPR reads nothing | `edge/anpr/compare_latin.py` | `tests/test_anpr.py::test_measurements_section_4_reproduces` | PASS | Devanagari head reads the synthetic plate; the Latin head does not (0.00% on the 2,076 synthetic val plates) |
| 2 | (b) Two channels on different principles, alert only when both agree | `edge/pipeline/fusion.py` | `tests/test_fusion.py` | PASS | appearance-only, motion-only and below-temporal-gate cases do not alert; agreement does |
| 2 | (c) A written reason on every alert | `pipeline/explain.py`, `edge/vlm/` | `tests/test_live_explain.py`; end-to-end run | PASS | every alert arrived with the rule sentence; model sentences patch it later; reason never empty in any failure path tested (22 cases) |
| 2 | (d) Evidence chained and signed as it is recorded | `edge/evidence/`, `pipeline/sealing.py` | `tests/test_evidence.py` (31), `test_sealing.py`; end-to-end run | PASS | 10 clips sealed at capture in the live run; verify 10/10 intact; one flipped byte in block 3 -> blocks 3-10 invalid, verify exit 1 |
| 3 | Seven-step flow camera -> channels -> agreement -> alert -> reason -> seal -> log | edge + backend | end-to-end run | PASS | section 3 |
| 3 | ~30 ms detector inference (YOLO11-s INT8, 640 px) | - | - | NOT TESTABLE HERE | Jetson Orin Nano target; CPU numbers in METRICS.md are not comparable |
| 3 | VLM reason ~3 s, asynchronous | `edge/vlm/`, `pipeline/explain.py` | `vlm/latency.py`; `test_live_explain.py` | PARTIAL | async ordering proven; Mac Apple GPU (MPS) p50 2.93 s / p95 4.11 s (n=24); Mac CPU p50 272.8 s / p95 309.9 s (n=6); Jetson not measured. Guardrails accepted 7 of 24 model sentences; the rest kept the rule sentence |
| 3 | Verifiable with one command | `python -m evidence.verify` | CLI run | PASS | exit 0 clean, exit 1 after tamper |
| 3 | SHA-256 per clip, chained, daily Merkle root, signed, mirrored to HQ | `edge/evidence/` | `tests/test_evidence.py` | PASS | Ed25519 key 0600; HQ mirror is a second local directory; no network import in `edge/evidence/` |
| 4 | Existing IP cameras (RTSP/ONVIF) | `edge/pipeline/source.py`, `ingest.py` | file replay only | PARTIAL | RTSP path implemented; only recorded files were used as sources |
| 4 | Board failure: chain resumes, gap logged | `evidence/gaps.py`, `store.py` | `test_evidence.py` (real SIGKILL) | PASS | exactly one gap logged after a killed writer |
| 4 | Face data stays on post, 30-day retention, access-controlled unsealing | `edge/face/privacy.py`, backend `/api/faces/:id/unseal` | `test_face_privacy.py`; backend role tests | PASS | Supervisor-only, audited; 31-day crop purged |
| 4 | Dismissal updates the camera baseline | `edge/rules/baseline.py`, edge `POST /pipeline/feedback`, backend dismiss | `test_rules.py::test_dismissal_measurably_shifts_the_baseline`, `test_service_rules.py::test_dismissal_through_the_feedback_route_...` | PASS | the route releases the throttle, folds every fired rule into the baseline and keeps it across the periodic relearn |
| 4 | Bhutan Latin plates | `anpr/postprocess.py` grammar, pretrained Latin head | `tests/test_anpr.py` | PARTIAL | grammar parses; embossed real plates read 57.1% exact (28 plates) with the pretrained Latin head |
| 4 | Tampering raised as camera-state anomaly | `pipeline/camera_state.py` | `test_fusion.py::test_camera_state_stands_fusion_down`, pipeline tests | PASS (synthetic) | black frame / covered lens raise camera-state events |
| 4 | IR replicated to 3 channels | `datasets/scripts/04_ir_to_3ch.py` | dataset gates | PASS | Phase 1 gates 17/17 |
| 4 | Motion weighted higher after dark (1.9x -> 3.4x contrast) | `pipeline/fusion.py` | `tools/reproduce_flow_contrast.py` on LLVIP | PARTIAL | direction reproduced on an independent LLVIP sequence (38 pairs): visible 2.92x, LWIR 4.75x. The published 1.9x/3.4x came from KAIST, which is not downloaded here |
| 4 | SAHI tiling of the far-field strip | `pipeline/appearance.py` | unit tests | PARTIAL | implemented and off by default; no far-field accuracy measured |
| 4 | DORI grading per IEC 62676-4 | `edge/rules/dori.py` | `test_rules.py` | PASS | bands exactly 25/62/125/250 px/m; refuses claims beyond the detect band |
| 5 | Person mAP@50 0.85 day | detector v2 | `training/results/METRICS.md` | **FAIL** | 0.549 (95% CI 0.495-0.610) |
| 5 | Person mAP@50 0.75 IR | detector v2 | METRICS.md | PASS | 0.845 (95% CI 0.782-0.876) |
| 5 | <= 5 false alerts per camera per day | throttle + fusion | `tools/measure_false_alerts.py` (self-check only) | NOT MEASURED | rate limiter proven (`test_rules.py`: 50 repeats -> within budget); harness built and proven on generated footage; no event-free footage exists yet |
| 5 | Detection range 150 m at 25 px/m | `rules/dori.py` | - | NOT TESTABLE HERE | needs a calibrated real camera |
| 5 | 85% Nepali plates under 25 m | ANPR | section 2 | **FAIL** | best real result 52.6% (57 plates); no range-labelled real set |
| 5 | < 2% missed fence crossings | fence rule | `tools/measure_fence_misses.py` (self-check only) | NOT MEASURED | harness proven on 50 generated crossings (counts 2 deliberately hidden ones as missed); needs recorded crossings with ground-truth times |

## 2. Plate reader (published model, round G6, HF `sreekar12/truewatch-anpr-devanagari@064b6a0`)

| Set | Plates | Exact-match | Mean CER |
|---|---|---|---|
| Real, strict (independent, double-transcribed) | 57 | 52.6% | 0.262 |
| Real, extended | 147 | 38.8% | 0.304 |
| Roboflow held-out (after removing 94 test plates whose text repeats a training vehicle) | 114 | 72.8% | 0.105 |
| Synthetic validation | 2,076 | 88.0% | 0.052 |

The synthetic figure is not evidence about real plates. Full history: `edge/anpr/results/ANPR_METRICS.md`.

## 3. End-to-end run (edge + backend, shipped defaults)

`samples/day.mp4` (640x512, 20 fps) replayed through the edge service (v2 detector ONNX, motion, tracking, fusion,
rules, evidence sealing; VLM off for this run) delivering to the backend with a real ingest key:
- 1,155 frames, 10 events, 10 clips sealed at capture (0 failures), 30 of 30 messages delivered (0 rejected).
- Backend: 10 alerts in the frozen shape, each with the edge's rule sentence and the sealed clip's hash;
  `GET /api/evidence/chain` served from the edge chain; `POST /api/evidence/verify` 10/10 intact.
- Tamper: one byte flipped in block 3's clip -> blocks 3-10 invalid, `evidence.verify` exit 1.

Frame rate: 1080p 16.5-17.9 FPS with the detector on every second frame, 640x512 about 30 FPS, thermal 640x512
17.7 FPS (VLM off; `edge/README.md`). With the VLM on the Apple GPU the thermal run dropped to 7.7-9.7 FPS
because the model shares the chip.

Memory soak (30 min, day sample looped, VLM off, evidence on): 40,460 frames, 372 events and sealed clips;
resident memory 448 MB at minute 1 and 468 MB at minute 30 (+4.5%, about 0.7 MB/min, tracking the event count);
open file handles flat at 188-189. No handle leak; the slow memory rise should be re-checked over several hours.

## 4. Defects found and fixed during verification

| Defect | Severity | Fix |
|---|---|---|
| Live pipeline used the pretrained plate model unless an env var was set by hand | P1 | `2abbe17` pinned, SHA-256-checked download of the published model |
| Roboflow test set leaked 94/208 vehicles into training | P1 (metric) | reported on the clean 114 only (`5e51c61`) |
| Embossed Latin fine-tune reads real plates worse end to end (21.4% vs 57.1%) | P1 (would have regressed) | not adopted (`c02ddcf`) |
| Unpinned `paddleocr` resolves 2.x: plate reading breaks and `edge/tools` is shadowed | P0 on a fresh install | pinned + `edge/requirements.lock` (`4289d9f`) |
| Backend validator dropped `nullable` when chained with `optional`: every edge template reason patch refused (400) | P0 | `f0ea645` + regression test |
| Two evidence chains (backend digest differs from edge canonical block hash); live pipeline never sealed clips | P1 | `d7cf23e` edge seals at capture; backend reads the edge chain first |
| Default `TARGET_FPS=8` decimated 20 fps cameras; fusion then emitted no alerts at all (0 at 8 and 10 fps, 8 at native rate) | P0 | fixed: fusion is frame-rate independent (section 4.1); default back to 8 |
| A track whose sampled ground point lands exactly on a fence was never counted as crossing (strict signs on both path segments) | P1 | found by `tools/measure_fence_misses.py` on scripted footage (25 of 50 crossings missed); on-line points are bridged, `test_rules.py` regression |
| Backend dismiss feedback posted to an edge route that did not exist; a baseline relearn would also have erased dismissals | P1 | edge `/pipeline/feedback` + dismissal replay after relearn |
| Detector metrics named the wrong owning phases | P2 | `bb6a54e` |

### 4.1 Frame-rate invariance, measured

Cause (measured): flow was scored in px per frame and the calibration warm-up was 100 frames, so at 10 fps the
learnt thresholds came from 10 s of scene instead of 5 s and in different units, and object energy over threshold
fell (median E_in/tau 0.57 at 20 fps, 0.40 at 10 fps). Fix: flow is scaled to px per 1/20 s (the section 1 frame
interval; factor 1 at 20 fps, so the 1.9x/3.4x figures, the day/night weights and the Farneback parameters are
unchanged), the warm-up is 5 s of the camera's clock (floor 25 frames), a file is decimated and timed on its own
media clock, and the detector runs at least every 0.125 s (at 8 fps a stride of 2 left boxes 0.25 s apart, too far
for the tracker to associate a fast walker: 0 of 6 scripted crossings tracked). The spatial gate held under the
larger per-frame displacement (S column). Temporal gate and appearance freshness stay in frames.

`Pipeline` over `samples/day.mp4` and `samples/ir.mp4` (640x512, 20 fps, v2 detector, stride 2), decimated by media
time. Gate columns are the share of post-warm-up candidate-frames passing each gate; fused = share of post-warm-up
frames with an agreeing candidate.

| Clip | fps | Before: warm-up frames | M | S | agree | alerts | After: warm-up frames (s) | M | S | agree | fused | alerts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| day | 20 | 100 | 8.3% | 27.8% | 7.7% | 4 | 101 (5.0) | 8.2% | 27.0% | 7.5% | 14.7% | 4 |
| day | 10 | 100 | 1.6% | 11.5% | 1.6% | 0 | 51 (5.0) | 8.2% | 30.0% | 6.8% | 13.8% | 3 |
| day | 8 | 100 | 0.0% | 4.8% | 0.0% | 0 | 41 (5.0) | 8.5% | 38.1% | 7.5% | 13.5% | 4 |
| ir | 20 | 100 | 71.9% | 93.9% | 39.4% | 7 | 101 (5.0) | 72.0% | 93.9% | 39.1% | 26.0% | 7 |
| ir | 10 | 100 | 51.9% | 88.5% | 23.7% | 3 | 51 (5.0) | 72.4% | 92.3% | 40.7% | 28.1% | 6 |
| ir | 8 | 100 | 63.5% | 89.6% | 46.9% | 4 | 41 (5.0) | 74.0% | 90.7% | 44.2% | 28.6% | 7 |

Service path (`pipeline.ingest` at TARGET_FPS -> `CameraService`, shipped defaults), events emitted:

| Clip | TARGET_FPS 25 (native) before / after | TARGET_FPS 10 before / after | TARGET_FPS 8 before / after |
|---|---|---|---|
| day | 4 / 4 | 0 / 3 | 0 / 4 |
| ir | 7 / 7 | 3 / 6 | 3 / 7 |

("Before" at TARGET_FPS 8 actually ran at 10 fps: the old decimation kept every round(20/8) = 2nd frame.)
Compute per second of footage on this machine (service path, unpaced): day 0.77 s at native, 0.52 s at 10 fps,
0.43 s at 8 fps; ir 0.82 / 0.59 / 0.50 s. `TARGET_FPS` default is therefore back to 8. Caveat: two 22 s clips; the
10 fps day run loses one of two alerts raised 0.2 s apart at native rate. Tests: `tests/test_frame_rate.py` runs the
same synthetic scene at 20 and 8 fps (four fusion cases, full pipeline with calibration) and asserts the same verdict.

## 5. Open items (honest remaining work)

- P1: false-alert rate per camera per 24 h and fence-crossing miss rate are still NOT MEASURED: no event-free
  footage and no recorded crossings exist. The tools are built and proven on generated footage with a scripted
  detector (`edge/README.md`, "Measuring the slide-5 field targets"):
  - `python -m tools.measure_false_alerts --self-check` (8 fps): two 60 s event-free recordings (textured ground,
    noise, swaying foliage, a parked box at 0.9) -> 0 alerts in 0.031 alertable camera-hours, reported as 0 per
    camera per 24 h with a 95% upper bound of 2,903 (the interval shows how little 2 minutes says); a positive
    control with 3 scripted walks -> 3 alerts.
  - `python -m tools.measure_fence_misses --self-check 50`: 50 scripted crossings, 2 of them hidden from the
    detector -> 48 detected, 2 missed, miss rate 4.0%, Wilson 95% 1.1%-13.5%, verdict "not demonstrated"; the same
    at 20 fps and 8 fps, 0 fence alerts without a true crossing. Showing < 2% needs n >= 189 crossings with no miss.
  Real numbers need someone to record event-free hours per camera and a scripted-crossing session with the times
  written down.
- P2: on fusion-only events of the scripted footage the rule sentence fails its own guardrail and the Phase 4 string
  is used (`truewatch.vlm` warning "rule sentence failed its own guardrail"); seen, not investigated.
- P2: `VITE_DEMO_MODE` (default on) opens the console on sample data when the API is unreachable; set it off for any real use.
- P2: backend's local fallback Merkle root duplicates an odd leaf; the authoritative edge root promotes it.
- Not built: trained plate locator (classical locator found 3 of 13 real plates earlier), cart class, SAHI accuracy on far field.
- Not testable here: Jetson latency, 150 m range, field false-alert rate.
- Deployment: out of scope by the owner's decision; the system runs locally.

## 6. Test suites

| Suite | Result |
|---|---|
| Edge (`~/.venvs/truewatch-edge`, one process) | 796 passed, 9 skipped (`onnx` package not installed: detector-export checks) |
| Evidence smoke (`scripts/evidence-check.sh`) | clean exit 0, tampered exit 1 |
| Backend (`npm test`) | 113 passed |
| Frontend (`node --test frontend/tests`) | 23 passed; build CSS byte-identical |
| Training | 288 passed, 56 skipped |
| Watermark scan (`scripts/watermark-scan.sh`) | clean |
| Lint (ruff, pyflakes rules) | 0 undefined names; 18 unused imports / empty f-strings removed; style-rule findings (`zip` strictness, `raise from`) left as P2 |

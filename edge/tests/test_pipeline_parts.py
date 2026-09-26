"""Frame source reconnect, camera-state anomalies, per-camera calibration and the appearance class map."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pipeline.appearance import AppearanceChannel, build_class_map, class_nms, letterbox, tile_windows, TilingConfig
from pipeline.calibration import CalibrationStore, Calibrator, learn_camera_thresholds
from pipeline.camera_state import CameraStateMonitor, frame_stats, thumbnail
from pipeline.source import FrameSource
from pipeline.types import SCHEMA_CLASSES
from tests.synth import texture

# ---------------------------------------------------------------------------------------------- source


class FakeCapture:
    """Scripted cv2.VideoCapture stand-in: `plan` is how many frames each successive open yields
    (None = the open fails)."""

    plan: list = []
    opened = 0

    def __init__(self, *args, **kwargs):
        FakeCapture.opened += 1
        self.frames = FakeCapture.plan.pop(0) if FakeCapture.plan else 0

    def isOpened(self):
        return self.frames is not None

    def set(self, *a):
        return True

    def get(self, prop):
        return 25.0

    def read(self):
        if not self.frames:
            return False, None
        self.frames -= 1
        return True, np.zeros((8, 8, 3), np.uint8)

    def release(self):
        pass


def test_rtsp_source_reconnects_with_backoff_and_reports_the_gap():
    FakeCapture.plan, FakeCapture.opened = [3, None, None, 2], 0
    clock = {"t": 100.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        clock["t"] += s

    def now():
        clock["t"] += 0.01
        return clock["t"]

    src = FrameSource("rtsp://127.0.0.1:8554/cam", camera_id="C1", max_frames=5, capture_factory=FakeCapture,
                      sleep=sleep, clock=now)
    frames = list(src)
    assert [f.index for f in frames] == [0, 1, 2, 3, 4]
    assert all(f.source == "rtsp" for f in frames)
    assert sleeps == [0.5, 1.0], "exponential backoff while the camera is down"
    assert frames[3].reconnects == 3 and frames[3].gap_s >= 1.5
    assert frames[4].gap_s == 0.0
    ts = [f.timestamp for f in frames]
    assert ts == sorted(ts) and len(set(ts)) == len(ts), "timestamps are strictly monotonic"


def test_file_source_uses_media_time(tmp_path):
    path = tmp_path / "c.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for i in range(12):
        w.write(np.full((24, 32, 3), i * 10, np.uint8))
    w.release()
    frames = list(FrameSource(str(path)))
    assert len(frames) == 12 and frames[0].source == "file"
    assert frames[-1].timestamp == pytest.approx(1.1, abs=0.05), "media time, not wall-clock time"


# ---------------------------------------------------------------------------------------------- camera state


def _scene(seed=3):
    g = np.clip(texture(240, 320, seed, amp=50, base=110, blur=1.5), 0, 255).astype(np.uint8)
    return cv2.merge([g, g, g])


def _baseline(mon: CameraStateMonitor, img):
    s = frame_stats(thumbnail(img))
    mon.set_baseline(s["brightness"], s["std"], s["detail"])


def _feed(mon, imgs, start=0):
    events = []
    for i, img in enumerate(imgs):
        events += mon.update(img, frame_index=start + i, timestamp=(start + i) / 20, captured_at=0.0)
    return events


def test_black_frame_raises_signal_loss_and_clears():
    mon = CameraStateMonitor("C1")
    scene = _scene()
    _baseline(mon, scene)
    rng = np.random.default_rng(0)
    live = lambda: np.clip(scene.astype(np.int16) + rng.integers(-2, 3, scene.shape), 0, 255).astype(np.uint8)
    ev = _feed(mon, [live() for _ in range(5)])
    assert ev == [] and not mon.blocked
    ev = _feed(mon, [np.zeros_like(scene)] * 4, start=5)
    assert [(e.anomaly, e.active) for e in ev] == [("SIGNAL_LOSS", True)]
    assert mon.blocked and ev[0].kind == "CAMERA_STATE"
    ev = _feed(mon, [live() for _ in range(6)], start=9)
    assert [(e.anomaly, e.active) for e in ev] == [("SIGNAL_LOSS", False)]
    assert not mon.blocked


def test_covered_lens_is_tamper_but_a_darker_scene_is_not():
    mon = CameraStateMonitor("C1")
    scene = _scene()
    _baseline(mon, scene)
    dusk = (scene * 0.3).astype(np.uint8)                        # the same scene, much darker
    assert _feed(mon, [dusk] * 6) == []
    rng = np.random.default_rng(1)
    covered = [np.clip(cv2.GaussianBlur(scene, (61, 61), 0) * 0.3 + rng.normal(0, 2, scene.shape), 0, 255)
               .astype(np.uint8) for _ in range(4)]                  # a hand over the lens: blur, dark, sensor noise
    ev = _feed(mon, covered, start=6)
    assert [(e.anomaly, e.active) for e in ev] == [("LENS_TAMPER", True)]


def test_ir_floodlight_is_raised_as_flood_not_darkness():
    mon = CameraStateMonitor("C1")
    scene = _scene()
    _baseline(mon, scene)
    flooded = np.clip(scene.astype(np.float32) * 0.15 + 235, 0, 255).astype(np.uint8)
    ev = _feed(mon, [flooded] * 4)
    assert [(e.anomaly, e.active) for e in ev] == [("IR_FLOOD", True)]


def test_frozen_picture_and_outage_gap_are_signal_loss():
    mon = CameraStateMonitor("C1", freeze_s=0.5)
    scene = _scene()
    ev = _feed(mon, [scene] * 16)                                 # bit-identical frames for 0.8 s
    assert ("SIGNAL_LOSS", True) in [(e.anomaly, e.active) for e in ev]
    mon2 = CameraStateMonitor("C2")
    ev = mon2.update(scene, frame_index=0, timestamp=10.0, captured_at=0.0, gap_s=4.0, reconnects=2)
    assert [(e.anomaly, e.active) for e in ev] == [("SIGNAL_LOSS", True), ("SIGNAL_LOSS", False)]
    assert "4.0 s" in ev[0].detail


# ---------------------------------------------------------------------------------------------- calibration


def test_calibration_learns_a_higher_threshold_over_foliage(tmp_path):
    rng = np.random.default_rng(0)
    gy, gx = 4, 5
    cells, bgs = [], []
    for _ in range(100):
        c = np.abs(rng.normal(0.1, 0.03, (gy, gx))).astype(np.float32)
        c[0, 0] = abs(rng.normal(2.0, 0.6))                      # swaying foliage in one cell
        c[2, 3] = 5.0 if rng.random() < 0.08 else 0.1            # a person crossing now and then
        cells.append(c)
        bgs.append(np.ones((gy, gx), bool))
    cal = learn_camera_thresholds("C1", "visible", (160, 128), cells, bgs, [{"brightness": 90, "std": 40, "detail": 0.1}], 0.55)
    tau = cal.tau_cells()
    assert tau[0, 0] > 2.0, "foliage cell learns a high threshold"
    assert tau[1, 1] < 0.6, "quiet cell stays sensitive"
    assert tau[2, 3] < 1.0, "an occasional passer-by does not raise the threshold (robust statistic)"

    store = CalibrationStore(tmp_path)
    path = store.save(cal)
    assert path.name == "C1.visible.json" and json.loads(path.read_text())["schema"] == "truewatch.calibration.v1"
    again = store.load("C1", "visible", (160, 128))
    assert again is not None and np.allclose(again.tau_cells(), tau)
    assert store.load("C1", "visible", (320, 256)) is None, "a different working size re-learns"
    assert store.reset("C1") == [path] and store.load("C1", "visible") is None


def test_calibrator_excludes_candidate_boxes_from_the_background():
    cal = Calibrator("C1", "visible", 0.55, warmup_frames=20)
    for _ in range(20):
        cells = np.full((4, 4), 0.1, np.float32)
        cells[1, 1] = 6.0                                        # a detected walker sits in cell (1, 1)
        cal.observe(cells, (128, 128), 32, [(32, 32, 63, 63)], {"brightness": 90, "std": 40, "detail": 0.1})
    assert cal.ready
    tau = cal.finalize().tau_cells()
    assert tau[1, 1] < 1.0, "a candidate's own motion must not become background"


# ---------------------------------------------------------------------------------------------- appearance


COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]


def test_class_map_is_by_name():
    cmap, schema = build_class_map(COCO)
    assert schema == list(SCHEMA_CLASSES)
    named = {COCO[i]: schema[j] for i, j in enumerate(cmap) if j >= 0}
    assert named == {"person": "person", "bicycle": "two_wheeler", "motorcycle": "two_wheeler", "car": "car",
                     "bus": "truck", "truck": "truck"}
    fine = ["person", "two_wheeler", "car", "truck", "cart"]
    cmap2, _ = build_class_map(fine)
    assert list(cmap2) == [0, 1, 2, 3, 4]
    cmap3, _ = build_class_map(fine, drop=("cart",))
    assert cmap3[4] == -1, "a withdrawn cart class is ignored"


class FakeSession:
    """A graph with 3 model classes (person, bus, truck) and 4 anchors, output (1, 7, 4)."""

    class _Meta:
        custom_metadata_map = {"names": "{0: 'person', 1: 'bus', 2: 'truck'}"}

    class _In:
        name = "images"
        shape = ["batch", 3, "height", "width"]

    def get_inputs(self):
        return [self._In()]

    def get_modelmeta(self):
        return self._Meta()

    def run(self, _, feeds):
        n = feeds["images"].shape[0]
        out = np.zeros((n, 7, 4), np.float32)
        out[:, :4, 0] = np.array([100, 300, 40, 80], np.float32)[None, :]   # a person, 0.9
        out[:, 4, 0] = 0.9
        out[:, :4, 1] = np.array([300, 300, 100, 60], np.float32)[None, :]  # scored bus 0.6 AND truck 0.7
        out[:, 5, 1], out[:, 6, 1] = 0.6, 0.7
        out[:, :4, 2] = np.array([302, 301, 100, 60], np.float32)[None, :]  # a near-duplicate: NMS removes it
        out[:, 6, 2] = 0.5
        out[:, :4, 3] = np.array([500, 300, 20, 20], np.float32)[None, :]
        out[:, 4, 3] = 0.05                                                  # below the candidate floor
        return [out]


def test_appearance_decodes_maps_and_unletterboxes():
    ch = AppearanceChannel(FakeSession(), imgsz=640)
    img = np.zeros((320, 640, 3), np.uint8)          # letterboxed to 640x640: scale 1, 160 px of padding on top
    dets = ch.detect(img, camera_id="C1", frame_index=7, timestamp=0.35)
    assert len(dets) == 2
    by = {d.class_name: d for d in dets}
    assert set(by) == {"person", "truck"}
    assert by["truck"].score == pytest.approx(0.7), "bus and truck merge into one truck box by max"
    assert by["person"].bbox == pytest.approx((80, 100, 120, 180))
    assert by["truck"].bbox == pytest.approx((250, 110, 350, 170))
    assert by["person"].frame_index == 7 and by["person"].camera_id == "C1"


def test_letterbox_and_nms_helpers():
    img = np.zeros((360, 640, 3), np.uint8)
    out, r, (px, py) = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and r == 1.0 and px == 0 and py == 140
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], np.float32)
    keep = class_nms(boxes, np.array([0.9, 0.8, 0.7], np.float32), np.array([0, 0, 1]), 0.5, 10)
    assert sorted(keep.tolist()) == [0, 2]
    wins = tile_windows(1920, 1080, TilingConfig(enabled=True, band=(0.0, 0.55), tile=640, overlap=0.2))
    assert wins and all(x2 - x1 == 594 and y2 <= 594 for x1, y1, x2, y2 in wins)
    assert max(x2 for _, _, x2, _ in wins) == 1920


def test_the_untrained_cart_column_is_dropped_by_default():
    """v2 was trained with 0 cart instances (cart gate withdrew class 4); its column must never map."""
    from pipeline.appearance import AppearanceConfig, build_class_map

    assert AppearanceConfig().drop_classes == ("cart",)
    cmap, schema = build_class_map(["person", "two_wheeler", "car", "truck", "cart"], AppearanceConfig().drop_classes)
    assert list(cmap) == [0, 1, 2, 3, -1] and schema[4] == "cart"

"""Fusion means the same thing at any frame rate (docs/VERIFICATION_REPORT.md section 5).

The same synthetic scene, the same physical motion (px per SECOND), rendered at 20 fps and at 8 fps:
the motion channel reports the same px-per-reference-interval energy, calibration warms up over the same
seconds, and fusion reaches the same agree / alert verdict. Before the fix the warm-up was counted in
frames and flow in px per frame, so the sample clips decimated to 8-10 fps raised no alert at all.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipeline.calibration import Calibrator
from pipeline.fusion import Fusion, FusionConfig
from pipeline.ingest import frames
from pipeline.motion import REFERENCE_INTERVAL_S, MotionChannel
from pipeline.runner import Pipeline, PipelineConfig
from pipeline.source import Frame
from pipeline.types import Detection
from tests.synth import Scene, make_track

SPEED_PX_S = 60.0          # the walker's speed on the image, px per second (3 px/frame at 20 fps)
CFG = FusionConfig(threshold=0.55, n_consecutive=3, appearance_max_age=1)


def scene_at(fps: float, **kw) -> Scene:
    """The same scene, with the walker's per-frame step set so that it moves SPEED_PX_S at `fps`."""
    return Scene(speed=SPEED_PX_S / fps, **kw)


def fuse(fps: float, seconds: float, box_of, score: float, moving: bool = True):
    scene = scene_at(fps)
    mc, fusion = MotionChannel(), Fusion("TEST", "visible", CFG)
    decisions, events, energy = [], [], []
    for k in range(int(round(seconds * fps))):
        t = k / fps
        field = mc.update(scene.frame(k, None if moving else 0), t)
        tr = make_track(1, box_of(scene, k), "person", k, t)
        out = fusion.evaluate([tr], field, appearance_of={1: (score, k)}, frame_index=k, timestamp=t,
                              captured_at=0.0, frame_size=(320, 256))
        decisions += out.decisions
        events += out.events
        if field is not None:
            x1, y1, x2, y2 = (int(v) for v in scene.obj_box(k))
            energy.append(float(field.mag[y1:y2, x1:x2].mean()))
    return decisions, events, energy


@pytest.mark.parametrize("fps", [20.0, 10.0, 8.0])
def test_motion_energy_is_per_reference_interval_not_per_frame(fps):
    _, _, e = fuse(fps, 1.5, lambda s, k: s.obj_box(k), 0.8)
    _, _, e20 = fuse(20.0, 1.5, lambda s, k: s.obj_box(k), 0.8)
    # px per 1/20 s: the same walker gives the same energy whatever the rate (within Farneback's own error)
    assert np.median(e) == pytest.approx(np.median(e20), rel=0.25)


def test_scaling_is_exact_at_the_reference_rate_and_proportional_elsewhere():
    assert MotionChannel.rate_scale(REFERENCE_INTERVAL_S) == pytest.approx(1.0)
    assert MotionChannel.rate_scale(1 / 8) == pytest.approx(8 / 20)
    assert MotionChannel.rate_scale(0.0) == 1.0 and MotionChannel.rate_scale(None) == 1.0


def test_a_gap_longer_than_a_second_restarts_the_flow():
    scene, mc = Scene(), MotionChannel()
    assert mc.update(scene.frame(0), 0.0) is None
    assert mc.update(scene.frame(1), 0.05) is not None
    assert mc.update(scene.frame(2), 5.0) is None, "a 5 s gap is a cut, not 5 s of motion"
    assert mc.update(scene.frame(3), 5.05) is not None


@pytest.mark.parametrize("case,box_of,score,moving,alert", [
    ("both agree", lambda s, k: s.obj_box(k), 0.8, True, True),
    ("appearance only (parked)", lambda s, k: s.parked_box(), 0.95, True, False),
    ("motion only (box below floor)", lambda s, k: s.obj_box(k), 0.05, True, False),
    ("box on still object", lambda s, k: s.obj_box(k), 0.8, False, False),
])
def test_same_scene_same_verdict_at_20_and_8_fps(case, box_of, score, moving, alert):
    verdicts = {}
    for fps in (20.0, 8.0):
        decisions, events, _ = fuse(fps, 1.5, box_of, score, moving)
        verdicts[fps] = (len(events) > 0, any(d.agree for d in decisions))
    assert verdicts[20.0] == verdicts[8.0], f"{case}: {verdicts}"
    assert verdicts[20.0][0] is alert


def test_calibration_warm_up_is_counted_in_seconds_with_a_frame_floor():
    cells = np.full((2, 2), 0.1, np.float32)
    for fps, need in ((20.0, 101), (8.0, 41), (2.0, 25)):
        cal = Calibrator("C", "visible", 0.55, warmup_seconds=5.0, min_frames=25)
        k = 0
        while not cal.ready:
            cal.observe(cells, (64, 64), 32, [], timestamp=k / fps)
            k += 1
        assert k == need, f"{fps} fps: ready after {k} frames"
    fixed = Calibrator("C", "visible", 0.55, warmup_frames=7)
    for k in range(7):
        fixed.observe(cells, (64, 64), 32, [], timestamp=k * 10.0)
    assert fixed.ready and fixed.count == 7


class SceneDetector:
    def __init__(self, scene: Scene) -> None:
        self.scene, self.last_ms = scene, 0.0

    def detect(self, image, *, camera_id, frame_index, timestamp, modality):
        return [Detection(bbox=self.scene.obj_box(frame_index), class_name="person", score=0.8, camera_id=camera_id,
                          frame_index=frame_index, timestamp=timestamp, modality=modality, class_id=0)]


@pytest.mark.parametrize("fps", [20.0, 8.0])
def test_full_pipeline_learns_and_alerts_at_either_rate(tmp_path, fps):
    """Calibration (seconds), motion (normalised), tracking and fusion together: one alert at both rates,
    raised after the same media time."""
    scene = scene_at(fps, start=(10.0, 110.0))
    pipe = Pipeline(SceneDetector(scene), PipelineConfig(camera_id=f"R{int(fps)}", modality="visible", stride=2,
                                                         warmup_seconds=2.0, warmup_min_frames=10,
                                                         calibration_dir=tmp_path, parallel=False))
    alerts, ready_at = [], None
    for k in range(int(4.0 * fps)):
        r = pipe.process(Frame(k, scene.frame(k), k / fps, 0.0, "file", "R"))
        if ready_at is None and pipe.calibration is not None:
            ready_at = k / fps
        alerts += r.fusion.events
    pipe.close()
    assert ready_at == pytest.approx(2.0, abs=2.5 / fps)
    assert len(alerts) == 1 and alerts[0].timestamp >= ready_at


def test_file_decimation_follows_media_time(tmp_path):
    """20 fps sampled to 8 fps keeps 8 frames per second (2-3 source frames apart), each with its media
    time, instead of round(20 / 8) = every 2nd frame = 10 fps."""
    path = tmp_path / "c.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    for i in range(40):
        vw.write(np.full((48, 64, 3), i * 5, np.uint8))
    vw.release()
    got = list(frames(path, is_rtsp=False, camera_id="C", target_fps=8.0))
    times = [f.media_time for f in got]
    assert len(got) == 16
    assert times[:5] == pytest.approx([0.0, 0.15, 0.25, 0.4, 0.5])
    gaps = np.diff(times)
    assert gaps.mean() == pytest.approx(1 / 8, abs=0.01) and set(np.round(gaps, 2)) <= {0.1, 0.15}


@pytest.mark.parametrize("fps,passes", [(20.0, 10), (8.0, 8)])
def test_detector_stride_has_a_ceiling_in_seconds(tmp_path, fps, passes):
    """Stride 2 at 20 fps is a pass every 0.1 s; at 8 fps it would be 0.25 s, too far apart for the tracker
    to associate a moving box, so the 0.125 s ceiling runs the detector on every 8 fps frame."""
    scene = scene_at(fps)
    calls = []

    class Counting(SceneDetector):
        def detect(self, image, **kw):
            calls.append(kw["frame_index"])
            return super().detect(image, **kw)

    pipe = Pipeline(Counting(scene), PipelineConfig(camera_id="S", modality="visible", stride=2,
                                                    calibration_dir=tmp_path, parallel=False))
    for k in range(int(fps)):
        pipe.process(Frame(k, scene.frame(k), k / fps, 0.0, "file", "S"))
    pipe.close()
    assert len(calls) == passes

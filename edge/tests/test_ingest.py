"""The one decode path, exercised on a generated file.

A file and an RTSP pull differ only in the string handed to OpenCV, so these
tests cover the path both sources take.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from pipeline.ingest import IngestError, Frame, frames, probe

WIDTH, HEIGHT, COUNT, FPS = 64, 48, 30, 10.0


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A short synthetic clip: a white square walking left to right."""
    path = tmp_path_factory.mktemp("clips") / "walk.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    for i in range(COUNT):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        x = int(i / COUNT * (WIDTH - 8))
        frame[HEIGHT // 2 - 4 : HEIGHT // 2 + 4, x : x + 8] = 255
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0
    return path


def test_probe_reports_the_real_geometry(clip):
    info = probe(clip, is_rtsp=False)
    assert info["width"] == WIDTH
    assert info["height"] == HEIGHT
    assert info["source"] == "file"


def test_frames_are_labelled_file_not_rtsp(clip):
    frame = next(frames(clip, is_rtsp=False, camera_id="RXL-01", target_fps=1000))
    assert isinstance(frame, Frame)
    assert frame.source == "file"
    assert frame.camera_id == "RXL-01"
    assert frame.shape == (WIDTH, HEIGHT)


def test_frames_are_indexed_from_zero_and_increase(clip):
    got = list(frames(clip, is_rtsp=False, camera_id="C", target_fps=1000, max_frames=5))
    assert [f.index for f in got] == [0, 1, 2, 3, 4]
    assert all(b.monotonic >= a.monotonic for a, b in zip(got, got[1:]))


def test_a_file_is_exhausted_rather_than_looping_by_default(clip):
    got = list(frames(clip, is_rtsp=False, camera_id="C", target_fps=1000))
    assert 0 < len(got) <= COUNT


def test_loop_keeps_producing_past_the_end(clip):
    got = list(
        frames(clip, is_rtsp=False, camera_id="C", target_fps=1000, loop=True,
               max_frames=COUNT + 5)
    )
    assert len(got) == COUNT + 5


def test_stop_event_ends_the_loop(clip):
    stop = threading.Event()
    stream = frames(clip, is_rtsp=False, camera_id="C", target_fps=1000, loop=True, stop=stop)
    assert next(stream).index == 0
    stop.set()
    assert list(stream) == []


def test_a_target_above_the_file_rate_keeps_every_frame(clip):
    # The clip is 10 fps. Asking for 20 cannot invent frames, so all 30 survive.
    got = list(frames(clip, is_rtsp=False, camera_id="C", target_fps=20.0))
    assert len(got) == COUNT


def test_a_target_below_the_file_rate_decimates(clip):
    # 10 fps clip sampled to 5 fps keeps every second frame.
    got = list(frames(clip, is_rtsp=False, camera_id="C", target_fps=5.0))
    assert len(got) == COUNT // 2


def test_pacing_spaces_emissions_by_the_target_interval(clip):
    got = list(frames(clip, is_rtsp=False, camera_id="C", target_fps=25.0,
                      pace=True, max_frames=4))
    gaps = [b.monotonic - a.monotonic for a, b in zip(got, got[1:])]
    assert len(got) == 4
    assert min(gaps) >= 1.0 / 25.0 - 0.005


def test_missing_source_raises_rather_than_hanging(tmp_path):
    with pytest.raises(IngestError, match="could not open"):
        next(frames(tmp_path / "nope.mp4", is_rtsp=False, camera_id="C"))


def test_non_positive_fps_is_rejected(clip):
    with pytest.raises(IngestError, match="target_fps"):
        next(frames(clip, is_rtsp=False, camera_id="C", target_fps=0))

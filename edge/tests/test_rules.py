"""Rule engine tests.

Four groups, matching the BUILD_PLAN acceptance criteria for Phase 4:

1. `TestMeasurementsSectionTwo` reproduces docs/MEASUREMENTS.md section 2's
   rule-sanity table exactly: which of the 10 named tracks fire which rule.
2. `TestHomography` — a known square, metre error < 5%.
3. `TestDori` — a synthetic camera geometry, assert the computed bands.
4. `TestThrottle` — 50 repeat events stay within budget; the per-camera and
   shift caps are enforced; a dismissal measurably shifts the baseline.

Every synthetic track below is a plain object satisfying `rules.tracks.
TrackLike` structurally (see `SimpleTrack`) — exactly the shape Phase 3's
`Track` dataclass is expected to satisfy once it merges (rules/tracks.py's
adapter contract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from rules import baseline as baseline_mod
from rules import config as config_mod
from rules import direction as direction_mod
from rules import dori as dori_mod
from rules import engine as engine_mod
from rules import fence as fence_mod
from rules import loitering as loitering_mod
from rules.homography import Homography
from rules.throttle import AlertThrottle
from rules.tracks import BBox, GroundPoint, TrackLike


# --------------------------------------------------------------------------
# A minimal, structural TrackLike used only by tests — this is exactly the
# adapter surface Phase 3's real `Track` dataclass needs to satisfy.
# --------------------------------------------------------------------------


@dataclass
class SimpleTrack:
    track_id: int | str
    camera_id: str
    class_name: str
    history: list[GroundPoint] = field(default_factory=list)
    bbox: BBox | None = None


assert isinstance(
    SimpleTrack(track_id=1, camera_id="c", class_name="person"), TrackLike
), "SimpleTrack must structurally satisfy TrackLike"


FPS = 20.0
DT = 1.0 / FPS
CAMERA_ID = "TEST-CAM"
Y_CONST = 500.0


def _track(track_id, waypoints_x: list[float], *, t0: float = 1_700_000_000.0) -> SimpleTrack:
    """Build a track whose ground point walks linearly through `waypoints_x`,
    one sub-segment per pair of consecutive waypoints, sampled at FPS.
    """
    history: list[GroundPoint] = []
    t = t0
    n_segments = len(waypoints_x) - 1
    samples_per_segment = 10
    history.append((t, waypoints_x[0], Y_CONST))
    for seg in range(n_segments):
        x_start, x_end = waypoints_x[seg], waypoints_x[seg + 1]
        for i in range(1, samples_per_segment + 1):
            t += DT
            frac = i / samples_per_segment
            x = x_start + frac * (x_end - x_start)
            history.append((t, x, Y_CONST))
    return SimpleTrack(track_id=track_id, camera_id=CAMERA_ID, class_name="person", history=history)


def _fixed_dwell_track(track_id, waypoints_x: list[float], dwell_s: float, *, t0: float = 1_700_000_000.0) -> SimpleTrack:
    """Like `_track`, but the total span is forced to exactly `dwell_s`."""
    track = _track(track_id, waypoints_x, t0=t0)
    n = len(track.history)
    if n < 2 or dwell_s <= 0:
        track.history = [(t0, x, Y_CONST) for _, x, _ in track.history]
        return track
    step = dwell_s / (n - 1)
    track.history = [(t0 + i * step, x, y) for i, (_, x, y) in enumerate(track.history)]
    return track


FENCE_X = 320.0
FENCE = fence_mod.FenceLine(name="pillar-42-line", points=[(FENCE_X, -10_000.0), (FENCE_X, 10_000.0)])

LOITER_CONFIG = loitering_mod.LoiteringConfig(dwell_s=3.0, net_displacement_px=90.0)
DIRECTION_CONFIG = direction_mod.DirectionConfig(min_displacement_px=40.0, opposing_angle_deg=90.0)


def _measurements_section_two_tracks() -> dict:
    """The 10 tracks from docs/MEASUREMENTS.md section 2, id -> SimpleTrack.

    Every track's dwell/net/path is constructed to land close to the
    published figures; the values that matter for a rule verdict (whether
    net crosses the 90 px loiter gate, whether the path crosses x=320,
    whether dwell reaches 3 s, whether the heading opposes the learnt
    majority) are exact.
    """
    return {
        1: _fixed_dwell_track(1, [50.0, 294.0], 3.5),  # net 244, no crossing, with flow
        2: _fixed_dwell_track(2, [600.0, 449.0], 1.4),  # net 151, against flow
        3: _fixed_dwell_track(3, [700.0, 510.0], 1.6),  # net 190, against flow
        4: _fixed_dwell_track(4, [200.0, 516.0], 4.8),  # net 316, crosses 320, with flow
        5: _fixed_dwell_track(5, [500.0, 401.0], 1.0),  # net 99, against flow
        6: _fixed_dwell_track(6, [300.0, 254.0, 344.0, 370.0], 5.0),  # net 70, path ~162, crosses 320
        9: _fixed_dwell_track(9, [250.0, 412.0], 4.7),  # net 162, crosses 320, with flow
        12: _fixed_dwell_track(12, [100.0, 97.0, 103.0], 1.2),  # net 3, noise
        26: _fixed_dwell_track(26, [200.0, 227.0, 222.0], 0.7),  # net 22, noise
        28: _fixed_dwell_track(28, [150.0, 184.0], 0.5),  # net 34, noise
    }


def _learnt_baseline() -> baseline_mod.CameraBaseline:
    """Baseline learnt from its own 10-track population: 6 with flow, 4 against.

    Kept separate from the evaluation tracks above — this reproduces
    MEASUREMENTS.md's *baseline statistic* ("6 with, 4 against") on its own
    population, independent of which of the 10 evaluated tracks later turn
    out to oppose it. That mirrors the source data: the baseline is learnt
    from the camera's ordinary traffic, not from the sample being graded.
    """
    with_flow = [_track(f"b-with-{i}", [100.0, 100.0 + 60.0]) for i in range(6)]
    against_flow = [_track(f"b-against-{i}", [700.0, 700.0 - 60.0]) for i in range(4)]
    return baseline_mod.learn(CAMERA_ID, with_flow + against_flow)


class TestBaselineLearning:
    def test_majority_flow_is_left_to_right_six_with_four_against(self):
        base = _learnt_baseline()
        assert base.majority_heading_deg == pytest.approx(0.0, abs=1e-6)
        assert base.with_flow_count == 6
        assert base.against_flow_count == 4


class TestMeasurementsSectionTwo:
    """Reproduces docs/MEASUREMENTS.md section 2 exactly: 10 tracks, 6 alerts, 4 normal."""

    EXPECTED_RULE_FIRED = {
        1: "",
        2: "wrong direction",
        3: "wrong direction",
        4: "fence crossed",
        5: "wrong direction",
        6: "fence crossed + loitering",
        9: "fence crossed",
        12: "",
        26: "",
        28: "",
    }

    @staticmethod
    @pytest.fixture(scope="class")
    def verdicts():
        tracks = _measurements_section_two_tracks()
        base = _learnt_baseline()
        config = config_mod.CameraRuleConfig(
            camera_id=CAMERA_ID,
            fences=(FENCE,),
            loitering=LOITER_CONFIG,
            direction=DIRECTION_CONFIG,
        )
        return {
            track_id: engine_mod.evaluate_track(track, config, base)
            for track_id, track in tracks.items()
        }

    @pytest.mark.parametrize("track_id", [1, 2, 3, 4, 5, 6, 9, 12, 26, 28])
    def test_rule_fired_matches_measurements_table(self, verdicts, track_id):
        verdict = verdicts[track_id]
        assert verdict.rule_fired == self.EXPECTED_RULE_FIRED[track_id]

    def test_track_six_fires_fence_and_loitering_with_evidence(self, verdicts):
        v6 = verdicts[6]
        names = {f.rule_name for f in v6.fired}
        assert names == {engine_mod.RULE_FENCE_CROSSED, engine_mod.RULE_LOITERING}
        loiter = next(f for f in v6.fired if f.rule_name == engine_mod.RULE_LOITERING)
        assert loiter.evidence["dwell_s"] == pytest.approx(5.0)
        assert loiter.evidence["net_displacement_px"] < 90.0
        assert loiter.evidence["path_length_px"] > loiter.evidence["net_displacement_px"]

    def test_result_is_ten_tracks_six_alerts_four_normal(self, verdicts):
        assert len(verdicts) == 10
        alerts = [v for v in verdicts.values() if v.fired]
        normal = [v for v in verdicts.values() if not v.fired]
        assert len(alerts) == 6
        assert len(normal) == 4

    def test_tracks_one_twelve_twentysix_twentyeight_fire_nothing(self, verdicts):
        for track_id in (1, 12, 26, 28):
            assert verdicts[track_id].fired == ()

    def test_every_fired_rule_carries_numeric_evidence_and_a_reason(self, verdicts):
        for verdict in verdicts.values():
            for fired in verdict.fired:
                assert fired.evidence, f"{fired.rule_name} on {fired.track_id} has no evidence"
                assert any(isinstance(v, (int, float)) for v in fired.evidence.values())
                assert fired.reason and fired.reason.strip()

    def test_reason_is_never_empty_even_with_no_vlm(self, verdicts):
        # The engine's `reason` is the fallback AlertQueue string when no VLM
        # explanation exists yet (docs/ARCHITECTURE_V2.md section 3.3).
        for verdict in verdicts.values():
            if verdict.fired:
                assert verdict.reason != ""


class TestFenceCrossing:
    def test_uses_segment_intersection_not_centre_in_polygon(self):
        # A track that jumps clean over the fence between two sampled frames
        # (never has a point ON x=320) must still be caught, because the
        # crossing test is against the path segment, not per-frame membership.
        track = SimpleTrack(
            track_id="jump",
            camera_id=CAMERA_ID,
            class_name="person",
            history=[(0.0, 300.0, 500.0), (0.05, 340.0, 500.0)],
        )
        hit = fence_mod.evaluate(track, FENCE)
        assert hit is not None
        assert hit.point_px[0] == pytest.approx(320.0, abs=0.5)

    def test_direction_filter_only_fires_for_the_configured_direction(self):
        inbound_only = fence_mod.FenceLine(
            name="in-only", points=[(320.0, -1000.0), (320.0, 1000.0)], direction="in"
        )
        going_in = SimpleTrack(1, CAMERA_ID, "person", [(0.0, 300.0, 0.0), (1.0, 340.0, 0.0)])
        going_out = SimpleTrack(2, CAMERA_ID, "person", [(0.0, 340.0, 0.0), (1.0, 300.0, 0.0)])
        one = fence_mod.evaluate(going_in, inbound_only)
        two = fence_mod.evaluate(going_out, inbound_only)
        assert one is not None and one.direction == "in"
        assert two is None


class TestHomography:
    def test_residual_within_five_percent_on_a_known_square(self):
        # A 5m x 5m square, viewed with mild perspective distortion, clicked
        # by an "operator" at integer pixel precision — the rounding is the
        # only source of error, and it must stay under 5% of the 5 m span.
        true_h = np.array(
            [
                [1.6, 0.05, 380.0],
                [0.02, 1.55, 610.0],
                [0.00006, 0.00022, 1.0],
            ]
        )

        def project(x, y):
            vec = true_h @ np.array([x, y, 1.0])
            return vec[0] / vec[2], vec[1] / vec[2]

        world_points = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (2.5, 2.5)]
        image_points = [tuple(round(c) for c in project(x, y)) for x, y in world_points]

        homography = Homography.calibrate("known-square", image_points, world_points)
        report = homography.residual

        assert report.max_error_pct < 5.0
        assert report.is_trustworthy

        # And a held-out distance (the square's own side) comes back close to 5 m.
        side_m = homography.distance_m(image_points[0], image_points[1])
        assert side_m == pytest.approx(5.0, rel=0.05)

    def test_degenerate_points_are_rejected(self):
        from rules.homography import HomographyError

        collinear = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        world = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
        with pytest.raises(HomographyError):
            Homography.calibrate("bad", collinear, world)


def _pinhole_ground_homography() -> Homography:
    """A synthetic, exact ground-plane pinhole camera, as a homography.

    Constructed directly as a 3x3 projective matrix (a flat ground plane
    under pinhole projection is *exactly* a homography — no approximation),
    so every row of the image has an analytically known world distance,
    letting the DORI test check `dori.grade()` against independently
    computed ground truth rather than only checking it against itself.
    """
    true_h = np.array(
        [
            [5.0, 0.0, -3000.0],
            [0.0, 0.0, 6000.0],
            [0.0, 1.0, 50.0],
        ]
    )

    def project(u, v):
        vec = true_h @ np.array([u, v, 1.0])
        return vec[0] / vec[2], vec[1] / vec[2]

    image_points, world_points = [], []
    for v in (0, 270, 540, 810, 1079):
        for u in (0, 600, 1199):
            image_points.append((float(u), float(v)))
            world_points.append(project(u, v))
    return Homography.calibrate("synthetic-pinhole", image_points, world_points)


class TestDori:
    IMAGE_WIDTH = 1200
    IMAGE_HEIGHT = 1080

    def test_bands_are_exactly_iec_62676_4(self):
        assert dori_mod.DORI_BANDS == {
            "detect": 25.0,
            "observe": 62.0,
            "recognise": 125.0,
            "identify": 250.0,
        }

    def test_detect_band_range_matches_independent_geometry(self):
        homography = _pinhole_ground_homography()
        grading = dori_mod.grade(homography, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)

        # Independently derived from the same true homography, without going
        # through dori.grade()'s row-walking code at all.
        true_h = np.array(
            [[5.0, 0.0, -3000.0], [0.0, 0.0, 6000.0], [0.0, 1.0, 50.0]]
        )

        def project_y(u, v):
            vec = true_h @ np.array([u, v, 1.0])
            return vec[1] / vec[2]

        probe_u = self.IMAGE_WIDTH / 2.0
        origin_y = project_y(probe_u, self.IMAGE_HEIGHT - 1)
        best = {"detect": 0.0, "observe": 0.0, "recognise": 0.0, "identify": 0.0}
        for v in range(self.IMAGE_HEIGHT - 1, -1, -1):
            y0 = project_y(probe_u, v)
            y1 = project_y(probe_u, v + 1)
            d = abs(y0 - y1)
            if d <= 0:
                continue
            ppm = 1.0 / d
            rng = y0 - origin_y
            for band, threshold in dori_mod.DORI_BANDS.items():
                if ppm >= threshold:
                    best[band] = max(best[band], rng)

        assert grading.detect_range_m == pytest.approx(best["detect"], rel=0.05)
        assert grading.observe_range_m == pytest.approx(best["observe"], rel=0.05)
        assert grading.recognise_range_m == pytest.approx(best["recognise"], rel=0.05)

    def test_bands_are_ordered_identify_within_detect(self):
        homography = _pinhole_ground_homography()
        grading = dori_mod.grade(homography, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        assert grading.identify_range_m <= grading.recognise_range_m
        assert grading.recognise_range_m <= grading.observe_range_m
        assert grading.observe_range_m <= grading.detect_range_m

    def test_never_claims_capability_beyond_the_computed_detect_band(self):
        homography = _pinhole_ground_homography()
        grading = dori_mod.grade(homography, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        beyond = grading.detect_range_m + 50.0
        assert grading.capability_at(beyond) is None
        assert grading.capability_at(0.0) in (None, "identify")

    def test_refuses_to_grade_an_untrustworthy_homography(self):
        # Four collinear-ish but not-quite-collinear points make a numerically
        # unstable fit; force a bad residual by hand-building one.
        from rules.homography import Homography as H
        from rules.homography import ResidualReport

        bad = H(
            camera_id="bad-cam",
            matrix=np.eye(3),
            inverse=np.eye(3),
            residual=ResidualReport(
                per_point_error_m=(10.0,),
                mean_error_m=10.0,
                max_error_m=10.0,
                reference_span_m=5.0,
                mean_error_pct=200.0,
                max_error_pct=200.0,
            ),
        )
        with pytest.raises(dori_mod.DoriError):
            dori_mod.grade(bad, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)


class TestThrottle:
    def test_fifty_repeat_events_stay_within_budget(self):
        throttle = AlertThrottle()
        now = 1_700_000_000.0
        alerts = []
        for i in range(50):
            alert = throttle.offer(CAMERA_ID, track_id=6, rule_name="loitering", fired_at_s=now + i)
            if alert is not None:
                alerts.append(alert)

        active = throttle.active_alerts()
        assert len(active) == 1
        assert active[0].repeat_count == 50
        assert len(active) <= throttle.per_camera_limit
        assert len(active) <= throttle.shift_budget

    def test_per_camera_rate_limit_caps_distinct_alerts_at_five(self):
        throttle = AlertThrottle()
        now = 1_700_000_000.0
        admitted = 0
        for track_id in range(50):
            alert = throttle.offer(CAMERA_ID, track_id=track_id, rule_name="loitering", fired_at_s=now)
            if alert is not None:
                admitted += 1
        assert admitted == 5
        assert len(throttle.active_alerts()) == 5

    def test_shift_budget_caps_total_alerts_across_cameras_at_twelve(self):
        throttle = AlertThrottle()
        now = 1_700_000_000.0
        admitted = 0
        cameras = [f"CAM-{i}" for i in range(6)]
        for camera in cameras:
            for track_id in range(5):  # 6 cameras * 5 = 30 candidate new alerts
                alert = throttle.offer(camera, track_id=track_id, rule_name="loitering", fired_at_s=now)
                if alert is not None:
                    admitted += 1
        assert admitted == 12
        assert admitted == throttle.shift_budget

    def test_dismiss_removes_the_alert(self):
        throttle = AlertThrottle()
        now = 1_700_000_000.0
        throttle.offer(CAMERA_ID, track_id=1, rule_name="loitering", fired_at_s=now)
        removed = throttle.dismiss(CAMERA_ID, 1, "loitering")
        assert removed is not None
        assert throttle.active_alerts() == []


class TestDismissalFeedback:
    def test_dismissal_measurably_shifts_the_baseline(self):
        base = _learnt_baseline()
        assert base.majority_heading_deg == pytest.approx(0.0, abs=1e-6)
        assert base.against_flow_count == 4

        # The operator dismisses a "wrong direction" alert on a track heading
        # the opposite way (180 deg) — the camera's baseline must move
        # measurably towards accepting that heading, and its against-flow
        # counter must drop by one (slide 4: "the operator dismisses one and
        # that camera's baseline updates").
        dismissed_track = _track("dismissed", [400.0, 340.0])  # heading 180 deg
        updated = baseline_mod.update_on_dismissal(base, dismissed_track, rule_name="wrong direction")

        assert updated.against_flow_count == base.against_flow_count - 1
        assert updated.with_flow_count == base.with_flow_count + 1
        assert updated.majority_heading_deg != base.majority_heading_deg
        # And it moved *towards* 180 deg (the dismissed heading), not away.
        assert abs(updated.majority_heading_deg - 0.0) > 0.0

    def test_loitering_dismissal_widens_typical_dwell(self):
        base = baseline_mod.CameraBaseline(camera_id=CAMERA_ID, typical_dwell_s=1.0)
        long_dwell_track = _fixed_dwell_track(1, [100.0, 105.0], 9.0)
        updated = baseline_mod.update_on_dismissal(base, long_dwell_track, rule_name="loitering")
        assert updated.typical_dwell_s > base.typical_dwell_s
        assert updated.dismissal_count == 1


class TestConfig:
    def test_sample_config_loads_end_to_end(self):
        sample = Path(__file__).resolve().parents[1] / "rules" / "examples" / "RXL-01.json"
        cfg = config_mod.load_file(sample)
        assert cfg.camera_id == "RXL-01"
        assert len(cfg.fences) == 1
        assert cfg.fences[0].points[0][0] == pytest.approx(320.0)
        assert cfg.homography is not None
        assert cfg.homography.residual.is_trustworthy

    def test_hot_reload_picks_up_a_changed_file(self, tmp_path):
        import json
        import os
        import time as time_mod

        camera_dir = tmp_path / "cams"
        camera_dir.mkdir()
        cfg_path = camera_dir / "CAM-X.json"
        cfg_path.write_text(json.dumps({"camera_id": "CAM-X", "loitering": {"dwell_s": 3.0}}))

        store = config_mod.ConfigStore(directory=camera_dir)
        first = store.get("CAM-X")
        assert first.loitering.dwell_s == 3.0

        cfg_path.write_text(json.dumps({"camera_id": "CAM-X", "loitering": {"dwell_s": 7.5}}))
        # Force the mtime forward so the change is detected even on
        # filesystems with coarse mtime resolution.
        future = time_mod.time() + 5
        os.utime(cfg_path, (future, future))

        second = store.get("CAM-X")
        assert second.loitering.dwell_s == 7.5
        assert second is not first

    def test_missing_camera_config_raises(self, tmp_path):
        store = config_mod.ConfigStore(directory=tmp_path)
        with pytest.raises(config_mod.RuleConfigError):
            store.get("NO-SUCH-CAMERA")

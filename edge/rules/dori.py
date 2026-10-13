"""Per-camera DORI grading, IEC 62676-4 bands (slide 6, ref 12).

MEASUREMENTS.md section 3 measured the underlying cliff directly: detection
recall holds at 90-100% down to a 19 px person height and collapses to 49% at
9 px. IEC 62676-4 turns "how tall must the target be in pixels" into a
standard vocabulary — Detect, Observe, Recognise, Identify — each requiring a
minimum pixel density **per metre of the target's real height**:

    detect     25 px/m
    observe    62 px/m
    recognise  125 px/m
    identify   250 px/m

These four numbers are the whole standard as far as this module is
concerned; they are never recomputed or approximated.

A camera's pixel density falls as range increases (the same 1.8 m person
covers fewer pixels the farther they stand from the camera), so each band is
satisfiable only out to some maximum range. This module derives that
density directly from the camera's own ground-plane homography: at a given
image row, one pixel of vertical extent maps to some real-world distance
near that row (via `homography.distance_m`); the reciprocal is that row's
px/m. Walking every row from the bottom of the frame (closest, highest
density) to the top (farthest, lowest density, subject to the horizon) finds
the farthest range at which each band's threshold still holds.

**The engine never reports a capability range beyond what it computed.** If
25 px/m is never reached within the image at all, `detect_range_m` is 0.0 —
not "unknown", not inherited from a spec sheet.
"""

from __future__ import annotations

from dataclasses import dataclass

from .homography import Homography

# IEC 62676-4, exactly. Order matters: identify is the strictest (highest
# px/m, shortest range), detect the loosest (lowest px/m, longest range).
DORI_BANDS: dict[str, float] = {
    "identify": 250.0,
    "recognise": 125.0,
    "observe": 62.0,
    "detect": 25.0,
}

_ROW_SAMPLE_STEP_PX = 1


class DoriError(ValueError):
    pass


@dataclass(frozen=True)
class DoriGrading:
    camera_id: str
    # metres; 0.0 means the band is never reached anywhere in this frame.
    identify_range_m: float
    recognise_range_m: float
    observe_range_m: float
    detect_range_m: float

    def range_for(self, band: str) -> float:
        if band not in DORI_BANDS:
            raise DoriError(f"unknown DORI band {band!r}")
        return getattr(self, f"{band}_range_m")

    def capability_at(self, range_m: float) -> str | None:
        """The best (strictest) band this camera can deliver at `range_m`.

        Returns None beyond `detect_range_m` — this camera makes no claim
        about anything past its own computed detect band, regardless of what
        a spec sheet or a wider band's threshold would imply.
        """
        if range_m > self.detect_range_m:
            return None
        for band in ("identify", "recognise", "observe", "detect"):
            if range_m <= self.range_for(band):
                return band
        return None  # unreachable given the check above, kept explicit

    def summary(self) -> str:
        return (
            f"{self.camera_id}: detect {self.detect_range_m:.1f}m, "
            f"observe {self.observe_range_m:.1f}m, "
            f"recognise {self.recognise_range_m:.1f}m, "
            f"identify {self.identify_range_m:.1f}m"
        )


def _px_per_m_at_row(homography: Homography, row: float, probe_u: float) -> float | None:
    """px/m at image row `row`, measured as the reciprocal of the real-world
    extent of one vertical pixel near that row.
    """
    try:
        world_dist_per_px = homography.distance_m((probe_u, row), (probe_u, row + 1.0))
    except Exception:  # noqa: BLE001 — a degenerate row (near/at the horizon) is skipped, not fatal
        return None
    if world_dist_per_px <= 0:
        return None
    return 1.0 / world_dist_per_px


def _range_at_row(homography: Homography, row: float, probe_u: float, origin_uv: tuple[float, float]) -> float | None:
    try:
        return homography.distance_m(origin_uv, (probe_u, row))
    except Exception:  # noqa: BLE001
        return None


def grade(
    homography: Homography,
    image_width: int,
    image_height: int,
    *,
    camera_origin_px: tuple[float, float] | None = None,
) -> DoriGrading:
    """Grade one camera by walking its image rows bottom (near) to top (far).

    `camera_origin_px` is the pixel whose ground-plane projection is treated
    as range zero — normally the bottom-centre of the frame, directly below
    the camera. Range at each sampled row is the homography-mapped distance
    from that origin.
    """
    if image_width <= 0 or image_height <= 0:
        raise DoriError("image_width and image_height must be positive")
    if not homography.residual.is_trustworthy:
        raise DoriError(
            f"refusing to grade {homography.camera_id}: {homography.residual.summary()}"
        )

    probe_u = image_width / 2.0
    origin_uv = camera_origin_px or (probe_u, float(image_height - 1))

    best_range: dict[str, float] = {band: 0.0 for band in DORI_BANDS}

    row = float(image_height - 1)
    while row >= 0.0:
        ppm = _px_per_m_at_row(homography, row, probe_u)
        rng = _range_at_row(homography, row, probe_u, origin_uv)
        if ppm is not None and rng is not None and rng >= 0:
            for band, threshold in DORI_BANDS.items():
                if ppm >= threshold:
                    best_range[band] = max(best_range[band], rng)
        row -= _ROW_SAMPLE_STEP_PX

    return DoriGrading(
        camera_id=homography.camera_id,
        identify_range_m=best_range["identify"],
        recognise_range_m=best_range["recognise"],
        observe_range_m=best_range["observe"],
        detect_range_m=best_range["detect"],
    )

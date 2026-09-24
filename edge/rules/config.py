"""Per-camera rule configuration: JSON on disk, hot-reloadable.

Slide 4's fence lines, sanctioned hours, thresholds and homography points are
all operator-edited. The HTTP endpoints that let an operator edit them come
in Phase 8 (build plan) — this module only defines the file format and the
loader those endpoints will eventually write through, and reloads a config
when its file changes on disk so an operator's edit takes effect without a
service restart.

One JSON file per camera, named `<camera_id>.json`, inside a configured
directory (`edge/rules/examples/` holds one worked sample). The schema is
intentionally flat and human-editable — an operator or a script, not a
programmer, is the one who edits fence points at a post.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .direction import DirectionConfig
from .fence import FenceLine
from .grouping import GroupingConfig
from .homography import Homography, HomographyError
from .hours import HoursConfig, SanctionedWindow
from .loitering import LoiteringConfig

REQUIRED_TOP_LEVEL_KEYS = {"camera_id"}


class RuleConfigError(ValueError):
    pass


def _parse_time(hhmm: str):
    from datetime import time as dt_time

    hh, mm = hhmm.split(":")
    return dt_time(hour=int(hh), minute=int(mm))


@dataclass(frozen=True)
class CameraRuleConfig:
    camera_id: str
    fences: tuple[FenceLine, ...] = ()
    hours: HoursConfig | None = None
    loitering: LoiteringConfig = field(default_factory=LoiteringConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    homography: Homography | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "CameraRuleConfig":
        missing = REQUIRED_TOP_LEVEL_KEYS - raw.keys()
        if missing:
            raise RuleConfigError(f"config missing required keys: {sorted(missing)}")

        camera_id = raw["camera_id"]

        fences = tuple(
            FenceLine(
                name=f["name"],
                points=[tuple(p) for p in f["points"]],
                direction=f.get("direction", "both"),
            )
            for f in raw.get("fences", [])
        )

        hours_cfg = None
        if raw.get("sanctioned_hours"):
            windows = tuple(
                SanctionedWindow(start=_parse_time(w["start"]), end=_parse_time(w["end"]))
                for w in raw["sanctioned_hours"]
            )
            hours_cfg = HoursConfig(camera_id=camera_id, sanctioned=windows)

        loitering_raw = raw.get("loitering", {})
        loitering_cfg = LoiteringConfig(
            dwell_s=loitering_raw.get("dwell_s", LoiteringConfig().dwell_s),
            net_displacement_m=loitering_raw.get("net_displacement_m"),
            net_displacement_px=loitering_raw.get(
                "net_displacement_px", LoiteringConfig().net_displacement_px
            ),
        )

        direction_raw = raw.get("direction", {})
        direction_cfg = DirectionConfig(
            min_displacement_px=direction_raw.get(
                "min_displacement_px", DirectionConfig().min_displacement_px
            ),
            opposing_angle_deg=direction_raw.get(
                "opposing_angle_deg", DirectionConfig().opposing_angle_deg
            ),
        )

        grouping_raw = raw.get("grouping", {})
        grouping_cfg = GroupingConfig(
            min_size=grouping_raw.get("min_size", GroupingConfig().min_size),
            radius_m=grouping_raw.get("radius_m"),
            radius_px=grouping_raw.get("radius_px", GroupingConfig().radius_px),
            persist_s=grouping_raw.get("persist_s", GroupingConfig().persist_s),
        )

        homography_obj = None
        homography_raw = raw.get("homography")
        if homography_raw:
            try:
                homography_obj = Homography.calibrate(
                    camera_id=camera_id,
                    image_points=[tuple(p) for p in homography_raw["image_points"]],
                    world_points=[tuple(p) for p in homography_raw["world_points"]],
                )
            except HomographyError as exc:
                raise RuleConfigError(f"bad homography calibration for {camera_id}: {exc}") from exc

        return cls(
            camera_id=camera_id,
            fences=fences,
            hours=hours_cfg,
            loitering=loitering_cfg,
            direction=direction_cfg,
            grouping=grouping_cfg,
            homography=homography_obj,
        )


def load_file(path: Path) -> CameraRuleConfig:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuleConfigError(f"{path}: invalid JSON: {exc}") from exc
    return CameraRuleConfig.from_dict(raw)


@dataclass
class ConfigStore:
    """Loads and hot-reloads every `<camera_id>.json` file in `directory`.

    `get(camera_id)` reloads that camera's file if its mtime has changed
    since it was last read, so an operator edit takes effect on the next
    call — no restart, no polling loop required. Each camera's config is
    read and parsed independently, so a mistake editing one camera's fence
    never blocks another camera's rules.
    """

    directory: Path
    _cache: dict[str, tuple[float, CameraRuleConfig]] = field(default_factory=dict)

    def _path_for(self, camera_id: str) -> Path:
        return self.directory / f"{camera_id}.json"

    def get(self, camera_id: str) -> CameraRuleConfig:
        path = self._path_for(camera_id)
        if not path.exists():
            raise RuleConfigError(f"no config for camera {camera_id!r} at {path}")

        mtime = path.stat().st_mtime
        cached = self._cache.get(camera_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        config = load_file(path)
        self._cache[camera_id] = (mtime, config)
        return config

    def known_cameras(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json"))

    def reload_all(self) -> None:
        """Force every cached camera to be re-read on its next `get()`."""
        self._cache.clear()

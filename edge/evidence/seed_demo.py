"""Seed a small, real evidence chain for the demo and the console.

    python -m evidence.seed_demo                       # ./var/evidence, 3 days ending today (IST)
    python -m evidence.seed_demo --data-dir /tmp/ev --days 4 --clips-per-day 5

Every block is produced by the same code path the pipeline uses: synthetic MJPEG clips are written
frame by frame through `EvidenceStore.open_clip` (hashed at capture), past days roll over and are
signed and mirrored to HQ, and one simulated power loss on the middle day leaves a logged gap. Only
the clock is simulated, so a few days of chain appear in seconds. Today's day stays open, exactly
as it would on a live post. The data directory must be empty (or not exist): this never touches an
existing chain.
"""

from __future__ import annotations

import argparse
import os
import sys
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from evidence.chain import DAY_TZ
from evidence.store import EvidenceStore, default_hq_dir
from evidence.verify import default_data_dir

CAMERAS = ("RXL-01", "JGB-03", "PNT-07", "SNL-01", "GLG-05", "PHU-02")  # backend/src/data/mockData.js
CLIP_TIMES_IST = ((1, 12), (2, 5), (3, 40), (4, 18), (5, 2), (5, 47), (6, 20), (6, 55))


class _SimClock:
    def __init__(self, start: datetime) -> None:
        self.t = start

    def __call__(self) -> datetime:
        return self.t

    def set(self, t: datetime) -> None:
        if t > self.t:
            self.t = t


def _frames(camera_id: str, when: datetime, count: int) -> list[bytes]:
    try:
        import cv2
    except ImportError:
        cv2 = None
    rng = np.random.default_rng(zlib.crc32(f"{camera_id}|{when.isoformat()}".encode()))
    frames = []
    for i in range(count):
        img = (rng.integers(0, 40, size=(90, 160, 3))).astype(np.uint8)
        img[40 + (i % 10), :, :] = 200  # a moving bar, so frames differ
        if cv2 is not None:
            cv2.putText(img, f"{camera_id} {when:%H:%M:%S}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            ok, buf = cv2.imencode(".jpg", img)
            frames.append(buf.tobytes() if ok else img.tobytes())
        else:
            frames.append(img.tobytes())
    return frames


def seed(
    data_dir: str | os.PathLike,
    hq_dir: str | os.PathLike | None = None,
    days: int = 3,
    clips_per_day: int = 4,
    end_day: date | None = None,
    power_loss: bool = True,
    device_id: str = "post-raxaul-01",
) -> dict:
    data_dir = Path(data_dir)
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(f"{data_dir} is not empty; seed into an empty directory")
    now_real = datetime.now(timezone.utc)
    end_day = end_day or now_real.astimezone(DAY_TZ).date()
    first_day = end_day - timedelta(days=days - 1)
    clips_per_day = max(1, min(clips_per_day, len(CLIP_TIMES_IST)))

    def ist(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
        return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=DAY_TZ).astimezone(timezone.utc)

    clock = _SimClock(ist(first_day, 0, 30))
    store = EvidenceStore(data_dir, hq_dir=hq_dir, device_id=device_id, clock=clock)
    cam_i = 0
    loss_day = first_day + timedelta(days=days // 2) if power_loss and days >= 2 else None
    for k in range(days):
        d = first_day + timedelta(days=k)
        for j, (hh, mm) in enumerate(CLIP_TIMES_IST[:clips_per_day]):
            start = ist(d, hh, mm)
            end = start + timedelta(seconds=12)
            if end + timedelta(seconds=2) > now_real:
                break  # never seal in the future
            if d == loss_day and j == clips_per_day // 2:
                # A clip is being written when the board loses power: nothing is closed cleanly.
                clock.set(start)
                w = store.open_clip(CAMERAS[cam_i % len(CAMERAS)], start)
                for f in _frames("CUT", start, 5):
                    w.write_frame(f)
                store.heartbeat()
                os.close(store._lock_fd)  # the process dies here: no close(), no run-marker update
                store._lock_fd = -1
                clock.set(start + timedelta(minutes=35))  # power back
                store = EvidenceStore(data_dir, hq_dir=hq_dir, clock=clock)
                continue
            clock.set(start)
            w = store.open_clip(CAMERAS[cam_i % len(CAMERAS)], start)
            for f in _frames(CAMERAS[cam_i % len(CAMERAS)], start, 12):
                w.write_frame(f)
            clock.set(end + timedelta(seconds=1))
            w.seal(end)
            cam_i += 1
    if end_day < now_real.astimezone(DAY_TZ).date():
        clock.set(ist(end_day, 23, 59, 59))
        store.seal_day(end_day.isoformat(), allow_open=True)
    summary = {
        "data_dir": str(data_dir),
        "hq_dir": str(Path(hq_dir) if hq_dir else default_hq_dir(data_dir)),
        "device_id": store.device_id,
        "key_id": store.signer.key_id,
        "blocks": len(store.blocks()),
        "days": [(first_day + timedelta(days=k)).isoformat() for k in range(days)],
        "sealed_days": sorted(p.stem for p in (data_dir / "roots").glob("*.json")),
        "gaps": len(store.gaps()),
    }
    store.close()
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evidence.seed_demo", description="Seed a demo evidence chain.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--hq-dir", type=Path, default=None)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--clips-per-day", type=int, default=4)
    p.add_argument("--end-date", type=date.fromisoformat, default=None, help="last IST day (default today)")
    p.add_argument("--no-power-loss", action="store_true", help="skip the simulated power loss")
    args = p.parse_args(argv)
    try:
        s = seed(
            args.data_dir or default_data_dir(),
            hq_dir=args.hq_dir or os.environ.get("EVIDENCE_HQ_DIR") or None,
            days=max(1, args.days),
            clips_per_day=args.clips_per_day,
            end_day=args.end_date,
            power_loss=not args.no_power_loss,
        )
    except FileExistsError as exc:
        print(f"refusing to seed: {exc}", file=sys.stderr)
        return 2
    print(f"seeded {s['blocks']} blocks over {', '.join(s['days'])} into {s['data_dir']}")
    print(f"device {s['device_id']}, signing key {s['key_id']}, HQ mirror {s['hq_dir']}")
    print(f"sealed + mirrored days: {', '.join(s['sealed_days']) or 'none'}; gaps logged: {s['gaps']}")
    print("next: python -m evidence.verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())

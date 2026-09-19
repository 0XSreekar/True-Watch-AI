#!/usr/bin/env python3
"""Single-channel LWIR -> 3-channel, per DATASET_SPEC.md section 3.4 and slide 4.

The invariant, and the whole point of this script: B = G = R. No false colour.
MEASUREMENTS.md section 1 measured what false-colour thermal costs a COCO-pretrained
detector: AP@50 = 0.133, precision 0.364, recall 0.093. This pipeline replicates the
channel instead, and gate G10 verifies the stored artefact really is 3-channel.

Steps: read single channel (16-bit sources are linearly rescaled to 8-bit and the
fact is recorded) -> CLAHE -> np.repeat to three channels -> write lossless PNG.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    Counters,
    Logger,
    base_parser,
    imwrite,
    load_augment,
    load_state,
    require_dirs,
    run,
    save_state,
    write_json,
    write_lines,
)

SCRIPT = "04_ir_to_3ch"


def to_three_channel(path: Path, clip: float, grid: int, counters: Counters, log: Logger):
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        log.warn("empty file", image=path.as_posix())
        counters.bump("image.empty")
        return None
    raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if raw is None:
        log.warn("undecodable", image=path.as_posix())
        counters.bump("image.undecodable")
        return None

    if raw.ndim == 3:
        if raw.shape[2] == 3 and np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(
            raw[..., 1], raw[..., 2]
        ):
            counters.bump("image.already_replicated")
            return raw
        channel = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        counters.bump("image.colour_input_flattened")
    else:
        channel = raw

    if channel.dtype != np.uint8:
        counters.bump("image.rescaled_from_16bit")
        channel = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    equalised = clahe.apply(channel)

    # The invariant. B = G = R, no colormap, ever.
    three = np.repeat(equalised[:, :, None], 3, axis=2)
    counters.bump("image.replicated")
    return three


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument(
        "--index",
        default=None,
        help="jsonl index to read (default: every *.jsonl under <processed>/index)",
    )
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    processed = Path(args.processed).resolve()
    index_dir = processed / "index"
    out_root = processed / "ir3"

    # Only the per-source indexes. Globbing *.jsonl would re-read combined.jsonl and the
    # later derived indexes, which appends this script's own output back into its input
    # and multiplies the corpus on every re-run.
    source_indexes = ("idd.jsonl", "kaist.jsonl", "llvip.jsonl")
    indexes = [Path(args.index)] if args.index else [index_dir / n for n in source_indexes]
    indexes = [p for p in indexes if p.exists()]
    if not indexes:
        log.error("no index files found - run the converters first", looked_in=str(index_dir))
        return 1

    augment = load_augment()["infrared_conversion"]
    if augment.get("false_colour"):
        log.error("augment.yaml enables false colour; section 3.4 forbids it")
        return 1
    clip = float(augment.get("clahe_clip", 2.0))
    grid = int(augment.get("clahe_tile_grid", 8))

    state = load_state(processed, SCRIPT)
    done = set(state.get("done", [])) if not args.force else set()
    require_dirs(out_root)

    rewritten: list[str] = []
    converted = 0
    failures = 0
    seen = 0

    for index_path in indexes:
        log.info("reading index", index=index_path.name)
        with index_path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("modality") != "lwir":
                    rewritten.append(json.dumps(record, sort_keys=True))
                    continue

                seen += 1
                if args.limit and seen > args.limit:
                    rewritten.append(json.dumps(record, sort_keys=True))
                    continue

                source = Path(record["image"])
                target = out_root / record.get("source", "unknown") / f"{source.stem}.png"

                if target.exists() and not args.force:
                    counters.bump("image.already_converted")
                    record["image"] = target.as_posix()
                    record["channels"] = 3
                    rewritten.append(json.dumps(record, sort_keys=True))
                    done.add(source.as_posix())
                    continue

                if args.dry_run:
                    counters.bump("image.would_convert")
                    rewritten.append(json.dumps(record, sort_keys=True))
                    continue

                if not source.exists():
                    log.warn("source image missing", image=source.as_posix())
                    counters.bump("image.missing")
                    failures += 1
                    rewritten.append(json.dumps(record, sort_keys=True))
                    continue

                three = to_three_channel(source, clip, grid, counters, log)
                if three is None:
                    failures += 1
                    rewritten.append(json.dumps(record, sort_keys=True))
                    continue

                if not imwrite(target, three):
                    log.error("write failed", image=target.as_posix())
                    counters.bump("image.write_failed")
                    failures += 1
                    rewritten.append(json.dumps(record, sort_keys=True))
                    continue

                record["image"] = target.as_posix()
                record["channels"] = 3
                rewritten.append(json.dumps(record, sort_keys=True))
                done.add(source.as_posix())
                converted += 1

    out_index = processed / "index" / "combined.jsonl"
    if not args.dry_run:
        write_lines(out_index, rewritten)
        state["done"] = sorted(done)
        save_state(processed, SCRIPT, state)
        write_json(
            processed / "reports" / "ir_to_3ch.json",
            {"converted": converted, "failures": failures, "counters": counters.as_dict()},
        )

    counters.report(log)
    log.info("done", lwir_seen=seen, converted=converted, failures=failures, index=str(out_index))
    return 1 if failures else 0


if __name__ == "__main__":
    run(main)

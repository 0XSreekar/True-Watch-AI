#!/usr/bin/env python3
"""Synthetic Nepali plate generator. DATASET_SPEC.md section 6.

MEASUREMENTS.md section 4 records that pretrained PP-OCRv5 Devanagari read about half the
characters on a real Nepali plate, so slide 5's 85% recognition target needs a fine-tune.
This produces the corpus for it: >= 20,000 samples with PaddleOCR recognition labels.

Every uncertain detail of the plate format - geometry, zone orthography, vehicle-class
letters, colour series - lives in plates.yaml and is marked OQ-7 to OQ-10 in the spec.
Nothing about the format is hard-coded here, because being wrong in a config file costs
thirty seconds and being wrong in code costs a rewrite.

Deterministic: one random.Random(seed) drives composition and degradation, and samples are
emitted in index order, so --seed 42 reproduces the corpus byte for byte.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import degrade as degrade_module  # noqa: E402

FONT_HINTS = ("NotoSansDevanagari", "NotoSerifDevanagari")
SYSTEM_FONT_DIRS = (
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path.home() / "Library/Fonts",
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
)


def log(level: str, message: str, **fields) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{stamp} {level:<5} gen_plates        {message}"
    if extra:
        line = f"{line} | {extra}"
    print(line, file=sys.stderr if level in ("ERROR", "FATAL") else sys.stdout, flush=True)


def find_font(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    local = sorted(HERE.glob("fonts/*.ttf")) + sorted(HERE.glob("fonts/**/*.ttf"))
    for path in local:
        if any(hint.lower() in path.name.lower() for hint in FONT_HINTS):
            return path
    if local:
        return local[0]
    for directory in SYSTEM_FONT_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*.ttf"):
            if any(hint.lower() in path.name.lower() for hint in FONT_HINTS):
                return path
    return None


# Measured on 58 Google Fonts Devanagari files: Latin-digit fonts score 0.89-0.99, true Devanagari digits 0.58 or less.
LATIN_DIGIT_LIMIT = 0.75


def latin_digit_score(font_path: Path) -> float:
    """How Latin the font's Devanagari digits look: the median overlap of each of १-९ with its Latin twin.

    Some Devanagari fonts (Hind, Rajdhani, Teko, Poppins, Sarpanch, Rozha One) draw the Devanagari digit code
    points with Latin-shaped glyphs. A plate rendered in one of them shows "2638" under the label "२६३८", which
    teaches the recogniser the wrong glyphs, so --font-dir drops fonts that score high here. Each glyph is cropped
    to its ink and resized before comparing, so only shape counts, not advance or baseline. Every named weight of a
    variable font is scored and the worst one is returned.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    def glyph(font, char):
        image = Image.new("L", (240, 240), 0)
        ImageDraw.Draw(image).text((40, 20), char, font=font, fill=255)
        box = image.getbbox()
        if box is None:
            return np.zeros((64, 64), np.float32)
        return np.asarray(image.crop(box).resize((64, 64)), dtype=np.float32) / 255.0

    font = ImageFont.truetype(str(font_path), 120)
    try:
        names = font.get_variation_names()
    except OSError:
        names = []
    worst = 0.0
    for name in names or [None]:
        if name:
            font.set_variation_by_name(name)
        overlaps = []
        for devanagari, latin in zip("१२३४५६७८९", "123456789"):
            a, b = glyph(font, devanagari), glyph(font, latin)
            overlaps.append(float(np.minimum(a, b).sum() / max(1e-6, np.maximum(a, b).sum())))
        worst = max(worst, float(np.median(overlaps)))
    return worst


def load_config(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def weighted_choice(rng: random.Random, options: list[dict]) -> dict:
    total = sum(float(o.get("weight", 1.0)) for o in options)
    point = rng.uniform(0, total)
    running = 0.0
    for option in options:
        running += float(option.get("weight", 1.0))
        if point <= running:
            return option
    return options[-1]


def to_devanagari(number: int, digits: list[str], width: int) -> str:
    text = str(number).zfill(width)
    return "".join(digits[int(ch)] for ch in text)


def compose_province(rng: random.Random, config: dict) -> dict:
    """Province plate: header "बागमती प्रदेश-०२", then lot (3 digits) + class, then the serial.

    Three printed lines (motorbikes) or two, with lot, class and serial on one line (cars). The registration text
    joins everything without spaces or hyphens: "बागमतीप्रदेश०२०३१प२०५०".
    """
    digits = config["digits"]
    province = weighted_choice(rng, config["provinces"])
    office = to_devanagari(rng.randint(1, int(config["province_office_max"])), digits, 2)
    lot = to_devanagari(rng.randint(1, int(config["province_lot_max"])), digits, 3)
    classes = config["vehicle_classes"]["confirmed"] + config["vehicle_classes"]["unverified"]
    vehicle_class = rng.choice(classes)
    serial = rng.randint(0, 10 ** int(config["serial"]["digits"]) - 1)
    serial_text = to_devanagari(serial, digits, int(config["serial"]["digits"]))
    header = f"{province['name']} {office}"
    three = int(config["layout"].get("lines", 3)) == 3
    lines = [header, f"{lot} {vehicle_class}", serial_text] if three else [header, f"{lot} {vehicle_class} {serial_text}"]
    shown_header = f"{province['name']}-{office}" if rng.random() < 0.7 else header
    return {
        "serial_int": serial,
        "text": f"{province['name']}{office}{lot}{vehicle_class}{serial_text}".replace(" ", ""),
        "line1": lines[0],
        "line2": lines[1],
        "lines": lines,
        "display_lines": [shown_header] + lines[1:],
        # Band share of the text height and font scale per line: a small header over larger numbers.
        "bands": [0.24, 0.33, 0.43] if three else [0.36, 0.64],
        "scales": [0.55, 0.95, 1.2] if three else [0.62, 1.0],
    }


def compose(rng: random.Random, config: dict) -> dict:
    if config.get("layout", {}).get("top_line") == "province":
        return compose_province(rng, config)
    digits = config["digits"]
    zones = config["zones"]["confirmed"] + config["zones"]["unverified"]
    classes = config["vehicle_classes"]["confirmed"] + config["vehicle_classes"]["unverified"]

    zone = rng.choice(zones)
    vehicle_class = rng.choice(classes)
    lot = rng.randint(int(config["lot_number"]["min"]), int(config["lot_number"]["max"]))
    serial_width = int(config["serial"]["digits"])
    serial = rng.randint(0, 10**serial_width - 1)

    # Real two-line plates (photographed in Kathmandu) print "zone lot class" over the serial, the lot without a
    # leading zero and often a dot after the zone ("बा.५८ प" / "४०९३"). plates_real.yaml selects that layout; the
    # default keeps the original "zone lot" / "class serial" composition byte for byte.
    real_layout = config.get("layout", {}).get("top_line") == "zone_lot_class"
    lot_text = to_devanagari(lot, digits, 1 if real_layout else 2)
    serial_text = to_devanagari(serial, digits, serial_width)

    fields = {
        "zone": zone,
        "lot": lot_text,
        "vehicle_class": vehicle_class,
        "serial": serial_text,
        "serial_int": serial,
        "line1": f"{zone} {lot_text}",
        "line2": f"{vehicle_class} {serial_text}",
        "text": f"{zone}{lot_text}{vehicle_class}{serial_text}",
    }
    if real_layout:
        one_line = int(config["layout"].get("lines", 2)) == 1  # long front plates: "बा.२० च ४६८०" on one line
        fields["line1"] = f"{zone} {lot_text} {vehicle_class}" + (f" {serial_text}" if one_line else "")
        fields["line2"] = "" if one_line else serial_text
        dot = rng.random() < float(config["layout"].get("zone_dot_probability", 0.0))
        fields["display1"] = fields["line1"].replace(f"{zone} ", f"{zone}.", 1) if dot else fields["line1"]
    return fields


def render_plate(fields: dict, colours: dict, layout: dict, font_path: Path, rng: random.Random):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    supersample = int(layout.get("render_supersample", 4))
    width = int(layout["plate_width_px"]) * supersample
    height = int(layout["plate_height_px"]) * supersample
    margin = int(layout["margin_px"]) * supersample
    radius = int(layout["corner_radius_px"]) * supersample
    border = int(layout["border_px"]) * supersample
    gap = int(layout["line_gap_px"]) * supersample

    background = tuple(int(c) for c in colours["background"])
    text_colour = tuple(int(c) for c in colours["text"])

    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        [(border // 2, border // 2), (width - border // 2, height - border // 2)],
        radius=radius,
        outline=text_colour,
        width=border,
    )

    usable_height = (height - 2 * margin - gap) // 2
    size = max(12, int(usable_height * 0.82))
    # A list of fonts (--font-dir) draws one per plate, plus a weight, a painted-or-embossed finish and a stroke
    # width. Every extra draw happens only on this branch, so a single-font run consumes the random stream exactly
    # as before and --seed 42 still reproduces the original corpus.
    varied = isinstance(font_path, (list, tuple))
    if varied:
        font_path = rng.choice(font_path)
        size = max(12, int(size * rng.uniform(0.82, 1.0)))
    try:
        font = ImageFont.truetype(str(font_path), size)
    except OSError:
        return None
    stroke = 0
    variation = None
    if varied:
        try:
            names = font.get_variation_names()
        except OSError:
            names = []
        if names:
            variation = rng.choice(names)
            font.set_variation_by_name(variation)
        stroke = rng.choice((0, 0, 0, 1, 2)) * supersample

    # Real-layout plates paint the serial larger than the top line: bands of 42% / 58% of the text height, fonts
    # scaled 0.84x / 1.16x and shrunk further if a line would overrun the plate. The default layout keeps two equal
    # bands and one font, exactly as before.
    bands = [(margin, usable_height), (margin + usable_height + gap, usable_height)]
    fonts = [font, font]
    if "display_lines" in fields:
        # Province plates: N bands sized by fields["bands"], each line's font scaled and shrunk to fit.
        text_height = height - 2 * margin - gap * (len(fields["display_lines"]) - 1)
        bands, fonts, top = [], [], margin
        for share, scale, line in zip(fields["bands"], fields["scales"], fields["display_lines"]):
            band = int(text_height * share)
            bands.append((top, band))
            top += band + gap
            line_size = max(12, int(band * 0.8 * min(1.0, scale + 0.15)))
            while True:
                line_font = ImageFont.truetype(str(font_path), line_size)
                if variation:
                    line_font.set_variation_by_name(variation)
                box = draw.textbbox((0, 0), line, font=line_font)
                if (box[2] - box[0] <= width - 2 * margin and box[3] - box[1] <= band * 1.05) or line_size <= 12:
                    break
                line_size = int(line_size * 0.92)
            fonts.append(line_font)
    elif "display1" in fields and not fields["line2"]:
        bands = [(margin, height - 2 * margin)]
        line_size = max(12, int((height - 2 * margin) * 0.78))
        while True:
            line_font = ImageFont.truetype(str(font_path), line_size)
            if variation:
                line_font.set_variation_by_name(variation)
            box = draw.textbbox((0, 0), fields["display1"], font=line_font)
            if box[2] - box[0] <= width - 2 * margin or line_size <= 12:
                break
            line_size = int(line_size * 0.92)
        fonts = [line_font]
    elif "display1" in fields:
        top = int(2 * usable_height * 0.42)
        bands = [(margin, top), (margin + top + gap, 2 * usable_height - top)]
        fonts = []
        for scale, line in ((0.84, fields["display1"]), (1.16, fields["line2"])):
            line_size = max(12, int(size * scale))
            while True:
                line_font = ImageFont.truetype(str(font_path), line_size)
                if variation:
                    line_font.set_variation_by_name(variation)
                box = draw.textbbox((0, 0), line, font=line_font)
                if box[2] - box[0] <= width - 2 * margin or line_size <= 12:
                    break
                line_size = int(line_size * 0.92)
            fonts.append(line_font)

    offsets = layout.get("emboss_offset_px", [1, 2])
    emboss = int(rng.choice(offsets)) * supersample
    if varied and rng.random() < 0.35:
        emboss = 0  # painted plate: no pressed relief

    shown = fields.get("display_lines") or [l for l in (fields.get("display1", fields["line1"]), fields["line2"]) if l]
    for index, line in enumerate(shown):
        font = fonts[index]
        band_top, band_height = bands[index]
        box = draw.textbbox((0, 0), line, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        x = (width - text_width) // 2 - box[0]
        y = band_top + (band_height - text_height) // 2 - box[1]

        # Emboss: a dark copy and a light copy under the flat glyph. This is what makes a
        # synthetic plate look pressed rather than printed, and printed-looking plates are
        # why naive synthetic corpora fail on real photographs.
        if emboss:
            draw.text((x + emboss, y + emboss), line, font=font, fill=(0, 0, 0))
            draw.text((x - emboss, y - emboss), line, font=font, fill=(255, 255, 255))
        draw.text((x, y), line, font=font, fill=text_colour, stroke_width=stroke, stroke_fill=text_colour)

    if layout.get("rivets"):
        rivet_radius = max(2, int(6 * supersample * 0.6))
        for cx, cy in (
            (margin, height // 2),
            (width - margin, height // 2),
        ):
            draw.ellipse(
                [(cx - rivet_radius, cy - rivet_radius), (cx + rivet_radius, cy + rivet_radius)],
                fill=tuple(max(0, c - 60) for c in background),
            )

    import cv2

    array = np.array(image)[:, :, ::-1]
    return cv2.resize(
        array,
        (int(layout["plate_width_px"]), int(layout["plate_height_px"])),
        interpolation=cv2.INTER_AREA,
    )


def check_font(font_path: Path | None) -> int:
    if font_path is None:
        log("ERROR", "no Devanagari font found", fix="see datasets/plates/fonts/README.md")
        return 2
    try:
        from PIL import ImageFont

        ImageFont.truetype(str(font_path), 48)
    except Exception as exc:
        log("ERROR", "font could not be loaded", font=str(font_path), detail=str(exc)[:160])
        return 2
    log("INFO", "font ok", font=str(font_path))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--count", type=int, default=20000, help="samples to generate")
    ap.add_argument("--out", default=str(HERE / "out"), help="output directory")
    ap.add_argument("--config", default=str(HERE / "plates.yaml"), help="plate composition config")
    ap.add_argument("--font", default=None, help="explicit path to a Devanagari ttf")
    ap.add_argument(
        "--font-dir",
        default=None,
        help="render each plate in a font drawn from every ttf in this directory (with random weight, stroke and "
        "painted or embossed finish); check glyph coverage with --check-font first",
    )
    ap.add_argument(
        "--labels",
        default=str(HERE / "labels"),
        help="where the PaddleOCR recognition labels are written; move it with --out when "
        "generating a throwaway corpus, or the two corpora share one label file",
    )
    ap.add_argument("--seed", type=int, default=42, help="seed for every random operation")
    ap.add_argument("--dry-run", action="store_true", help="plan the work, write nothing")
    ap.add_argument("--force", action="store_true", help="regenerate samples that already exist")
    ap.add_argument("--check-font", action="store_true", help="resolve and load the font, then exit")
    ap.add_argument("--clean", action="store_true", help="render 500 undegraded plates as well")
    args = ap.parse_args()

    font_path = find_font(args.font)
    if args.font_dir:
        fonts = sorted(Path(args.font_dir).glob("*.ttf"))
        if args.check_font:
            return max([check_font(path) for path in fonts] or [2])
        dropped = [path for path in fonts if latin_digit_score(path) > LATIN_DIGIT_LIMIT]
        for path in dropped:
            log("WARN", "font dropped: Devanagari digits drawn as Latin", font=path.name)
        fonts = [path for path in fonts if path not in dropped]
        if not fonts:
            log("ERROR", "no usable ttf files in --font-dir", font_dir=args.font_dir)
            return 2
        log("INFO", "fonts resolved", count=len(fonts), dropped=len(dropped), font_dir=args.font_dir)
        font_path = fonts
    else:
        if args.check_font:
            return check_font(font_path)
        if font_path is None:
            log("ERROR", "no Devanagari font found", fix="see datasets/plates/fonts/README.md")
            return 2
        log("INFO", "font resolved", font=str(font_path))

    config = load_config(Path(args.config))
    layout = config["layout"]
    if layout.get("top_line") == "province":
        from PIL import features

        if not features.check("raqm"):
            # Without complex shaping "प्रदेश" renders with a visible virama and "लुम्बिनी" with its vowel sign after the
            # consonants: glyph shapes no real plate has, so the corpus would teach the wrong thing.
            log("ERROR", "province plates need Pillow with libraqm", fix="brew install libraqm; "
                "DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python3 gen_plates.py ...")
            return 2
    out_root = Path(args.out).resolve()
    images_dir = out_root / "images"
    clean_dir = out_root / "clean"
    labels_dir = Path(args.labels).resolve()

    rng = random.Random(args.seed)
    counts = {"written": 0, "skipped": 0, "failed": 0}
    rows: list[tuple[str, str, int]] = []
    charset: set[str] = set()
    stage_counter: dict[str, int] = {}

    if args.dry_run:
        log("INFO", "dry-run", would_generate=args.count, out=str(out_root))
        sample = compose(rng, config)
        log("INFO", "example composition", text=sample["text"], line1=sample["line1"], line2=sample["line2"])
        return 0

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        clean_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    for index in range(args.count):
        name = f"{index:06d}.jpg"
        target = images_dir / name
        fields = compose(rng, config)
        charset.update(fields["text"])

        # Real-layout plates carry their line texts as a third column (build_line_labels.py reads it).
        if "lines" in fields:
            label = fields["text"] + "\t" + "|".join(fields["lines"])
        else:
            label = fields["text"] + (
                f"\t{fields['line1']}" + (f"|{fields['line2']}" if fields["line2"] else "") if "display1" in fields else ""
            )
        if target.exists() and not args.force:
            counts["skipped"] += 1
            rows.append((f"out/images/{name}", label, fields["serial_int"]))
            continue

        colours = weighted_choice(rng, config["colour_series"])
        plate = render_plate(fields, colours, layout, font_path, rng)
        if plate is None:
            counts["failed"] += 1
            log("ERROR", "render failed", index=index)
            continue

        if args.clean and index < 500:
            cv2.imwrite(str(clean_dir / name), plate)

        degraded, applied = degrade_module.degrade(plate, rng)
        for stage in applied:
            stage_counter[stage] = stage_counter.get(stage, 0) + 1

        if not cv2.imwrite(str(target), degraded, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            counts["failed"] += 1
            log("ERROR", "write failed", path=str(target))
            continue

        rows.append((f"out/images/{name}", label, fields["serial_int"]))
        counts["written"] += 1

        if counts["written"] and counts["written"] % 2000 == 0:
            log("INFO", "progress", written=counts["written"], target=args.count)

    # Split by serial so no serial appears in both files (section 6.4).
    val_share = float(config["split"]["val_share"])
    serials = sorted({serial for _p, _t, serial in rows})
    rng_split = random.Random(args.seed)
    rng_split.shuffle(serials)
    val_serials = set(serials[: int(round(len(serials) * val_share))])

    train_lines = [f"{p}\t{t}" for p, t, s in rows if s not in val_serials]
    val_lines = [f"{p}\t{t}" for p, t, s in rows if s in val_serials]

    (labels_dir / "rec_gt_train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (labels_dir / "rec_gt_val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    (labels_dir / "charset.txt").write_text("\n".join(sorted(charset)) + "\n", encoding="utf-8")

    summary = {
        "generated": counts["written"],
        "skipped_existing": counts["skipped"],
        "failed": counts["failed"],
        "total_labelled": len(rows),
        "train": len(train_lines),
        "val": len(val_lines),
        "charset_size": len(charset),
        "seed": args.seed,
        "degradation_stages": dict(sorted(stage_counter.items())),
    }
    (labels_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    log("INFO", "labels written", train=len(train_lines), val=len(val_lines), charset=len(charset))
    log(
        "INFO",
        "done",
        generated=counts["written"],
        skipped=counts["skipped"],
        failed=counts["failed"],
        out=str(out_root),
    )
    if counts["failed"]:
        return 1
    if len(rows) < 20000:
        log("WARN", "corpus is below the 20,000 gate G17 requires", have=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

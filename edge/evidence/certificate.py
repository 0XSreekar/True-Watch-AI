"""Bharatiya Sakshya Adhiniyam 2023, Section 63 certificate for sealed clips.

    python -m evidence.certificate --block 6 --out cert-6.md
    python -m evidence.certificate --from 2026-09-27 --to 2026-09-28 \
        --partA-name "..." --partA-designation "..." --partB-name "..." --place "..."

The legal wording lives in `templates/section63_certificate.md` so it can be edited without touching
code; this module only fills in the technical facts: the record, the device, the process, the hash
values (recomputed from disk now, not copied from the ledger), the Merkle inclusion proof for each
record (checked here with merkle.verify_inclusion), the Ed25519 signature on the day's root
(checked against the key registered at sector HQ), the logged gaps, and the signatory blocks.

The template's own note — that the wording must be reviewed by a qualified person before any real
use — is part of the template and is therefore carried into every generated certificate.

Exit status: 0 certificate written and every record verified; 1 certificate written but some
record failed verification or is not yet covered by a signed root (the certificate says so on its
face); 2 bad arguments or unreadable store.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import string
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from evidence.chain import DAY_TZ_LABEL, Ledger, ledger_path, to_rfc3339
from evidence.hasher import hash_file
from evidence.merkle import verify_inclusion
from evidence.mirror import HQMirror
from evidence.store import default_hq_dir, proof_for_block, read_meta, resolve_clip
from evidence.verify import StoreUnreadable, default_data_dir, verify_store

TEMPLATE_PATH = Path(__file__).with_name("templates") / "section63_certificate.md"
BLANK = "______________________"


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(_md_escape(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_certificate(
    root: str | os.PathLike,
    *,
    block_ns: list[int] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    hq_dir: str | os.PathLike | None = None,
    signatories: dict | None = None,
    device_description: str | None = None,
    template_path: str | os.PathLike | None = None,
    now: datetime | None = None,
) -> tuple[str, bool]:
    """Render the certificate. Returns (markdown, all_records_verified)."""
    root = Path(root)
    if not (root / "store.json").exists():
        raise StoreUnreadable(f"{root} is not an evidence store (no store.json)")
    meta = read_meta(root)
    device_id = meta["device_id"]
    blocks = Ledger(ledger_path(root)).blocks()
    if block_ns:
        wanted = set(block_ns)
        selected = [b for b in blocks if b.n in wanted]
        missing = wanted - {b.n for b in selected}
        if missing:
            raise KeyError(f"block(s) not in the ledger: {sorted(missing)}")
        scope = "block(s) " + ", ".join(str(b.n) for b in selected)
    else:
        selected = [b for b in blocks if (date_from is None or b.day >= date_from) and (date_to is None or b.day <= date_to)]
        scope = f"all records sealed on IST days {date_from or 'first'} to {date_to or 'last'}"
    if not selected:
        raise KeyError("no records match the selection")

    days = sorted({b.day for b in selected})
    report = verify_store(root, days[0], days[-1], hq_dir)
    result_by_n = {r.n: r for r in report.blocks}
    mirror = HQMirror(hq_dir or default_hq_dir(root), device_id)
    hq_key = mirror.registered_key_raw()

    record_rows, hash_rows, proof_parts = [], [], []
    all_ok = report.ok
    for b in selected:
        clip = resolve_clip(root, b)
        now_sha = hash_file(clip, frame_count=b.frame_count).sha256 if clip.is_file() else "(file missing)"
        res = result_by_n.get(b.n)
        ok = bool(res and res.ok)
        all_ok &= ok
        record_rows.append([
            str(b.n), b.clip_id, b.camera_id, b.started_at, b.ended_at, b.sealed_at,
            Path(b.clip_path).name, f"{b.byte_length} bytes", str(b.frame_count),
        ])
        hash_rows.append([
            str(b.n), b.clip_sha256, now_sha, b.prev_block_hash, b.block_hash,
            "VERIFIED" if ok else "FAILED: " + "; ".join(res.reasons if res else ["not checked"]),
        ])

        info = proof_for_block(root, b.n)
        proof, signed = info["proof"], info["signed_root"]
        lines = [f"### Inclusion proof and signature — block {b.n} (day {b.day})", ""]
        if signed is None:
            all_ok = False
            lines += [
                f"The day {b.day} has not been sealed yet, so no signed root covers this record. "
                "Re-issue this certificate after the day's root has been signed.",
                "",
            ]
        else:
            m = signed.manifest
            included = verify_inclusion(b.block_hash, proof, m.merkle_root, expected_leaf_count=m.leaf_count)
            sig_ok = hq_key is not None and signed.verify_with(hq_key)
            hq_copy = mirror.read_root(b.day)
            mirrored = hq_copy == signed
            all_ok &= included and sig_ok and mirrored
            lines += [
                _table(["Field", "Value"], [
                    ["Leaf index / leaf count", f"{proof.index} / {proof.leaf_count}"],
                    ["Leaf", f"SHA-256(0x00 ‖ {b.block_hash})"],
                    ["Signed daily Merkle root", m.merkle_root],
                    ["Blocks covered by the root", f"{m.first_n} to {m.last_n}"],
                    ["Proof checks against the signed root", "YES" if included else "NO"],
                    ["Root signed at", m.signed_at],
                    ["Ed25519 signature (hex)", signed.signature],
                    ["Signature valid under the HQ-registered key", "YES" if sig_ok else "NO"],
                    ["Identical copy held by the sector HQ mirror", "YES" if mirrored else "NO"],
                ]),
                "",
                "Sibling hashes, leaf level upward (node = SHA-256(0x01 ‖ left ‖ right); a level with no "
                "sibling promotes the node unchanged):",
                "",
                "```",
                *(f"{i}: {s}" for i, s in enumerate(proof.siblings)),
                *([] if proof.siblings else ["(single-leaf day: the leaf hash is the root)"]),
                "```",
                "",
                "Signed manifest (the exact bytes the signature covers):",
                "",
                "```",
                m.canonical().decode("ascii").rstrip("\n"),
                "```",
                "",
            ]
        proof_parts.append("\n".join(lines))

    gaps = report.gaps
    gap_table = (
        _table(["#", "Kind", "Start (UTC)", "End (UTC)", "Last block before", "Reason"],
               [[str(g.g), g.kind, g.start, g.end, str(g.last_block_n), g.reason] for g in gaps])
        if gaps else "No gaps were logged in the period covered by this certificate."
    )
    summary = _table(["Check", "Scope", "Result", "Detail"], [[c.name, c.scope, c.status, c.detail] for c in report.checks])
    if not all_ok:
        summary = (
            "**WARNING — VERIFICATION DID NOT FULLY PASS. This certificate must not be signed until the "
            "failures below are explained.**\n\n" + summary
        )
    sig = {k: (v or BLANK) for k, v in (signatories or {}).items()}
    now = now or datetime.now(timezone.utc)
    cert_id_src = "|".join([device_id, *(b.block_hash for b in selected), to_rfc3339(now)])
    values = {
        "certificate_id": "S63-" + hashlib.sha256(cert_id_src.encode()).hexdigest()[:16],
        "generated_at": to_rfc3339(now),
        "record_scope": scope,
        "partA_name": sig.get("partA_name", BLANK),
        "partA_designation": sig.get("partA_designation", BLANK),
        "partB_name": sig.get("partB_name", BLANK),
        "partB_designation": sig.get("partB_designation", BLANK),
        "place": sig.get("place", BLANK),
        "date": sig.get("date", BLANK),
        "device_id": device_id,
        "device_description": device_description or "TRUEWATCH edge inference board (make, model and serial number to be entered)",
        "key_id": (hq_key and hashlib.sha256(hq_key).hexdigest()[:16]) or "(no key registered at HQ)",
        "public_key_hex": hq_key.hex() if hq_key else "(none)",
        "hq_key_status": "yes" if hq_key else "NO — signatures cannot be verified",
        "day_tz": DAY_TZ_LABEL,
        "record_table": _table(
            ["Block", "Clip id", "Camera", "Recording start (UTC)", "Recording end (UTC)", "Sealed (UTC)", "File", "Size", "Frames"],
            record_rows,
        ),
        "hash_table": _table(
            ["Block", "Clip SHA-256 sealed at capture", "Clip SHA-256 recomputed now", "Previous block hash", "Block hash", "Result"],
            hash_rows,
        ),
        "verification_summary": summary,
        "verification_opinion": "" if all_ok else "**(Not applicable: verification failed — see Section 4.)**",
        "proof_section": "\n".join(proof_parts),
        "gap_table": gap_table,
    }
    template = Path(template_path or TEMPLATE_PATH).read_text("utf-8")
    return string.Template(template).safe_substitute(values), all_ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evidence.certificate", description="Generate a Section 63 (BSA 2023) certificate.")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--hq-dir", type=Path, default=None)
    p.add_argument("--block", type=int, action="append", help="block number (repeatable)")
    p.add_argument("--from", dest="date_from", type=lambda s: date.fromisoformat(s).isoformat())
    p.add_argument("--to", dest="date_to", type=lambda s: date.fromisoformat(s).isoformat())
    p.add_argument("--out", type=Path, default=None, help="write here instead of stdout")
    p.add_argument("--template", type=Path, default=None, help="alternative template file")
    p.add_argument("--device-description", default=None)
    for name in ("partA-name", "partA-designation", "partB-name", "partB-designation", "place", "date"):
        p.add_argument(f"--{name}", default=None)
    args = p.parse_args(argv)
    if not args.block and not (args.date_from or args.date_to):
        p.error("give --block N (repeatable) or a --from/--to date range")
    hq = args.hq_dir or (Path(os.environ["EVIDENCE_HQ_DIR"]) if os.environ.get("EVIDENCE_HQ_DIR") else None)
    signatories = {
        "partA_name": args.partA_name,
        "partA_designation": args.partA_designation,
        "partB_name": args.partB_name,
        "partB_designation": args.partB_designation,
        "place": args.place,
        "date": args.date,
    }
    try:
        text, ok = build_certificate(
            args.data_dir or default_data_dir(),
            block_ns=args.block,
            date_from=args.date_from,
            date_to=args.date_to,
            hq_dir=hq,
            signatories=signatories,
            device_description=args.device_description,
            template_path=args.template,
        )
    except (StoreUnreadable, KeyError, OSError) as exc:
        print(f"cannot generate certificate: {exc}", file=sys.stderr)
        return 2
    if args.out:
        args.out.write_text(text, "utf-8")
        print(f"wrote {args.out} ({'all records verified' if ok else 'VERIFICATION DID NOT FULLY PASS'})")
    else:
        sys.stdout.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

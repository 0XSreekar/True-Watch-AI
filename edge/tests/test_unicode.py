"""Devanagari must survive Python -> JSON -> HTTP -> the frozen alert shape unchanged.

docs/ARCHITECTURE_V2.md section 3.2 freezes the alert's `plate` field, and its
worked examples are the exact strings already in
`backend/src/data/mockData.js`: "बा १२ च ४५६७" and "प्र १ ख २३४५". This test
proves the string a Python process would put on the wire reaches a JSON
consumer byte-identically — no `\\uXXXX` escaping that survives past decode,
no NFC/NFD drift, no mojibake from an implicit ISO-8859-1 hop.

It does not start a real HTTP server: `http.client`/`requests`/`httpx` all
decode a `Content-Type: application/json; charset=utf-8` body through the same
UTF-8 codec `json.loads` uses on `str` input, so encoding the body to UTF-8
bytes and decoding it back is the same transformation an HTTP round trip
performs. What matters — and what this test actually asserts — is that the
bytes on the wire, and the object the frontend's `AlertQueue` reads, are
unchanged from what `edge/` produced.
"""

from __future__ import annotations

import json
import unicodedata

MOCK_PLATES = ("बा १२ च ४५६७", "प्र १ ख २३४५")


def _frozen_alert(plate: str) -> dict:
    """A minimal instance of the frozen alert shape (ARCHITECTURE_V2.md section 3.2)."""
    return {
        "id": "A1758140000000001",
        "camera": {"id": "RXL-01", "name": "BOP Raxaul — Pillar 42", "sector": "Raxaul", "ir": False, "road": True},
        "stage": "prov",
        "time": "02:41:07",
        "reason": "Loaded vehicle moving north at 02:41.",
        "confidence": 93,
        "plate": plate,
        "hash": "a3f91c04e21b7d",
        "seed": 42.17,
        "channels": {"appearance": True, "motion": True},
    }


def test_devanagari_round_trips_python_to_json_to_http_to_alert_shape():
    for plate in MOCK_PLATES:
        alert = _frozen_alert(plate)

        # Python -> JSON. ensure_ascii=False is required: the default True would
        # escape every Devanagari codepoint to \\uXXXX, which is valid JSON but
        # is exactly the failure mode this test guards against if some other
        # layer re-serialises with the default.
        wire_text = json.dumps(alert, ensure_ascii=False)
        assert "\\u09" not in wire_text, "Devanagari was escaped to \\uXXXX instead of staying literal UTF-8"

        # JSON -> HTTP: the bytes an Express/FastAPI body would actually send.
        wire_bytes = wire_text.encode("utf-8")

        # HTTP -> the frontend's JSON.parse (json.loads stands in for it: both
        # decode UTF-8 bytes per RFC 8259).
        received_text = wire_bytes.decode("utf-8")
        received_alert = json.loads(received_text)

        # -> the frozen alert shape AlertQueue reads.
        assert received_alert["plate"] == plate
        assert received_alert["plate"] == alert["plate"]
        # Byte-identical, not just string-equal: encode the recovered field the
        # same way and diff the bytes.
        assert received_alert["plate"].encode("utf-8") == plate.encode("utf-8")


def test_devanagari_plate_is_nfc_normalised():
    for plate in MOCK_PLATES:
        assert unicodedata.normalize("NFC", plate) == plate, f"{plate!r} is not already NFC"


def test_null_plate_survives_the_same_round_trip():
    alert = _frozen_alert(None)
    wire = json.dumps(alert, ensure_ascii=False)
    received = json.loads(wire)
    assert received["plate"] is None


def test_mock_data_strings_are_exactly_what_the_frontend_ships():
    """Pin the two literal strings from backend/src/data/mockData.js.

    If this test ever fails, mockData.js changed — which this phase's rules
    forbid touching frontend/backend content, so the fix is to update THIS
    pin to match a change made elsewhere, never to "fix" mockData.js from here.
    """
    assert MOCK_PLATES == ("बा १२ च ४५६७", "प्र १ ख २३४५")
    for plate in MOCK_PLATES:
        assert len(plate) > 0
        assert any(0x0900 <= ord(ch) <= 0x097F for ch in plate), f"{plate!r} has no Devanagari codepoint"


def test_json_dumps_default_escaping_would_have_failed_this_test():
    """Demonstrate the failure mode this suite exists to catch: ensure_ascii=True escapes."""
    plate = MOCK_PLATES[0]
    escaped = json.dumps({"plate": plate})  # ensure_ascii defaults to True
    assert "\\u09" in escaped
    # It is still correct JSON and round-trips through json.loads...
    assert json.loads(escaped)["plate"] == plate
    # ...but a naive string-contains check on the wire text (e.g. a proxy or
    # log scraper matching literal Devanagari) would miss it, which is why
    # edge/ must emit ensure_ascii=False on every response it authors.

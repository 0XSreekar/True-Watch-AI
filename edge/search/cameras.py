"""The camera roster search results are joined against.

Phase 8 wires the edge service to the backend's real `cameras` table
(docs/ARCHITECTURE_V2.md section 6). Until then this module is the same six
demo cameras backend/src/data/mockData.js ships, kept in the exact shape the
frozen contract requires: `{id, name, sector, ir, road}`. A clip whose
metadata names a camera_id outside this list still gets a result — `lookup`
falls back to a generic camera built from the id — so indexing never breaks
just because the roster hasn't caught up yet.
"""

from __future__ import annotations

CAMERAS: dict[str, dict] = {
    c["id"]: c
    for c in [
        {"id": "RXL-01", "name": "BOP Raxaul — Pillar 42", "sector": "Raxaul", "ir": False, "road": True},
        {"id": "JGB-03", "name": "Jogbani Ridge — Trail Head", "sector": "Jogbani", "ir": True, "road": False},
        {"id": "PNT-07", "name": "Panitanki — River Bank", "sector": "Panitanki", "ir": True, "road": True},
        {"id": "SNL-01", "name": "Sunauli Gate — Lane 2", "sector": "Sunauli", "ir": False, "road": True},
        {"id": "GLG-05", "name": "Galgalia — Culvert Approach", "sector": "Galgalia", "ir": True, "road": False},
        {"id": "PHU-02", "name": "Phuentsholing Line — Post 9", "sector": "Phuentsholing", "ir": False, "road": True},
    ]
}


def lookup(camera_id: str) -> dict:
    known = CAMERAS.get(camera_id)
    if known is not None:
        return dict(known)
    return {"id": camera_id, "name": camera_id, "sector": "Unknown", "ir": False, "road": True}

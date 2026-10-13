"""ANPR: Nepali Devanagari plate reading, with a Latin path for Bhutan posts.

MEASUREMENTS.md section 4 is the reason this package exists: off-the-shelf ANPR
stacks ship Latin (and sometimes Chinese) recognition models, so a Nepali plate
is not partially read at a Bhutan-facing post running the wrong model — it is
not read at all. PP-OCRv5 ships a Devanagari recognition head that off-the-shelf
LPR products do not wire up; this package wires it up and fine-tunes it on the
Phase 1 synthetic corpus (datasets/plates/).

Pipeline: detect_plate.py (locate a plate region inside a vehicle detection) ->
rectify.py (perspective correction, upscaling, two-line splitting) ->
recognise.py (script-routed Devanagari/Latin recognition) -> postprocess.py
(grammar validation, confidence, confusion correction).
"""

from __future__ import annotations

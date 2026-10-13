"""Make training/scripts importable as top-level modules, the way the scripts import each other."""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def pytest_unconfigure(config):
    """macOS only: skip native static destructors once the session has reported.

    With torch, onnxruntime and the Albumentations/OpenCV stack all loaded in one process, a
    native library's static destructor on macOS aborts at interpreter shutdown
    ("recursive_mutex lock failed"), after every test has already passed and been reported,
    turning a green run into exit 134. Linux (Kaggle) is unaffected. Exiting here with pytest's
    own status keeps the result honest: a failing session still exits non-zero.
    """
    if sys.platform != "darwin":
        return
    import os

    status = getattr(config, "_tw_exitstatus", None)
    if status is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(status))


def pytest_sessionfinish(session, exitstatus):
    session.config._tw_exitstatus = exitstatus

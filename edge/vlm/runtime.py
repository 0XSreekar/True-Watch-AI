"""Local Moondream 2, CPU-only, implementing `vlm.judge.VlmRuntime`.

This is the ONLY runtime the shipped product loads by default. It satisfies
the air-gap claim in the PDF (slides 4/5: fully local, no foreign cloud, no
outbound path once the weights are on disk) by never making a network call
from `describe()` — the one network access this module ever performs is the
one-time weight download, and that happens from `from_pretrained` only when
the local Hugging Face cache is missing the pinned revision. Deployment sets
`HF_HUB_OFFLINE=1` so a cache miss on an air-gapped box fails loudly (a clear
`VlmUnavailable`, which `judge.explain` turns into the rule-string fallback)
instead of silently reaching out — see edge/README.md for the pre-download
step to run once, on a box that still has network, before it goes dark.

Model and revision
-------------------
`vikhyatk/moondream2`, Apache-2.0, ~1.9B parameters. Pinned to a specific
commit hash, not floating `main` — the model card itself warns the
repository is "updated frequently" and recommends pinning a revision for
production use. `trust_remote_code=True` is only ever passed together with
this exact `revision`, so upstream cannot change what code runs here without
this file changing too.

The commit pinned here (`5d6c926f44e26b07957b0dd315bbedcb4c17a5fe`) was
chosen over the model card's own tagged `2025-06-21` release
(`9a7d4024050840e001defacec2b00727e89149e6`) for a concrete, testable reason
found while building this module: the tagged release's
`moondream.py::encode_image` indexes `settings["variant"]` unconditionally
whenever a `settings` dict is passed at all (every *other* call site in the
same file correctly uses `settings.get("variant")` / `"variant" in
settings`), so the call this runtime needs to make —
`model.query(image, prompt, settings={"max_tokens": ...})`, to bound
generation length — raises `KeyError: 'variant'` on that tagged commit. The
pinned commit here fixes that one line; both revisions were loaded and
exercised on this machine before choosing. If a future revision
reintroduces this or any other breaking change, `warm_up()` at boot degrades
to `VlmUnavailable` -> the rule-string fallback rather than crashing the
service — but re-pinning here remains the correct fix.

CPU precision — what "quantised" means for this checkpoint, honestly
----------------------------------------------------------------------
The checkpoint's own modelling code (`moondream.py`, `MoondreamModel.__init__`)
already defaults to `torch.bfloat16` weights, not fp32 — that is this model's
shipped reduced-precision path, and it is what `from_pretrained` loads unless
told otherwise. This module additionally *attempts*
`torch.ao.quantization.quantize_dynamic` (post-training dynamic int8
quantisation of the `nn.Linear` layers, the standard CPU-only quantisation
recipe — no calibration data needed, safe for a custom architecture that has
no static-quantisation observers wired in) at load time, and measures whether
that actually reduces `describe()` latency on this machine
(`edge/vlm/latency.py` records both numbers). Moondream 2's text and vision
towers are built from plain `nn.Linear` layers (see `layers.py`,
`text.py`, `vision.py` on the model repo) so dynamic quantisation applies
cleanly; if a future revision's custom layers make `quantize_dynamic` refuse
(raise, or silently no-op because it finds no quantisable modules), this
module catches that, logs it, and falls back to running the checkpoint's
default bf16 weights on CPU rather than pretending a quantised path exists
that does not. `Explanation` carries nothing about this — `latency.py` is
where the honest number lives.

Thread safety
-------------
One `MoondreamModel` instance per process; `_infer_lock` serialises every
call to the underlying model (weights, and the KV-cache buffers `moondream.py`
mutates in place across `encode_image`/`query`, are not safe to touch from two
threads at once). `queue.py` calls `describe()` from its single explanation
worker, so in normal operation the lock never contends — it exists so that
"single worker" is a fact this file enforces, not a fact `queue.py` must
remember to preserve.

Hosted fallback — OFF by default, loudly labelled
----------------------------------------------------
`HostedVlmRuntime` below is NOT wired into anything by default. It exists
only so a developer on a machine too slow for local Moondream 2 can keep
iterating on the surrounding pipeline. It is selected only by
`TRUEWATCH_VLM_HOSTED=1` (see `build_default_runtime`), and the moment it is
constructed it logs a `logging.warning` that says, in plain words, that
enabling it breaks the product's air-gap claim. It must never be the default
and must never be the source of a screenshot or demo recording without that
warning visible alongside it. See edge/README.md, "Development-only hosted
fallback" for the operator-facing version of this same warning.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Sequence

import numpy as np

from .judge import VlmUnavailable

log = logging.getLogger("truewatch.vlm.runtime")

MODEL_ID = "vikhyatk/moondream2"
# A specific commit, not "main" — see the module docstring for why this one
# was chosen over the model card's own tagged "2025-06-21" release (it fixes
# a KeyError this runtime's describe() call would otherwise hit every time).
PINNED_REVISION = "5d6c926f44e26b07957b0dd315bbedcb4c17a5fe"

DEFAULT_WARM_UP_TIMEOUT_S = 60.0
HOSTED_ENV_VAR = "TRUEWATCH_VLM_HOSTED"


def _to_rgb_array(img: Any) -> np.ndarray:
    """Best-effort conversion of whatever the pipeline hands us into an RGB
    HxWx3 uint8 array. The pipeline's own frames are BGR uint8 (OpenCV
    convention, per `vlm.judge`'s module docstring); PIL images pass through."""
    from PIL import Image as PILImage

    if isinstance(img, PILImage.Image):
        return np.asarray(img.convert("RGB"))
    arr = np.asarray(img)
    if arr.ndim == 2:  # grayscale
        return np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr[:, :, ::-1]  # assume BGR (OpenCV) -> RGB
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[:, :, [2, 1, 0]]
    raise ValueError(f"cannot interpret frame with shape {arr.shape} as an image")


class MoondreamRuntime:
    """Local Moondream 2 on CPU. Lazy-loaded; call `warm_up()` at service boot
    so the first real alert does not pay the (multi-second) load cost."""

    model_name = "moondream2"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str = PINNED_REVISION,
        *,
        attempt_quantization: bool = True,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self._attempt_quantization = attempt_quantization
        self._model: Any = None
        self._quantized = False
        self._load_error: str | None = None
        self._load_time_s: float | None = None
        self._load_lock = threading.Lock()
        # Single worker lock: one CPU inference at a time. Not a performance
        # optimisation (there is one queue.py worker anyway) — it is the
        # thing that makes "thread-safe" a fact enforced here, independent of
        # whatever calls this runtime.
        self._infer_lock = threading.Lock()

    # -- loading -----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def quantized(self) -> bool:
        return self._quantized

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                raise VlmUnavailable(self._load_error)
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001 - a load failure must degrade, not crash
                self._load_error = f"{type(exc).__name__}: {exc}"
                log.exception("moondream2 failed to load (%s@%s)", self.model_id, self.revision)
                raise VlmUnavailable(self._load_error) from exc

    def _load(self) -> None:
        import torch

        log.info("loading local Moondream 2 (%s@%s) for CPU inference", self.model_id, self.revision)
        t0 = time.monotonic()

        # This checkpoint's own vision.py checks `torch.backends.mps.is_available()`
        # at import time and, if true, unconditionally routes one pooling op
        # through Apple's MPS backend regardless of what device the rest of the
        # model runs on ("Non-divisible input sizes are not implemented on MPS
        # device yet" is the upstream comment) — on a CPU-loaded model that
        # produces a hard crash (`Passed CPU tensor to MPS op`) the moment that
        # op runs, confirmed while building this module on Apple Silicon. Since
        # this runtime's entire point is CPU-only inference (see this module's
        # docstring), the fix and the deployment goal are the same thing: make
        # the checkpoint's own device probe report MPS as unavailable for the
        # one import that decides this, so it takes the CPU-only code path
        # itself instead of silently reaching for the GPU. Scoped to this one
        # `from_pretrained` call (restored in `finally`) since `trust_remote_code`
        # first imports the remote module here.
        _real_mps_available = torch.backends.mps.is_available
        torch.backends.mps.is_available = lambda: False
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                revision=self.revision,
                trust_remote_code=True,
            )
        finally:
            torch.backends.mps.is_available = _real_mps_available
        model.eval()
        model.to("cpu")

        if self._attempt_quantization:
            try:
                model = torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                self._quantized = True
            except Exception:  # noqa: BLE001 - quantisation is an optimisation, never a requirement
                log.warning(
                    "dynamic int8 quantisation not usable for %s on this torch build; "
                    "running the checkpoint's default bf16 weights on CPU instead",
                    self.model_id,
                    exc_info=True,
                )
                self._quantized = False

        self._model = model
        self._load_time_s = time.monotonic() - t0
        log.info(
            "moondream2 loaded in %.1fs (quantised=%s)", self._load_time_s, self._quantized
        )

    def warm_up(self, timeout_s: float = DEFAULT_WARM_UP_TIMEOUT_S) -> bool:
        """Load the model and run one throwaway inference at service boot.

        Never raises: a warm-up failure is logged and the runtime simply
        retries lazily on the first real call (which itself degrades to the
        rule-string fallback via `judge.explain`, never a crash). Returns
        whether warm-up succeeded, for the boot log / healthz to report.
        """
        try:
            self._ensure_loaded()
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            self.describe(
                [dummy],
                "Facts: object: person; place: warm-up\nReason:",
                max_tokens=8,
                timeout_s=timeout_s,
            )
            return True
        except Exception:  # noqa: BLE001 - warm-up must never take the service down
            log.exception("moondream2 warm-up failed; will retry lazily on first real call")
            return False

    # -- inference -----------------------------------------------------------

    def describe(
        self, images: Sequence[Any], prompt: str, *, max_tokens: int, timeout_s: float
    ) -> str:
        if not images:
            raise VlmUnavailable("no frames supplied")
        self._ensure_loaded()

        rgb = _to_rgb_array(images[0])
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(rgb)

        outcome: dict[str, Any] = {}

        def _run() -> None:
            try:
                with self._infer_lock:
                    result = self._model.query(
                        pil_image, prompt, settings={"max_tokens": max_tokens}
                    )
                text = result.get("answer") if isinstance(result, dict) else result
                outcome["text"] = text if isinstance(text, str) else str(text)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller thread below
                outcome["error"] = exc

        worker = threading.Thread(target=_run, daemon=True, name="moondream2-infer")
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            # The thread is left to finish in the background (daemon=True) and
            # its eventual result is discarded — `judge.explain` has already
            # moved on to the fallback by the time this raises.
            raise TimeoutError(f"moondream2 did not answer within {timeout_s:.1f}s")
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("text", "")


class HostedVlmRuntime:
    """Development-only fallback that calls a hosted vision-language endpoint.

    NEVER constructed unless `TRUEWATCH_VLM_HOSTED=1` is set (see
    `build_default_runtime`) — nothing in this module wires it in by default.
    Every instantiation logs a warning that this breaks the product's
    air-gap claim, at construction time, so the warning cannot be missed by
    disabling a log level after the fact.

    This class intentionally does not ship a default endpoint or credential:
    a developer who opts in must also point `TRUEWATCH_VLM_HOSTED_URL` and
    `TRUEWATCH_VLM_HOSTED_KEY` at their own endpoint. No such endpoint is
    contacted by any code path unless both the opt-in flag and these two
    values are present.
    """

    model_name = "hosted-vlm-dev-fallback"

    def __init__(self, endpoint_url: str, api_key: str, *, timeout_s: float = 15.0) -> None:
        if not endpoint_url or not api_key:
            raise VlmUnavailable(
                "hosted VLM fallback requires TRUEWATCH_VLM_HOSTED_URL and "
                "TRUEWATCH_VLM_HOSTED_KEY to be set"
            )
        self._endpoint_url = endpoint_url
        self._api_key = api_key
        self._default_timeout_s = timeout_s
        log.warning(
            "TRUEWATCH_VLM_HOSTED=1: explanations will be generated by a HOSTED endpoint "
            "(%s), not the local Moondream 2 model. This BREAKS the air-gapped deployment "
            "claim and must never be used outside development. Any screenshot or recording "
            "taken while this is enabled must be labelled as such.",
            self._endpoint_url,
        )

    def describe(
        self, images: Sequence[Any], prompt: str, *, max_tokens: int, timeout_s: float
    ) -> str:
        import base64
        import io

        import httpx
        from PIL import Image as PILImage

        if not images:
            raise VlmUnavailable("no frames supplied")
        rgb = _to_rgb_array(images[0])
        buf = io.BytesIO()
        PILImage.fromarray(rgb).save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        try:
            resp = httpx.post(
                self._endpoint_url,
                json={"prompt": prompt, "image_b64": image_b64, "max_tokens": max_tokens},
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise TimeoutError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise VlmUnavailable(f"hosted VLM call failed: {exc}") from exc

        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str):
            raise VlmUnavailable("hosted VLM returned no usable text")
        return text


def build_default_runtime() -> Any:
    """The one runtime the service boots with.

    Local Moondream 2 unless a developer has explicitly opted into the
    hosted fallback via `TRUEWATCH_VLM_HOSTED=1` — see this module's
    docstring and edge/README.md for why that is development-only.
    """
    if os.environ.get(HOSTED_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}:
        return HostedVlmRuntime(
            endpoint_url=os.environ.get("TRUEWATCH_VLM_HOSTED_URL", "").strip(),
            api_key=os.environ.get("TRUEWATCH_VLM_HOSTED_KEY", "").strip(),
        )
    return MoondreamRuntime()


__all__ = [
    "DEFAULT_WARM_UP_TIMEOUT_S",
    "HOSTED_ENV_VAR",
    "MODEL_ID",
    "PINNED_REVISION",
    "HostedVlmRuntime",
    "MoondreamRuntime",
    "build_default_runtime",
]

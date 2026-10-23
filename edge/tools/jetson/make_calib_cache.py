"""DEVICE: write the INT8 calibration cache for the pinned ONNX detector from the calibration images.

trtexec can consume a calibration cache but cannot make one from images, so this uses the TensorRT
Python API (entropy calibrator 2) with the same letterbox and scaling the edge pipeline feeds the
detector (pipeline.appearance.letterbox / to_blob), so calibration sees what inference will see.

  python edge/tools/jetson/make_calib_cache.py --onnx final.onnx --images calib/ --out calib.cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from tools.jetson.common import DEFAULT_MANIFEST, load_manifest, verify_onnx  # noqa: E402


def list_images(folder: Path, list_file: Path | None) -> list[Path]:
    if list_file:
        return [folder / line.split("\t")[0] for line in list_file.read_text().splitlines() if line.strip()]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="folder of calibration images (the extracted tarball)")
    ap.add_argument("--list", type=Path, default=None, help="calib_list.txt; default every image in --images")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--no-verify", action="store_true", help="skip the sha256 check against the pinned manifest")
    args = ap.parse_args(argv)

    import cv2
    import numpy as np
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
    from pipeline.appearance import letterbox, to_blob

    manifest = load_manifest(args.manifest)
    if not args.no_verify:
        verify_onnx(args.onnx, manifest)
    imgsz = int(manifest["imgsz"])
    files = list_images(args.images, args.list)
    if not files:
        raise SystemExit("no calibration images found")

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.i = 0
            self.dev = cuda.mem_alloc(3 * imgsz * imgsz * 4)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            while self.i < len(files):
                img = cv2.imread(str(files[self.i]))
                self.i += 1
                if img is not None:
                    blob = np.ascontiguousarray(to_blob([letterbox(img, imgsz)[0]]), dtype=np.float32)
                    cuda.memcpy_htod(self.dev, blob)
                    return [int(self.dev)]
            return None

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            args.out.write_bytes(cache)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(args.onnx.read_bytes()):
        raise SystemExit("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.int8_calibrator = Calib()
    prof = builder.create_optimization_profile()
    shape = (1, 3, imgsz, imgsz)
    prof.set_shape(net.get_input(0).name, shape, shape, shape)
    cfg.add_optimization_profile(prof)
    cfg.set_calibration_profile(prof)
    if builder.build_serialized_network(net, cfg) is None:
        raise SystemExit("calibration build failed")
    print(f"wrote {args.out} from {len(files)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

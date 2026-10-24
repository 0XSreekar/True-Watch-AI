"""A TensorRT engine behind the onnxruntime session interface the edge pipeline already uses
(`get_inputs`, `run`, `get_modelmeta`, `get_providers`), so `pipeline.appearance.AppearanceChannel`
runs on the Jetson unchanged. DEVICE ONLY: needs the `tensorrt` module from JetPack and `pycuda`.
Imports are lazy so this file can be imported (and its metadata helper tested) anywhere.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np


def meta_map(class_names: list[str]) -> dict[str, str]:
    """The `custom_metadata_map` appearance.model_class_names reads: a JSON list under 'class_names'."""
    return {"class_names": json.dumps(list(class_names))} if class_names else {}


class TrtSession:
    def __init__(self, engine_path, class_names: list[str] | None = None) -> None:
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as fh, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {engine_path}: built by a different TensorRT version or GPU?")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.ctx.set_input_shape(self.in_name, self.in_shape)
        self.stream = cuda.Stream()
        self.d_in = cuda.mem_alloc(int(np.prod(self.in_shape)) * 4)
        self.h_out = cuda.pagelocked_empty(self.out_shape, trt.nptype(self.engine.get_tensor_dtype(self.out_name)))
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self.ctx.set_tensor_address(self.in_name, int(self.d_in))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out))
        self._names = class_names or []

    # ---- onnxruntime-shaped surface
    def get_inputs(self):
        return [SimpleNamespace(name=self.in_name, shape=list(self.in_shape), type="tensor(float)")]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=meta_map(self._names))

    def get_providers(self):
        return ["TensorRT-engine"]

    def run(self, _outputs, feed):
        x = np.ascontiguousarray(feed[self.in_name], dtype=np.float32)
        cuda = self._cuda
        cuda.memcpy_htod_async(self.d_in, x, self.stream)
        self.ctx.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return [np.array(self.h_out, dtype=np.float32, copy=True)]

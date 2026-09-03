"""Server tail"""

import asyncio
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .config import ROOT, Config

try:  # Pi: aarch64 wheel
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # dev box
    from tensorflow.lite.python.interpreter import Interpreter


@dataclass
class TailResult:
    pred_class: int
    t_queue_in: float
    t_queue_out: float
    t_tail_us: int
    t_done: float


class Tail:
    def __init__(self, cfg: Config, num_threads: int = 2, sim_extra_ms: dict | None = None):
        art = ROOT / cfg.model.artifacts_dir
        self.cfg = cfg
        self.sim_extra_ms = sim_extra_ms  # dev-host Pi-speed emulation (config sim.tail_extra_ms)
        self.cuts = json.loads((art / "cuts.json").read_text())
        self.interp = {}
        self.input_detail = {}
        self.output_detail = {}
        for cut in cfg.cuts:
            path = art / ("model.tflite" if cut.boundary_op is None else f"tail_{cut.name}.tflite")
            it = Interpreter(model_path=str(path), num_threads=num_threads)
            it.allocate_tensors()
            self.interp[cut.name] = it
            self.input_detail[cut.name] = it.get_input_details()[0]
            self.output_detail[cut.name] = it.get_output_details()[0]
        self.queue: asyncio.Queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)  # FIFO, one at a time
        self._worker = None

    def start(self):
        self._worker = asyncio.ensure_future(self._run())

    def stop(self):
        if self._worker:
            self._worker.cancel()
        self.executor.shutdown(wait=False)

    async def infer(self, arm: str, blob: bytes) -> TailResult:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((arm, blob, time.monotonic(), fut))
        return await fut

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            arm, blob, t_queue_in, fut = await self.queue.get()
            t_queue_out = time.monotonic()
            try:
                pred, t_tail_us = await loop.run_in_executor(self.executor, self._infer_sync, arm, blob)
                fut.set_result(TailResult(pred, t_queue_in, t_queue_out, t_tail_us, time.monotonic()))
            except Exception as e:
                if not fut.done():
                    fut.set_exception(e)

    def _infer_sync(self, arm: str, blob: bytes) -> tuple[int, int]:
        t0 = time.monotonic_ns()
        it = self.interp[arm]
        detail = self.input_detail[arm]
        if arm == self.cfg.cuts[0].name:  # k0: JPEG in, full model
            img = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.float32)
            scale, zp = detail["quantization"]
            x = np.clip(np.round(img / scale + zp), -128, 127).astype(np.int8)[None]
        else:
            x = np.frombuffer(blob, dtype=np.int8).reshape(detail["shape"])
        it.set_tensor(detail["index"], x)
        it.invoke()
        out = it.get_tensor(self.output_detail[arm]["index"])
        if self.sim_extra_ms:
            time.sleep(self.sim_extra_ms.get(arm, 0.0) / 1000)
        return int(np.argmax(out[0])), (time.monotonic_ns() - t0) // 1000

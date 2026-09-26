"""Benchmark the detector backends on one image.

    python bench.py [image] [runs]

Runs the fast pass (longest side 640) and a range-pass tile through every
model file that exists (yolo26n.pt, yolo26n.onnx), prints the average
inference time and checks that the backends agree on the boxes. Use it on the
target machine (e.g. `fly ssh console -C "python bench.py"`) to see what one
inference really costs there — that number decides how many streams fit.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "frame.jpg"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
MODELS = [m for m in ("yolo26n.pt", "yolo26n.onnx") if m.endswith(".pt") or os.path.exists(m)]


def inputs(frame):
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    fast = cv2.resize(frame, (int(w * scale), int(h * scale)))
    tile = frame[: int(h * 0.55), : int(w * 0.55)]
    return {"fast pass": fast, "range tile": tile}


def main() -> None:
    frame = cv2.imread(IMAGE)
    if frame is None:
        raise SystemExit(f"Cannot read {IMAGE}")
    print(f"{IMAGE}: {frame.shape[1]}x{frame.shape[0]}, {RUNS} runs, {os.cpu_count()} CPU(s)")
    boxes: dict[tuple[str, str], np.ndarray] = {}
    for path in MODELS:
        model = YOLO(path, task="detect")
        for name, img in inputs(frame).items():
            for _ in range(3):
                model.predict(img, conf=0.15, verbose=False)
            started = time.perf_counter()
            for _ in range(RUNS):
                result = model.predict(img, conf=0.15, verbose=False)[0]
            ms = (time.perf_counter() - started) / RUNS * 1000
            boxes[(path, name)] = result.boxes.xyxy.cpu().numpy()
            print(f"  {path:14s} {name:10s} {ms:6.1f} ms  {len(result.boxes)} box(es)")
    if len(MODELS) > 1:
        for name in inputs(frame):
            a, b = boxes[(MODELS[0], name)], boxes[(MODELS[1], name)]
            same = a.shape == b.shape and np.allclose(np.sort(a, 0), np.sort(b, 0), atol=1.0)
            print(f"  {name}: backends {'agree' if same else 'DIFFER'}")


if __name__ == "__main__":
    main()

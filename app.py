"""atlas-detector — continuous object detection and tracking on Atlas drone video.

Reads one MediaMTX stream over RTSP, runs a detector (YOLOv8n by default, or
Grounding DINO for open-vocabulary text-prompted detection) plus ByteTrack on a
subset of the frames, and upserts one row per tracked object into the Supabase
table `atlas_detections` (unique on flight_session_id + track_id). A cleanup
loop deletes tracks that stopped being updated.

CPU only. Everything is configured through environment variables:

  MEDIAMTX_RTSP_URL             rtsp://live-video.internal:8554/<serial>/<sensor>?detector=<secret>
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  FLIGHT_SESSION_ID             active_flights.id the detections belong to
  DETECTION_ENGINE              yolo (default) | grounding_dino
  DETECTION_FPS                 analysis rate for yolo, default 10
  DETECTION_FPS_GROUNDING_DINO  analysis rate for grounding_dino, default 0.2
  DETECTION_CLASSES             yolo only, default person,car,truck,bus,motorcycle,boat
  DETECTION_CONFIDENCE          yolo only, default 0.30
  DETECTION_TEXT_PROMPT         grounding_dino only, "person . car . boat . ..."
  DETECTION_BOX_THRESHOLD       grounding_dino only, default 0.25
  DETECTION_TEXT_THRESHOLD      grounding_dino only, default 0.20
  DETECTION_FPS_TILED           grounding_dino only, tiled pass rate, default 1.5
  TILE_GRID                     tiled grid rows x cols, default 2x3
  TILE_OVERLAP                  tile overlap fraction, default 0.15
  INFER_MAX_SIDE                downscale longest side before inference, default 480
  TRACKER_LOST_BUFFER           analysed frames a lost track survives, default 5
  TRACK_TTL_SECONDS             default 0.8
"""


from __future__ import annotations

import itertools
import json

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import supervision as sv
from supabase import create_client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("atlas-detector")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RTSP_URL = os.environ.get("MEDIAMTX_RTSP_URL", "").strip()
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
FLIGHT_SESSION_ID = os.environ.get("FLIGHT_SESSION_ID", "").strip()

DETECTION_ENGINE = (os.environ.get("DETECTION_ENGINE", "yolo") or "yolo").strip().lower()

DETECTION_FPS = float(os.environ.get("DETECTION_FPS", "10") or 10)
DETECTION_FPS_GROUNDING_DINO = float(
    os.environ.get("DETECTION_FPS_GROUNDING_DINO", "0.2") or 0.2
)
DETECTION_CONFIDENCE = float(os.environ.get("DETECTION_CONFIDENCE", "0.30") or 0.30)
TRACK_TTL_SECONDS = float(os.environ.get("TRACK_TTL_SECONDS", "0.8") or 0.8)
# Downscale before inference: the single biggest latency win on CPU. 0 = off.
INFER_MAX_SIDE = int(os.environ.get("INFER_MAX_SIDE", "480") or 480)
# How many analysed frames a lost track survives inside the tracker.
TRACKER_LOST_BUFFER = int(os.environ.get("TRACKER_LOST_BUFFER", "5") or 5)
DETECTION_CLASSES = [
    c.strip().lower()
    for c in os.environ.get(
        "DETECTION_CLASSES", "person,car,truck,bus,motorcycle,boat"
    ).split(",")
    if c.strip()
]

# Grounding DINO expects categories separated by " . ".
DETECTION_TEXT_PROMPT = (
    os.environ.get(
        "DETECTION_TEXT_PROMPT", "person . car . boat . truck . bus . motorcycle"
    ).strip()
    or "person . car . boat . truck . bus . motorcycle"
)
DETECTION_BOX_THRESHOLD = float(os.environ.get("DETECTION_BOX_THRESHOLD", "0.25") or 0.25)
DETECTION_TEXT_THRESHOLD = float(os.environ.get("DETECTION_TEXT_THRESHOLD", "0.20") or 0.20)

# --- Tiled full-resolution pass (Grounding DINO only) ----------------------- #
# Runs beside the fast downscaled pass and analyses overlapping full-resolution
# tiles in parallel, so small/distant objects survive to the model.
DETECTION_FPS_TILED = float(os.environ.get("DETECTION_FPS_TILED", "1.5") or 1.5)
TILE_GRID = (os.environ.get("TILE_GRID", "2x3") or "2x3").strip().lower()
TILE_OVERLAP = float(os.environ.get("TILE_OVERLAP", "0.15") or 0.15)
TILE_IOU_THRESHOLD = float(os.environ.get("TILE_IOU_THRESHOLD", "0.5") or 0.5)

# --- Search-and-track ROIs -------------------------------------------------- #
# The tiled pass never writes rows: it only nominates regions of interest that
# the fast pass then re-checks at full resolution, so a single tracker owns
# every box and nothing is ever drawn twice.
ROI_PADDING = float(os.environ.get("ROI_PADDING", "0.2") or 0.2)
ROI_CONFIRM_FRAMES = int(os.environ.get("ROI_CONFIRM_FRAMES", "5") or 5)
ROI_MISS_LIMIT = int(os.environ.get("ROI_MISS_LIMIT", "10") or 10)
ROI_MAX = int(os.environ.get("ROI_MAX", "8") or 8)
ROI_IOU_MATCH = float(os.environ.get("ROI_IOU_MATCH", "0.3") or 0.3)



def _parse_grid(value: str) -> tuple[int, int]:
    try:
        rows, cols = value.split("x")
        return max(1, int(rows)), max(1, int(cols))
    except Exception:
        log.warning("Invalid TILE_GRID '%s', falling back to 2x3", value)
        return 2, 3


TILE_ROWS, TILE_COLS = _parse_grid(TILE_GRID)

MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")
GROUNDING_DINO_CHECKPOINT = os.environ.get(
    "GROUNDING_DINO_CHECKPOINT", "IDEA-Research/grounding-dino-tiny"
)

# Backoff bounds for reconnecting to MediaMTX.
BACKOFF_MIN = 1.0
BACKOFF_MAX = 30.0

# Force TCP for RTSP — UDP is unreliable across the Fly private network.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

ANALYSIS_FPS = (
    DETECTION_FPS_GROUNDING_DINO if DETECTION_ENGINE == "grounding_dino" else DETECTION_FPS
)
MIN_FRAME_INTERVAL = 1.0 / ANALYSIS_FPS if ANALYSIS_FPS > 0 else 0.2

TILED_ENABLED = DETECTION_ENGINE == "grounding_dino" and DETECTION_FPS_TILED > 0
TILED_FRAME_INTERVAL = 1.0 / DETECTION_FPS_TILED if DETECTION_FPS_TILED > 0 else 0.0



# --------------------------------------------------------------------------- #
# Shared status (for /health)
# --------------------------------------------------------------------------- #


class Status:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.connected = False
        self.last_frame_at: float | None = None
        self.active_tracks = 0
        self.active_rois = 0
        self.reconnects = 0
        self.started_at = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            last = self.last_frame_at
            return {
                "ok": True,
                "service": "atlas-detector",
                "connected": self.connected,
                "last_frame_age_seconds": None if last is None else round(time.time() - last, 2),
                "active_tracks": self.active_tracks,
                "active_rois": self.active_rois,
                "reconnects": self.reconnects,
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "flight_session_id": FLIGHT_SESSION_ID or None,
                "engine": DETECTION_ENGINE,
                "detection_fps": ANALYSIS_FPS,
                "tiled_enabled": TILED_ENABLED,
                "tiled_fps": DETECTION_FPS_TILED if TILED_ENABLED else None,
                "tile_grid": f"{TILE_ROWS}x{TILE_COLS}" if TILED_ENABLED else None,
                "classes": DETECTION_CLASSES,
                "text_prompt": (
                    DETECTION_TEXT_PROMPT if DETECTION_ENGINE == "grounding_dino" else None
                ),
            }



status = Status()


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] not in ("/health", "/"):
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(status.snapshot()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence per-request logging
        return


def start_health_server() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8080), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True, name="health").start()
    log.info("Health server listening on :8080")


# --------------------------------------------------------------------------- #
# Detectors — both return (sv.Detections, list[str] labels aligned by index)
# --------------------------------------------------------------------------- #


class YoloDetector:
    """YOLOv8n with a fixed COCO class list. Unchanged default behaviour."""

    name = "yolo"

    def __init__(self) -> None:
        from ultralytics import YOLO

        log.info("Loading YOLOv8n (%s)", MODEL_PATH)
        self.model = YOLO(MODEL_PATH)

        names: dict[int, str] = {int(k): str(v).lower() for k, v in self.model.names.items()}
        self.class_ids = {cid: n for cid, n in names.items() if n in DETECTION_CLASSES}
        unknown = set(DETECTION_CLASSES) - set(self.class_ids.values())
        if unknown:
            log.warning("Unknown classes ignored: %s", ", ".join(sorted(unknown)))
        if not self.class_ids:
            raise SystemExit("No valid classes in DETECTION_CLASSES")

    def describe(self) -> str:
        return (
            f"engine=yolo model={MODEL_PATH} "
            f"classes={','.join(sorted(self.class_ids.values()))} "
            f"confidence={DETECTION_CONFIDENCE} fps={ANALYSIS_FPS}"
        )

    def detect(self, frame) -> tuple[sv.Detections, list[str]]:
        result = self.model.predict(
            frame,
            conf=DETECTION_CONFIDENCE,
            classes=sorted(self.class_ids.keys()),
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        class_ids = (
            detections.class_id
            if detections.class_id is not None
            else np.full(len(detections), -1)
        )
        labels = [self.class_ids.get(int(cid), "") for cid in class_ids]
        return detections, labels


class GroundingDinoDetector:
    """Open-vocabulary detection driven by a free-text prompt (CPU, Swin-T)."""

    name = "grounding_dino"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        log.info("Loading Grounding DINO (%s)", GROUNDING_DINO_CHECKPOINT)
        self.processor = AutoProcessor.from_pretrained(GROUNDING_DINO_CHECKPOINT)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_CHECKPOINT
        )
        self.model.eval()
        # Grounding DINO wants lowercase categories ending with a period.
        prompt = DETECTION_TEXT_PROMPT.strip().lower()
        self.prompt = prompt if prompt.endswith(".") else prompt + " ."

        self._pool: ThreadPoolExecutor | None = None
        if TILED_ENABLED:
            # One inference per tile, all at once. PyTorch's own intra-op
            # threading would otherwise fight the pool for the same cores.
            torch.set_num_threads(1)
            self._pool = ThreadPoolExecutor(
                max_workers=TILE_ROWS * TILE_COLS, thread_name_prefix="tile"
            )
            log.info(
                "Tiled pass enabled: grid=%dx%d overlap=%.2f fps=%.2f",
                TILE_ROWS,
                TILE_COLS,
                TILE_OVERLAP,
                DETECTION_FPS_TILED,
            )

    def describe(self) -> str:
        return (
            f"engine=grounding_dino checkpoint={GROUNDING_DINO_CHECKPOINT} "
            f"prompt=\"{self.prompt}\" box_threshold={DETECTION_BOX_THRESHOLD} "
            f"text_threshold={DETECTION_TEXT_THRESHOLD} fps={ANALYSIS_FPS} "
            f"tiled={'on' if TILED_ENABLED else 'off'} "
            f"tile_grid={TILE_ROWS}x{TILE_COLS} tile_overlap={TILE_OVERLAP} "
            f"tile_fps={DETECTION_FPS_TILED}"
        )

    def _infer(self, image) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Run the model on one BGR image. Returns (xyxy, scores, labels)."""
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, text=self.prompt, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=DETECTION_BOX_THRESHOLD,
            text_threshold=DETECTION_TEXT_THRESHOLD,
            target_sizes=[(height, width)],
        )[0]

        boxes = results["boxes"].cpu().numpy().astype(np.float32).reshape(-1, 4)
        scores = results["scores"].cpu().numpy().astype(np.float32).reshape(-1)
        labels = [str(t).strip().lower() for t in results.get("text_labels", results["labels"])]
        return boxes, scores, labels

    def detect(self, frame) -> tuple[sv.Detections, list[str]]:
        boxes, scores, labels = self._infer(frame)
        if len(boxes) == 0:
            return sv.Detections.empty(), []
        return (
            sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=np.zeros(len(boxes), dtype=int),
            ),
            labels,
        )

    def detect_tiled(self, frame) -> tuple[sv.Detections, list[str]]:
        """Analyse overlapping full-resolution tiles in parallel.

        Each tile keeps its native pixel density, so distant objects that the
        downscaled fast pass loses are still large enough for the model.
        """
        if self._pool is None:
            return sv.Detections.empty(), []

        height, width = frame.shape[:2]
        tiles = _tile_boxes(width, height, TILE_ROWS, TILE_COLS, TILE_OVERLAP)

        def work(box: tuple[int, int, int, int]):
            x0, y0, x1, y1 = box
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), []
            b, s, l = self._infer(crop)
            if len(b):
                b = b.copy()
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
            return b, s, l

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_labels: list[str] = []
        for boxes, scores, labels in self._pool.map(work, tiles):
            if len(boxes) == 0:
                continue
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.extend(labels)

        if not all_boxes:
            return sv.Detections.empty(), []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        keep = _class_aware_nms(boxes, scores, all_labels, TILE_IOU_THRESHOLD)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = [all_labels[i] for i in keep]

        # Clamp to the frame so tiles at the edges cannot report outside it.
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

        return (
            sv.Detections(
                xyxy=boxes.astype(np.float32),
                confidence=scores.astype(np.float32),
                class_id=np.zeros(len(boxes), dtype=int),
            ),
            labels,
        )


def _tile_boxes(
    width: int, height: int, rows: int, cols: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Grid of overlapping (x0, y0, x1, y1) tiles covering the whole frame."""
    tile_w = width / cols
    tile_h = height / rows
    pad_x = tile_w * overlap
    pad_y = tile_h * overlap
    boxes: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(max(0, round(c * tile_w - pad_x)))
            y0 = int(max(0, round(r * tile_h - pad_y)))
            x1 = int(min(width, round((c + 1) * tile_w + pad_x)))
            y1 = int(min(height, round((r + 1) * tile_h + pad_y)))
            if x1 > x0 and y1 > y0:
                boxes.append((x0, y0, x1, y1))
    return boxes


def _class_aware_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: list[str], iou_threshold: float
) -> list[int]:
    """Greedy NMS per label — removes duplicates from tile overlap regions."""
    order = sorted(range(len(boxes)), key=lambda i: float(scores[i]), reverse=True)
    keep: list[int] = []
    for idx in order:
        drop = False
        for kept in keep:
            if labels[kept] != labels[idx]:
                continue
            if _iou(boxes[idx], boxes[kept]) > iou_threshold:
                drop = True
                break
        if not drop:
            keep.append(idx)
    return sorted(keep)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0 = max(float(a[0]), float(b[0]))
    iy0 = max(float(a[1]), float(b[1]))
    ix1 = min(float(a[2]), float(b[2]))
    iy1 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Search-and-track: shared regions of interest
# --------------------------------------------------------------------------- #


def _pad_box(
    box, width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    """Grow a box by `padding` on each side, clamped to the frame."""
    x0, y0, x1, y1 = (float(v) for v in box)
    pad_x = (x1 - x0) * padding
    pad_y = (y1 - y0) * padding
    return (
        int(max(0, round(x0 - pad_x))),
        int(max(0, round(y0 - pad_y))),
        int(min(width, round(x1 + pad_x))),
        int(min(height, round(y1 + pad_y))),
    )


class RoiRegistry:
    """Thread-safe set of regions the tiled pass wants re-checked up close.

    The tiled thread writes candidates; the fast loop reads them, crops them at
    full resolution and reports hits/misses so a region retires once the main
    tracker owns the object (or once it turns out to be nothing).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rois: list[dict] = []
        self._last_boxes: np.ndarray = np.zeros((0, 4), dtype=np.float32)

    # -- fast pass ---------------------------------------------------------- #

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._rois]

    def set_last_boxes(self, boxes: np.ndarray) -> None:
        """Boxes the fast pass just tracked, in full-frame coordinates."""
        with self._lock:
            self._last_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    def report(self, results: dict[int, bool]) -> int:
        """Record per-ROI hit/miss and drop the ones that are done."""
        with self._lock:
            kept: list[dict] = []
            for roi in self._rois:
                outcome = results.get(roi["id"])
                if outcome is True:
                    roi["hits"] += 1
                    roi["misses"] = 0
                elif outcome is False:
                    roi["misses"] += 1
                if roi["hits"] >= ROI_CONFIRM_FRAMES:
                    continue  # the main tracker owns this object now
                if roi["misses"] >= ROI_MISS_LIMIT:
                    continue  # false lead or the object left
                kept.append(roi)
            self._rois = kept
            return len(self._rois)

    # -- tiled pass --------------------------------------------------------- #

    def add_candidates(self, boxes: np.ndarray, width: int, height: int) -> int:
        """Add tiled hits that no ROI and no tracked box already covers."""
        added = 0
        with self._lock:
            known = [np.array(r["box"], dtype=np.float32) for r in self._rois]
            known.extend(self._last_boxes)
            for box in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
                if len(self._rois) >= ROI_MAX:
                    break
                if any(_iou(box, other) > ROI_IOU_MATCH for other in known):
                    continue
                padded = _pad_box(box, width, height, ROI_PADDING)
                if padded[2] <= padded[0] or padded[3] <= padded[1]:
                    continue
                roi = {
                    "id": next(self._counter),
                    "box": padded,
                    "added_at": time.time(),
                    "hits": 0,
                    "misses": 0,
                }
                self._rois.append(roi)
                known.append(np.array(padded, dtype=np.float32))
                added += 1
            return added

    _counter = itertools.count(1)


def detect_in_rois(detector, frame, rois: list[dict], pool):
    """Run the detector on each ROI crop at native resolution.

    Returns (boxes in full-frame coordinates, scores, labels, per-ROI outcome).
    """
    if not rois:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
            {},
        )

    def work(roi: dict):
        x0, y0, x1, y1 = roi["box"]
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return roi["id"], np.zeros((0, 4), dtype=np.float32), np.zeros(0), []
        detections, labels = detector.detect(crop)
        boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
        scores = (
            np.asarray(detections.confidence, dtype=np.float32)
            if detections.confidence is not None
            else np.zeros(len(boxes), dtype=np.float32)
        )
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
        return roi["id"], boxes, scores, labels

    if pool is not None and len(rois) > 1:
        results = list(pool.map(work, rois))
    else:
        results = [work(roi) for roi in rois]

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[str] = []
    outcome: dict[int, bool] = {}
    for roi_id, boxes, scores, labels in results:
        outcome[roi_id] = len(boxes) > 0
        if len(boxes) == 0:
            continue
        all_boxes.append(boxes)
        all_scores.append(np.asarray(scores, dtype=np.float32).reshape(-1))
        all_labels.extend(labels)

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
            outcome,
        )
    return (
        np.concatenate(all_boxes, axis=0),
        np.concatenate(all_scores, axis=0),
        all_labels,
        outcome,
    )



def build_detector():
    if DETECTION_ENGINE == "yolo":
        return YoloDetector()
    if DETECTION_ENGINE == "grounding_dino":
        return GroundingDinoDetector()
    raise SystemExit(
        f"Unknown DETECTION_ENGINE '{DETECTION_ENGINE}' — use 'yolo' or 'grounding_dino'"
    )


# --------------------------------------------------------------------------- #
# Supabase writer
# --------------------------------------------------------------------------- #


class DetectionStore:
    """Upserts tracks and prunes stale ones."""

    def __init__(self) -> None:
        self.client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        try:
            self.client.table("atlas_detections").upsert(
                rows, on_conflict="flight_session_id,track_id"
            ).execute()
        except Exception as exc:  # never let a write error kill the loop
            log.warning("Upsert failed: %s", exc)

    def prune(self) -> None:
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(seconds=TRACK_TTL_SECONDS)).isoformat()
        try:
            # One tracker, one freshness window: every row comes from the fast
            # pass and refreshes at the same rate.
            self.client.table("atlas_detections").delete().eq(
                "flight_session_id", FLIGHT_SESSION_ID
            ).lt("updated_at", cutoff).execute()
        except Exception as exc:
            log.warning("Prune failed: %s", exc)


    def clear(self) -> None:
        try:
            self.client.table("atlas_detections").delete().eq(
                "flight_session_id", FLIGHT_SESSION_ID
            ).execute()
        except Exception as exc:
            log.warning("Clear failed: %s", exc)


def cleanup_loop(store: DetectionStore) -> None:
    while True:
        # Sweep at least twice per TTL so boxes vanish quickly after an object
        # leaves the frame, with a 0.4s floor to keep write volume sane.
        time.sleep(max(0.4, TRACK_TTL_SECONDS / 2))
        store.prune()



# --------------------------------------------------------------------------- #
# Detection loop
# --------------------------------------------------------------------------- #


def open_capture(url: str) -> cv2.VideoCapture | None:
    # Low-latency FFmpeg options: TCP transport, no buffering, no reordering.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|reorder_queue_size;0|max_delay;0"
    )
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    # Keep the buffer tiny so we always analyse near-live frames.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


class FrameGrabber:
    """Continuously drains the RTSP stream and keeps only the newest frame.

    Without this, cv2.VideoCapture.read() returns queued frames one by one, so
    inference slower than the stream's framerate makes the analysed frame drift
    further and further behind live video — boxes then lag and look inaccurate.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="grabber")
        self._thread.start()

    def _run(self) -> None:
        empty_reads = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                empty_reads += 1
                if empty_reads > 60:
                    with self._lock:
                        self._error = "stream returned no frames"
                    return
                time.sleep(0.05)
                continue
            empty_reads = 0
            with self._lock:
                self._frame = frame
                self._seq += 1

    def latest(self):
        with self._lock:
            return self._frame, self._seq, self._error

    def stop(self) -> None:
        self._stop.set()


def build_rows(detections, width: int, height: int) -> list[dict]:
    """Turn tracked detections into atlas_detections rows (normalised boxes)."""
    tracked_labels = detections.data.get("label") if detections.data else None
    timestamp = datetime.now(timezone.utc).isoformat()
    confidences = (
        detections.confidence
        if detections.confidence is not None
        else np.zeros(len(detections))
    )
    tracker_ids = (
        detections.tracker_id
        if detections.tracker_id is not None
        else np.full(len(detections), -1)
    )

    rows: list[dict] = []
    for idx, (xyxy, conf, track_id) in enumerate(
        zip(detections.xyxy, confidences, tracker_ids)
    ):
        if track_id is None or int(track_id) < 0:
            continue
        name = (
            str(tracked_labels[idx]).strip().lower()
            if tracked_labels is not None and idx < len(tracked_labels)
            else ""
        )
        if not name:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        # Normalise to 0–1 and clamp so partially off-screen boxes stay valid.
        nx = max(0.0, min(1.0, x1 / width))
        ny = max(0.0, min(1.0, y1 / height))
        nw = max(0.0, min(1.0 - nx, (x2 - x1) / width))
        nh = max(0.0, min(1.0 - ny, (y2 - y1) / height))
        rows.append(
            {
                "flight_session_id": FLIGHT_SESSION_ID,
                "track_id": int(track_id),
                "object_class": name,
                "confidence": round(float(conf), 4),
                "bbox": {
                    "x": round(nx, 5),
                    "y": round(ny, 5),
                    "width": round(nw, 5),
                    "height": round(nh, 5),
                },
                "updated_at": timestamp,
            }
        )
    return rows


def attach_labels(detections, labels: list[str]):
    detections.data = dict(detections.data or {})
    detections.data["label"] = np.array(labels, dtype=object)
    return detections


def tiled_worker(detector, rois: RoiRegistry, grabber, stop: threading.Event) -> None:
    """Slow, full-resolution tiled scout — runs beside the fast pass.

    It never writes to the database: every hit that is not already covered by a
    tracked box or an existing ROI becomes a new region for the fast pass to
    inspect up close, so a single tracker owns all output.
    """
    while not stop.is_set():
        started = time.time()
        frame, _, error = grabber.latest()
        if error or frame is None:
            stop.wait(0.2)
            continue
        height, width = frame.shape[:2]
        if not width or not height:
            stop.wait(0.2)
            continue
        try:
            detections, _labels = detector.detect_tiled(frame)
            boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
            added = rois.add_candidates(boxes, width, height)
            if added:
                log.info("Tiled scout nominated %d new ROI(s)", added)
        except Exception as exc:
            log.warning("Tiled pass failed: %s", exc)
        elapsed = time.time() - started
        stop.wait(max(0.05, TILED_FRAME_INTERVAL - elapsed))


def run_stream(detector, store: DetectionStore) -> None:
    """One connection attempt. Returns when the stream drops."""
    cap = open_capture(RTSP_URL)
    if cap is None:
        raise ConnectionError("could not open RTSP stream")

    with status.lock:
        status.connected = True
    log.info("Connected to stream")

    # Fast-reacting tracker: short memory for lost tracks and a frame rate that
    # matches the actual analysis rate, so positions are not extrapolated far.
    tracker = sv.ByteTrack(
        lost_track_buffer=TRACKER_LOST_BUFFER,
        frame_rate=max(1, int(round(ANALYSIS_FPS))),
        track_activation_threshold=min(0.25, DETECTION_BOX_THRESHOLD if DETECTION_ENGINE == "grounding_dino" else DETECTION_CONFIDENCE),
    )
    grabber = FrameGrabber(cap)
    last_seq = 0
    last_inference = 0.0

    tiled_stop = threading.Event()
    tiled_thread: threading.Thread | None = None
    roi_registry: RoiRegistry | None = None
    pool = getattr(detector, "_pool", None)
    if TILED_ENABLED and hasattr(detector, "detect_tiled"):
        roi_registry = RoiRegistry()
        tiled_thread = threading.Thread(
            target=tiled_worker,
            args=(detector, roi_registry, grabber, tiled_stop),
            daemon=True,
            name="tiled",
        )
        tiled_thread.start()

    try:
        while True:
            frame, seq, error = grabber.latest()
            if error:
                raise ConnectionError(error)
            if frame is None or seq == last_seq:
                # No new frame yet — wait briefly instead of re-analysing.
                time.sleep(0.02)
                continue

            now = time.time()
            # Pace inference to the engine's analysis rate, but always on the
            # newest available frame (never on a queued, stale one).
            if now - last_inference < MIN_FRAME_INTERVAL:
                time.sleep(min(0.02, MIN_FRAME_INTERVAL))
                continue
            last_seq = seq
            last_inference = now
            with status.lock:
                status.last_frame_at = now
            source = frame
            src_h, src_w = frame.shape[:2]
            if not src_w or not src_h:
                continue

            # Downscale before inference — boxes stay correct because they are
            # normalised against the frame we actually analysed.
            scale = 1.0
            if INFER_MAX_SIDE and max(src_w, src_h) > INFER_MAX_SIDE:
                scale = INFER_MAX_SIDE / float(max(src_w, src_h))
                frame = cv2.resize(
                    frame,
                    (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
                    interpolation=cv2.INTER_LINEAR,
                )
            height, width = frame.shape[:2]

            detections, labels = detector.detect(frame)
            log.info("RAW DETECT: %d boxes labels=%s", len(detections), labels)

            if roi_registry is not None:
                active = roi_registry.snapshot()
                boxes = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
                scores = (
                    np.asarray(detections.confidence, dtype=np.float32).reshape(-1)
                    if detections.confidence is not None
                    else np.zeros(len(boxes), dtype=np.float32)
                )
                labels = list(labels)
                if active:
                    # Full-resolution crops of the regions the scout flagged.
                    rb, rs, rl, outcome = detect_in_rois(
                        detector, source, active, pool
                    )
                    remaining = roi_registry.report(outcome)
                    with status.lock:
                        status.active_rois = remaining
                    if len(rb):
                        # ROI boxes are full-frame; bring them into the same
                        # (downscaled) coordinate space as the main pass.
                        rb = rb * scale
                        boxes = np.concatenate([boxes, rb], axis=0)
                        scores = np.concatenate([scores, rs], axis=0)
                        labels.extend(rl)

                if len(boxes):
                    keep = _class_aware_nms(boxes, scores, labels, TILE_IOU_THRESHOLD)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = [labels[i] for i in keep]
                    detections = sv.Detections(
                        xyxy=boxes.astype(np.float32),
                        confidence=scores.astype(np.float32),
                        class_id=np.zeros(len(boxes), dtype=int),
                    )
                else:
                    detections = sv.Detections.empty()

            # Keep the label list index-aligned with the detections we track.
            detections = tracker.update_with_detections(attach_labels(detections, labels))
            rows = build_rows(detections, width, height)
            log.info("AFTER TRACKER: %d tracked, data_keys=%s, labels=%s, rows_built=%d", len(detections), list(detections.data.keys()) if detections.data else None, detections.data.get("label") if detections.data else None, len(rows))

            if roi_registry is not None:
                tracked = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
                roi_registry.set_last_boxes(
                    tracked / scale if scale != 1.0 else tracked
                )

            with status.lock:
                status.active_tracks = len(rows)

            store.upsert(rows)
    finally:
        tiled_stop.set()
        if tiled_thread is not None:
            tiled_thread.join(timeout=5)
        grabber.stop()
        cap.release()

        with status.lock:
            status.connected = False
            status.active_tracks = 0
            status.active_rois = 0




def main() -> None:
    missing = [
        name
        for name, value in (
            ("MEDIAMTX_RTSP_URL", RTSP_URL),
            ("SUPABASE_URL", SUPABASE_URL),
            ("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY),
            ("FLIGHT_SESSION_ID", FLIGHT_SESSION_ID),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    start_health_server()

    detector = build_detector()
    log.info("ACTIVE DETECTION ENGINE: %s", detector.name)
    log.info("Detector config: %s", detector.describe())

    store = DetectionStore()
    store.clear()
    threading.Thread(target=cleanup_loop, args=(store,), daemon=True, name="cleanup").start()

    backoff = BACKOFF_MIN
    while True:
        try:
            run_stream(detector, store)
            log.warning("Stream ended")
        except Exception as exc:
            log.warning("Stream error: %s", exc)

        # Drop stale boxes immediately so the UI does not show frozen overlays.
        store.clear()
        with status.lock:
            status.reconnects += 1

        log.info("Reconnecting in %.1fs", backoff)
        time.sleep(backoff)
        backoff = min(BACKOFF_MAX, backoff * 2)
        # A successful connection resets the backoff inside run_stream's first
        # frame; approximate that by resetting whenever we got frames recently.
        with status.lock:
            recent = status.last_frame_at and (time.time() - status.last_frame_at) < 60
        if recent:
            backoff = BACKOFF_MIN


if __name__ == "__main__":
    main()

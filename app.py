"""atlas-detector — continuous object detection and tracking on Atlas drone video.

Discovers every active flight that has a live Atlas video stream, reads each
stream over RTSP from MediaMTX, runs YOLOv8n + ByteTrack on a subset of the
frames, and upserts one row per tracked object into the Supabase table
`atlas_detections` (unique on flight_session_id + track_id). A cleanup loop
deletes tracks that stopped being updated.

No serial number is configured anywhere: the supervisor polls Supabase and
starts/stops a worker per live stream, so any drone works out of the box.

CPU only. Everything is configured through environment variables:

  RTSP_BASE_URL                 rtsp://live-video.internal:8554
  DETECTOR_SHARED_SECRET        appended as ?detector=<secret> when reading
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  MEDIAMTX_RTSP_URL             optional, pins one stream (manual testing)
  FLIGHT_SESSION_ID             optional, required together with the URL above
  MAX_STREAMS                   simultaneous workers, default 2
  DISCOVERY_INTERVAL_SECONDS    how often we look for new streams, default 10
  SENSOR_STALE_SECONDS          a sensor counts as live for this long, default 300
  DETECTION_FPS                 analysed frames per second, default 10
  DETECTION_CLASSES             default: person,bicycle,car,motorcycle,airplane,
                                bus,train,truck,boat,bird,dog,horse,sheep,cow,
                                kite,surfboard (any COCO class works)
  DETECTION_CONFIDENCE          default 0.20 (the UI filters further)
  INFER_MAX_SIDE                downscale longest side before inference, default 640
  TRACKER_LOST_BUFFER           analysed frames a lost track survives, default 5
  TRACK_TTL_SECONDS             default 0.8
  RANGE_PASS_ENABLED            tiled full-res pass for small/distant objects
  RANGE_PASS_INTERVAL_SECONDS   cadence of the range pass, default 2.0
  RANGE_TILE_COLS / _ROWS       tile grid, default 3x2 with 15% overlap
  RANGE_TILE_OVERLAP
  RANGE_CONFIDENCE              separate threshold for the range pass, 0.15
  RANGE_DEDUPE_IOU              overlap at which duplicate boxes are merged, 0.5
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
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

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

RTSP_BASE_URL = (
    os.environ.get("RTSP_BASE_URL", "rtsp://live-video.internal:8554").strip().rstrip("/")
)
DETECTOR_SHARED_SECRET = os.environ.get("DETECTOR_SHARED_SECRET", "").strip()

# Optional manual override — pins the detector to a single stream for testing.
RTSP_URL = os.environ.get("MEDIAMTX_RTSP_URL", "").strip()
FLIGHT_SESSION_ID = os.environ.get("FLIGHT_SESSION_ID", "").strip()

MAX_STREAMS = int(os.environ.get("MAX_STREAMS", "2") or 2)
DISCOVERY_INTERVAL_SECONDS = float(
    os.environ.get("DISCOVERY_INTERVAL_SECONDS", "10") or 10
)
SENSOR_STALE_SECONDS = float(os.environ.get("SENSOR_STALE_SECONDS", "300") or 300)

DETECTION_FPS = float(os.environ.get("DETECTION_FPS", "10") or 10)
# Written raw and low: the video dialog has a sensitivity slider that filters
# client-side, so raising sensitivity never needs a redeploy.
DETECTION_CONFIDENCE = float(os.environ.get("DETECTION_CONFIDENCE", "0.20") or 0.20)
TRACK_TTL_SECONDS = float(os.environ.get("TRACK_TTL_SECONDS", "0.8") or 0.8)
# Downscale before inference: the single biggest latency win on CPU. 0 = off.
INFER_MAX_SIDE = int(os.environ.get("INFER_MAX_SIDE", "640") or 640)
# How many analysed frames a lost track survives inside the tracker.
TRACKER_LOST_BUFFER = int(os.environ.get("TRACKER_LOST_BUFFER", "5") or 5)
DEFAULT_CLASSES = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,"
    "bird,dog,horse,sheep,cow,kite,surfboard"
)
DETECTION_CLASSES = [
    c.strip().lower()
    for c in os.environ.get("DETECTION_CLASSES", DEFAULT_CLASSES).split(",")
    if c.strip()
]

# Long-range pass: the full-resolution frame is tiled and analysed at a low
# cadence to catch small, distant objects that vanish in the fast downscale.
# Range detections are merged into the SAME tracker, so one object = one box.
RANGE_PASS_ENABLED = os.environ.get("RANGE_PASS_ENABLED", "true").lower() != "false"
RANGE_PASS_INTERVAL_SECONDS = float(
    os.environ.get("RANGE_PASS_INTERVAL_SECONDS", "2.0") or 2.0
)
RANGE_TILE_COLS = int(os.environ.get("RANGE_TILE_COLS", "3") or 3)
RANGE_TILE_ROWS = int(os.environ.get("RANGE_TILE_ROWS", "2") or 2)
RANGE_TILE_OVERLAP = float(os.environ.get("RANGE_TILE_OVERLAP", "0.15") or 0.15)
# Separate confidence for the range pass — small far objects score lower.
RANGE_CONFIDENCE = float(os.environ.get("RANGE_CONFIDENCE", "0.15") or 0.15)
# IoU at which a range box that overlaps a fast box of the same class is dropped.
RANGE_DEDUPE_IOU = float(os.environ.get("RANGE_DEDUPE_IOU", "0.5") or 0.5)
# Containment suppression: a tile can only see part of an object (a torso, a
# head), so its box sits INSIDE the full-frame box. Two such boxes have low IoU
# by definition, which is why IoU alone let duplicates through. This compares
# the overlap against the SMALLER box instead: 0.7 means "70% of the smaller box
# lies inside the larger one -> same object". Two genuinely separate people
# standing side by side barely overlap, so they are never merged.
RANGE_CONTAINMENT = float(os.environ.get("RANGE_CONTAINMENT", "0.7") or 0.7)
# Range results are re-fed to the tracker between passes, but only while fresh:
# a stale box keeps its old position while the object moves on, which spawns a
# ghost track next to the real one.
RANGE_RESULT_MAX_AGE_SECONDS = float(
    os.environ.get("RANGE_RESULT_MAX_AGE_SECONDS", "0") or 0
) or (RANGE_PASS_INTERVAL_SECONDS + 0.5)
# Temporary diagnostics: log per-source counts and every suppressed duplicate.
RANGE_DEBUG = os.environ.get("RANGE_DEBUG", "false").lower() == "true"

MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")

# Backoff bounds for reconnecting to MediaMTX.
BACKOFF_MIN = 1.0
BACKOFF_MAX = 30.0

MIN_FRAME_INTERVAL = 1.0 / DETECTION_FPS if DETECTION_FPS > 0 else 0.2

# Force TCP for RTSP — UDP is unreliable across the Fly private network.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


# --------------------------------------------------------------------------- #
# Shared status (for /health)
# --------------------------------------------------------------------------- #


class Status:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.streams: dict[str, dict] = {}
        self.started_at = time.time()

    def update(self, session_id: str, **fields) -> None:
        with self.lock:
            entry = self.streams.setdefault(session_id, {})
            entry.update(fields)

    def remove(self, session_id: str) -> None:
        with self.lock:
            self.streams.pop(session_id, None)

    def snapshot(self) -> dict:
        with self.lock:
            streams = []
            for session_id, entry in self.streams.items():
                last = entry.get("last_frame_at")
                streams.append(
                    {
                        "flight_session_id": session_id,
                        "path": entry.get("path"),
                        "connected": entry.get("connected", False),
                        "active_tracks": entry.get("active_tracks", 0),
                        "reconnects": entry.get("reconnects", 0),
                        "last_frame_age_seconds": (
                            None if last is None else round(time.time() - last, 2)
                        ),
                    }
                )
            return {
                "ok": True,
                "service": "atlas-detector",
                "engine": "yolo",
                "model": MODEL_PATH,
                "detection_fps": DETECTION_FPS,
                "confidence": DETECTION_CONFIDENCE,
                "classes": DETECTION_CLASSES,
                "max_streams": MAX_STREAMS,
                "active_streams": len(streams),
                "streams": streams,
                "uptime_seconds": round(time.time() - self.started_at, 1),
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
# Detector
# --------------------------------------------------------------------------- #


class YoloDetector:
    """YOLOv8n on a fixed COCO class list. Shared by every stream worker."""

    name = "yolo"

    def __init__(self) -> None:
        from ultralytics import YOLO

        log.info("Loading YOLOv8n (%s)", MODEL_PATH)
        self.model = YOLO(MODEL_PATH)
        self._lock = threading.Lock()

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
            f"confidence={DETECTION_CONFIDENCE} fps={DETECTION_FPS}"
        )

    def detect(
        self, frame, conf: float | None = None
    ) -> tuple[sv.Detections, list[str]]:
        # One model instance shared by all workers: serialise inference so two
        # streams cannot corrupt each other's state.
        with self._lock:
            result = self.model.predict(
                frame,
                conf=conf if conf is not None else DETECTION_CONFIDENCE,
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


# --------------------------------------------------------------------------- #
# Supabase
# --------------------------------------------------------------------------- #


class DetectionStore:
    """Upserts tracks and prunes stale ones."""

    def __init__(self) -> None:
        self.client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    # -- detections --------------------------------------------------------- #

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        try:
            self.client.table("atlas_detections").upsert(
                rows, on_conflict="flight_session_id,track_id"
            ).execute()
        except Exception as exc:  # never let a write error kill the loop
            log.warning("Upsert failed: %s", exc)

    def prune(self, session_ids: list[str]) -> None:
        if not session_ids:
            return
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=TRACK_TTL_SECONDS)
        ).isoformat()
        try:
            self.client.table("atlas_detections").delete().in_(
                "flight_session_id", session_ids
            ).lt("updated_at", cutoff).execute()
        except Exception as exc:
            log.warning("Prune failed: %s", exc)

    def clear(self, session_id: str) -> None:
        try:
            self.client.table("atlas_detections").delete().eq(
                "flight_session_id", session_id
            ).execute()
        except Exception as exc:
            log.warning("Clear failed: %s", exc)

    # -- discovery ---------------------------------------------------------- #

    def live_streams(self) -> list[dict]:
        """Active flights that currently have an Atlas video stream.

        A flight qualifies when its drone has registered at least one sensor
        through /atlas-video-endpoint recently. EO (sensor 1) wins when the
        drone publishes both, since that is what the UI opens by default.
        """
        try:
            flights = (
                self.client.table("active_flights")
                .select("id, drone_id")
                .not_.is_("drone_id", "null")
                .execute()
                .data
                or []
            )
        except Exception as exc:
            log.warning("Discovery failed (active_flights): %s", exc)
            return []

        drone_ids = sorted({f["drone_id"] for f in flights if f.get("drone_id")})
        if not drone_ids:
            return []

        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=SENSOR_STALE_SECONDS)
        ).isoformat()
        try:
            sensors = (
                self.client.table("atlas_drone_sensors")
                .select("drone_id, serial, sensor, last_seen_at")
                .in_("drone_id", drone_ids)
                .gte("last_seen_at", cutoff)
                .execute()
                .data
                or []
            )
        except Exception as exc:
            log.warning("Discovery failed (atlas_drone_sensors): %s", exc)
            return []

        by_drone: dict[str, dict] = {}
        for row in sensors:
            current = by_drone.get(row["drone_id"])
            # Prefer EO (1), otherwise the lowest sensor number available.
            if current is None or int(row["sensor"]) < int(current["sensor"]):
                by_drone[row["drone_id"]] = row

        streams = []
        for flight in flights:
            sensor = by_drone.get(flight.get("drone_id"))
            if not sensor:
                continue
            streams.append(
                {
                    "flight_session_id": flight["id"],
                    "path": f"{sensor['serial']}/{int(sensor['sensor'])}",
                }
            )
        return streams


# --------------------------------------------------------------------------- #
# Video capture
# --------------------------------------------------------------------------- #


def rtsp_url_for(path: str) -> str:
    url = f"{RTSP_BASE_URL}/{path}"
    if DETECTOR_SHARED_SECRET:
        url = f"{url}?detector={DETECTOR_SHARED_SECRET}"
    return url


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


# --------------------------------------------------------------------------- #
# Detection rows
# --------------------------------------------------------------------------- #


def attach_labels(detections, labels: list[str]):
    detections.data = dict(detections.data or {})
    detections.data["label"] = np.array(labels, dtype=object)
    return detections


def build_rows(detections, session_id: str, width: int, height: int) -> list[dict]:
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
                "flight_session_id": session_id,
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


# --------------------------------------------------------------------------- #
# Long-range pass (tiled full-resolution analysis)
# --------------------------------------------------------------------------- #


def tile_offsets(width: int, height: int, cols: int, rows: int, overlap: float):
    """Yield (x0, y0, x1, y1) tile windows covering the WHOLE frame with overlap.

    The tiles are sized so that `cols` overlapping windows span the full width
    (and `rows` the full height) — otherwise the right/bottom edge of the frame
    is never analysed by the range pass.
    """
    overlap = min(max(overlap, 0.0), 0.5)
    tile_w = max(1, int(round(width / (cols - (cols - 1) * overlap)))) if cols > 1 else width
    tile_h = max(1, int(round(height / (rows - (rows - 1) * overlap)))) if rows > 1 else height
    step_x = max(1, int(round(tile_w * (1.0 - overlap))))
    step_y = max(1, int(round(tile_h * (1.0 - overlap))))
    for r in range(rows):
        for c in range(cols):
            x0 = min(c * step_x, max(0, width - tile_w))
            y0 = min(r * step_y, max(0, height - tile_h))
            yield x0, y0, min(width, x0 + tile_w), min(height, y0 + tile_h)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy box arrays a (n) and b (m) -> (n, m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 0, None)
    area_b = np.clip((bx2 - bx1) * (by2 - by1), 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def containment_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection over the SMALLER box area -> (n, m).

    Catches the tile artefact IoU misses: a partial box (torso) fully inside a
    full-body box scores ~1.0 here but only ~0.3 on IoU. Two distinct objects
    next to each other still score near 0, so they are never merged.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 0, None)
    area_b = np.clip((bx2 - bx1) * (by2 - by1), 0, None)
    smaller = np.minimum(area_a, area_b)
    return np.where(smaller > 0, inter / np.maximum(smaller, 1e-9), 0.0)


def is_duplicate(box_a: np.ndarray, box_b: np.ndarray) -> bool:
    """True when two same-class boxes describe the same physical object."""
    a = box_a.reshape(1, 4)
    b = box_b.reshape(1, 4)
    if iou_matrix(a, b)[0, 0] > RANGE_DEDUPE_IOU:
        return True
    return containment_matrix(a, b)[0, 0] > RANGE_CONTAINMENT


def dedupe_class_aware(detections, labels: list[str], iou_thr: float | None = None):
    """Keep the highest-confidence box per physical object, per class.

    Suppression is IoU *and* containment based — see `is_duplicate`. Applied to
    the merged fast+range set right before the tracker, so one object can only
    ever hand the tracker one box, no matter which pass found it.
    """
    if len(detections) <= 1:
        return detections, labels
    confidence = (
        detections.confidence
        if detections.confidence is not None
        else np.zeros(len(detections))
    )
    order = np.argsort(-confidence)
    keep: list[int] = []
    boxes = detections.xyxy
    for idx in order:
        duplicate = False
        for kept in keep:
            if labels[kept] != labels[idx]:
                continue
            if is_duplicate(boxes[idx], boxes[kept]):
                duplicate = True
                if RANGE_DEBUG:
                    log.info(
                        "dedupe: dropped %s %.2f (iou %.2f / containment %.2f vs %s %.2f)",
                        labels[idx],
                        confidence[idx],
                        iou_matrix(boxes[idx].reshape(1, 4), boxes[kept].reshape(1, 4))[0, 0],
                        containment_matrix(boxes[idx].reshape(1, 4), boxes[kept].reshape(1, 4))[0, 0],
                        labels[kept],
                        confidence[kept],
                    )
                break
        if not duplicate:
            keep.append(int(idx))
    keep.sort()
    if len(keep) == len(detections):
        return detections, labels
    return detections[keep], [labels[i] for i in keep]


class RangeScanner:
    """Background thread analysing the full-resolution frame in tiles.

    Small, distant objects disappear when the fast pass downscales the frame.
    This scanner tiles the full frame, runs YOLO per tile at a low cadence and
    hands the boxes (full-resolution coordinates) to the stream worker, which
    merges them into the shared tracker — one object still gets one box.
    """

    def __init__(self, path: str, detector, frame_source) -> None:
        self._path = path
        self._detector = detector
        self._frame_source = frame_source  # callable -> full-res frame or None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._detections: sv.Detections | None = None
        self._labels: list[str] = []
        self._updated_at: float = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"range-{path}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[sv.Detections | None, list[str], float]:
        with self._lock:
            return self._detections, list(self._labels), self._updated_at

    def _run(self) -> None:
        log.info(
            "[%s] range pass: %dx%d tiles every %.1fs",
            self._path,
            RANGE_TILE_COLS,
            RANGE_TILE_ROWS,
            RANGE_PASS_INTERVAL_SECONDS,
        )
        while not self._stop.is_set():
            frame = self._frame_source()
            if frame is None:
                if self._stop.wait(0.2):
                    break
                continue
            started = time.time()
            try:
                self._scan(frame)
            except Exception as exc:
                log.warning("[%s] range pass error: %s", self._path, exc)
            elapsed = time.time() - started
            remaining = RANGE_PASS_INTERVAL_SECONDS - elapsed
            if self._stop.wait(max(0.1, remaining)):
                break
        log.info("[%s] range pass stopped", self._path)

    def _scan(self, frame) -> None:
        height, width = frame.shape[:2]
        all_xyxy: list[np.ndarray] = []
        all_conf: list[float] = []
        all_labels: list[str] = []
        for x0, y0, x1, y1 in tile_offsets(
            width, height, RANGE_TILE_COLS, RANGE_TILE_ROWS, RANGE_TILE_OVERLAP
        ):
            tile = frame[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            detections, labels = self._detector.detect(tile, conf=RANGE_CONFIDENCE)
            if len(detections) == 0:
                continue
            boxes = detections.xyxy.copy()
            confidence = (
                detections.confidence
                if detections.confidence is not None
                else np.zeros(len(detections))
            )
            # A tile can cut an object in half — the resulting torso/head box is
            # a fragment, not an object. Drop boxes that touch a tile edge which
            # is not also a frame edge; the overlapping neighbour tile sees the
            # whole object anyway.
            margin = 2.0
            keep_tile: list[int] = []
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes[i]
                truncated = (
                    (bx1 <= margin and x0 > 0)
                    or (by1 <= margin and y0 > 0)
                    or (bx2 >= (x1 - x0) - margin and x1 < width)
                    or (by2 >= (y1 - y0) - margin and y1 < height)
                )
                if truncated:
                    if RANGE_DEBUG:
                        log.info(
                            "[%s] range: dropped truncated %s %.2f at tile %d,%d",
                            self._path,
                            labels[i],
                            confidence[i],
                            x0,
                            y0,
                        )
                    continue
                keep_tile.append(i)
            if not keep_tile:
                continue
            boxes = boxes[keep_tile]
            boxes[:, 0] += x0
            boxes[:, 2] += x0
            boxes[:, 1] += y0
            boxes[:, 3] += y0
            all_xyxy.append(boxes)
            all_conf.append(confidence[keep_tile])
            all_labels.extend([labels[i] for i in keep_tile])
        if all_xyxy:
            merged = sv.Detections(
                xyxy=np.concatenate(all_xyxy),
                confidence=np.concatenate(all_conf),
            )
            merged, all_labels = dedupe_class_aware(merged, all_labels)
        else:
            merged = sv.Detections.empty()
            all_labels = []
        with self._lock:
            self._detections = merged
            self._labels = all_labels
            self._updated_at = time.time()
        log.info(
            "[%s] range pass: %d object(s) in full frame", self._path, len(merged)
        )


# --------------------------------------------------------------------------- #
# One worker per live stream
# --------------------------------------------------------------------------- #


class StreamWorker(threading.Thread):
    """Analyses a single RTSP stream until it is asked to stop."""

    def __init__(self, session_id: str, path: str, detector, store: DetectionStore) -> None:
        super().__init__(daemon=True, name=f"stream-{path}")
        self.session_id = session_id
        self.path = path
        self.detector = detector
        self.store = store
        self.url = RTSP_URL if RTSP_URL else rtsp_url_for(path)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    # -- main loop ---------------------------------------------------------- #

    def run(self) -> None:
        log.info("[%s] worker started (%s)", self.path, self.session_id)
        status.update(self.session_id, path=self.path, connected=False, reconnects=0)
        backoff = BACKOFF_MIN
        reconnects = 0
        while not self._stop.is_set():
            try:
                self._run_once()
                log.warning("[%s] stream ended", self.path)
            except Exception as exc:
                log.warning("[%s] stream error: %s", self.path, exc)

            # Drop stale boxes immediately so the UI never shows frozen overlays.
            self.store.clear(self.session_id)
            reconnects += 1
            status.update(
                self.session_id, connected=False, active_tracks=0, reconnects=reconnects
            )
            if self._stop.wait(backoff):
                break
            backoff = min(BACKOFF_MAX, backoff * 2)

        self.store.clear(self.session_id)
        status.remove(self.session_id)
        log.info("[%s] worker stopped", self.path)

    def _run_once(self) -> None:
        cap = open_capture(self.url)
        if cap is None:
            raise ConnectionError("could not open RTSP stream")

        status.update(self.session_id, connected=True)
        log.info("[%s] connected", self.path)

        # Fast-reacting tracker: short memory for lost tracks and a frame rate
        # that matches the actual analysis rate.
        tracker = sv.ByteTrack(
            lost_track_buffer=TRACKER_LOST_BUFFER,
            frame_rate=max(1, int(round(DETECTION_FPS))),
            track_activation_threshold=min(0.25, DETECTION_CONFIDENCE),
        )
        grabber = FrameGrabber(cap)
        last_seq = 0
        last_inference = 0.0

        # Latest full-resolution frame, shared with the long-range scanner.
        full_frame_slot: dict = {"frame": None, "seq": -1}

        def latest_full_frame():
            return full_frame_slot["frame"]

        scanner = (
            RangeScanner(self.path, self.detector, latest_full_frame)
            if RANGE_PASS_ENABLED
            else None
        )

        try:
            while not self._stop.is_set():
                frame, seq, error = grabber.latest()
                if error:
                    raise ConnectionError(error)
                if frame is None or seq == last_seq:
                    time.sleep(0.02)
                    continue

                now = time.time()
                if now - last_inference < MIN_FRAME_INTERVAL:
                    time.sleep(min(0.02, MIN_FRAME_INTERVAL))
                    continue
                last_seq = seq
                last_inference = now
                status.update(self.session_id, last_frame_at=now)

                src_h, src_w = frame.shape[:2]
                if not src_w or not src_h:
                    continue

                # Share the full-resolution frame with the range scanner before
                # downscaling — that is where the small/distant objects live.
                full_frame_slot["frame"] = frame
                full_frame_slot["seq"] = seq

                # Downscale before inference — boxes stay correct because they
                # are normalised against the frame we actually analysed.
                scale = 1.0
                if INFER_MAX_SIDE and max(src_w, src_h) > INFER_MAX_SIDE:
                    scale = INFER_MAX_SIDE / float(max(src_w, src_h))
                    frame = cv2.resize(
                        frame,
                        (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
                        interpolation=cv2.INTER_LINEAR,
                    )
                height, width = frame.shape[:2]

                started = time.time()
                detections, labels = self.detector.detect(frame)
                infer_ms = (time.time() - started) * 1000.0
                raw_count = len(detections)

                # Merge long-range detections (full-res coords -> fast-frame
                # coords) and suppress duplicates ONCE, over the combined set,
                # so the tracker only ever sees one box per physical object.
                # Stale range results are skipped: their coordinates describe
                # where the object was, and feeding them spawns a ghost track.
                range_count = 0
                if scanner is not None:
                    range_dets, range_labels, range_at = scanner.latest()
                    fresh = (now - range_at) <= RANGE_RESULT_MAX_AGE_SECONDS
                    if range_dets is not None and len(range_dets) > 0 and fresh:
                        range_count = len(range_dets)
                        detections = sv.Detections(
                            xyxy=np.concatenate(
                                [detections.xyxy, range_dets.xyxy * scale]
                            ),
                            confidence=np.concatenate(
                                [
                                    detections.confidence
                                    if detections.confidence is not None
                                    else np.zeros(len(detections)),
                                    range_dets.confidence
                                    if range_dets.confidence is not None
                                    else np.zeros(len(range_dets)),
                                ]
                            ),
                        )
                        labels = labels + list(range_labels)

                merged_count = len(detections)
                detections, labels = dedupe_class_aware(detections, labels)
                if RANGE_DEBUG:
                    log.info(
                        "[%s] sources: fast=%d range=%d merged=%d after-dedupe=%d",
                        self.path,
                        raw_count,
                        range_count,
                        merged_count,
                        len(detections),
                    )

                detections = tracker.update_with_detections(
                    attach_labels(detections, labels)
                )
                rows = build_rows(detections, self.session_id, width, height)

                log.info(
                    "[%s] %d raw -> %d tracked -> %d row(s) in %.0f ms",
                    self.path,
                    raw_count,
                    len(detections),
                    len(rows),
                    infer_ms,
                )

                status.update(self.session_id, active_tracks=len(rows))
                self.store.upsert(rows)
        finally:
            if scanner is not None:
                scanner.stop()
            grabber.stop()
            cap.release()
            status.update(self.session_id, connected=False, active_tracks=0)


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #


def cleanup_loop(store: DetectionStore, workers: dict[str, StreamWorker]) -> None:
    while True:
        # Sweep at least twice per TTL so boxes vanish quickly after an object
        # leaves the frame, with a 0.4s floor to keep write volume sane.
        time.sleep(max(0.4, TRACK_TTL_SECONDS / 2))
        store.prune(list(workers.keys()))


def supervise(detector, store: DetectionStore) -> None:
    """Start a worker per live stream, stop workers whose flight ended."""
    workers: dict[str, StreamWorker] = {}
    threading.Thread(
        target=cleanup_loop, args=(store, workers), daemon=True, name="cleanup"
    ).start()

    if RTSP_URL and FLIGHT_SESSION_ID:
        log.info("Pinned to %s (MEDIAMTX_RTSP_URL override)", FLIGHT_SESSION_ID)
        worker = StreamWorker(FLIGHT_SESSION_ID, "manual", detector, store)
        workers[FLIGHT_SESSION_ID] = worker
        worker.start()
        while True:
            time.sleep(DISCOVERY_INTERVAL_SECONDS)

    while True:
        streams = store.live_streams()
        wanted = {s["flight_session_id"]: s["path"] for s in streams}

        for session_id, worker in list(workers.items()):
            if session_id not in wanted or not worker.is_alive():
                log.info("Stopping worker for %s", session_id)
                worker.stop()
                workers.pop(session_id, None)

        for session_id, path in wanted.items():
            if session_id in workers:
                continue
            if len(workers) >= MAX_STREAMS:
                log.warning(
                    "MAX_STREAMS=%d reached — not analysing %s", MAX_STREAMS, path
                )
                break
            worker = StreamWorker(session_id, path, detector, store)
            workers[session_id] = worker
            worker.start()

        time.sleep(DISCOVERY_INTERVAL_SECONDS)


def main() -> None:
    missing = [
        name
        for name, value in (
            ("SUPABASE_URL", SUPABASE_URL),
            ("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")
    if RTSP_URL and not FLIGHT_SESSION_ID:
        raise SystemExit("MEDIAMTX_RTSP_URL requires FLIGHT_SESSION_ID")

    start_health_server()

    detector = YoloDetector()
    log.info("Detector config: %s", detector.describe())
    log.info(
        "Auto-discovery every %.0fs from %s (max %d stream(s))",
        DISCOVERY_INTERVAL_SECONDS,
        RTSP_BASE_URL,
        MAX_STREAMS,
    )

# --------------------------------------------------------------------------- #
# Shared status (for /health)
# --------------------------------------------------------------------------- #


class Status:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.streams: dict[str, dict] = {}
        self.started_at = time.time()

    def update(self, session_id: str, **fields) -> None:
        with self.lock:
            entry = self.streams.setdefault(session_id, {})
            entry.update(fields)

    def remove(self, session_id: str) -> None:
        with self.lock:
            self.streams.pop(session_id, None)

    def snapshot(self) -> dict:
        with self.lock:
            streams = []
            for session_id, entry in self.streams.items():
                last = entry.get("last_frame_at")
                streams.append(
                    {
                        "flight_session_id": session_id,
                        "path": entry.get("path"),
                        "connected": entry.get("connected", False),
                        "active_tracks": entry.get("active_tracks", 0),
                        "reconnects": entry.get("reconnects", 0),
                        "last_frame_age_seconds": (
                            None if last is None else round(time.time() - last, 2)
                        ),
                    }
                )
            return {
                "ok": True,
                "service": "atlas-detector",
                "engine": "yolo",
                "model": MODEL_PATH,
                "detection_fps": DETECTION_FPS,
                "confidence": DETECTION_CONFIDENCE,
                "classes": DETECTION_CLASSES,
                "max_streams": MAX_STREAMS,
                "active_streams": len(streams),
                "streams": streams,
                "uptime_seconds": round(time.time() - self.started_at, 1),
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
# Detector
# --------------------------------------------------------------------------- #


class YoloDetector:
    """YOLO26n (NMS-free, end-to-end) on a fixed COCO class list. Shared by every stream worker."""

    name = "yolo"

    def __init__(self) -> None:
        from ultralytics import YOLO

        log.info("Loading YOLO26 (%s)", MODEL_PATH)
        self.model = YOLO(MODEL_PATH)
        self._lock = threading.Lock()

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
            f"confidence={DETECTION_CONFIDENCE} fps={DETECTION_FPS}"
        )

    def detect(
        self, frame, conf: float | None = None
    ) -> tuple[sv.Detections, list[str]]:
        # One model instance shared by all workers: serialise inference so two
        # streams cannot corrupt each other's state.
        with self._lock:
            result = self.model.predict(
                frame,
                conf=conf if conf is not None else DETECTION_CONFIDENCE,
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


# --------------------------------------------------------------------------- #
# Supabase
# --------------------------------------------------------------------------- #


class DetectionStore:
    """Upserts tracks and prunes stale ones."""

    def __init__(self) -> None:
        self.client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

    # -- detections --------------------------------------------------------- #

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        try:
            self.client.table("atlas_detections").upsert(
                rows, on_conflict="flight_session_id,track_id"
            ).execute()
        except Exception as exc:  # never let a write error kill the loop
            log.warning("Upsert failed: %s", exc)

    def prune(self, session_ids: list[str]) -> None:
        if not session_ids:
            return
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=TRACK_TTL_SECONDS)
        ).isoformat()
        try:
            self.client.table("atlas_detections").delete().in_(
                "flight_session_id", session_ids
            ).lt("updated_at", cutoff).execute()
        except Exception as exc:
            log.warning("Prune failed: %s", exc)

    def clear(self, session_id: str) -> None:
        try:
            self.client.table("atlas_detections").delete().eq(
                "flight_session_id", session_id
            ).execute()
        except Exception as exc:
            log.warning("Clear failed: %s", exc)

    # -- discovery ---------------------------------------------------------- #

    def live_streams(self) -> list[dict]:
        """Active flights that currently have an Atlas video stream.

        A flight qualifies when its drone has registered at least one sensor
        through /atlas-video-endpoint recently. EO (sensor 1) wins when the
        drone publishes both, since that is what the UI opens by default.
        """
        try:
            flights = (
                self.client.table("active_flights")
                .select("id, drone_id")
                .not_.is_("drone_id", "null")
                .execute()
                .data
                or []
            )
        except Exception as exc:
            log.warning("Discovery failed (active_flights): %s", exc)
            return []

        drone_ids = sorted({f["drone_id"] for f in flights if f.get("drone_id")})
        if not drone_ids:
            return []

        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=SENSOR_STALE_SECONDS)
        ).isoformat()
        try:
            sensors = (
                self.client.table("atlas_drone_sensors")
                .select("drone_id, serial, sensor, last_seen_at")
                .in_("drone_id", drone_ids)
                .gte("last_seen_at", cutoff)
                .execute()
                .data
                or []
            )
        except Exception as exc:
            log.warning("Discovery failed (atlas_drone_sensors): %s", exc)
            return []

        by_drone: dict[str, dict] = {}
        for row in sensors:
            current = by_drone.get(row["drone_id"])
            # Prefer EO (1), otherwise the lowest sensor number available.
            if current is None or int(row["sensor"]) < int(current["sensor"]):
                by_drone[row["drone_id"]] = row

        streams = []
        for flight in flights:
            sensor = by_drone.get(flight.get("drone_id"))
            if not sensor:
                continue
            streams.append(
                {
                    "flight_session_id": flight["id"],
                    "path": f"{sensor['serial']}/{int(sensor['sensor'])}",
                }
            )
        return streams


# --------------------------------------------------------------------------- #
# Video capture
# --------------------------------------------------------------------------- #


def rtsp_url_for(path: str) -> str:
    url = f"{RTSP_BASE_URL}/{path}"
    if DETECTOR_SHARED_SECRET:
        url = f"{url}?detector={DETECTOR_SHARED_SECRET}"
    return url


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


# --------------------------------------------------------------------------- #
# Detection rows
# --------------------------------------------------------------------------- #


def attach_labels(detections, labels: list[str]):
    detections.data = dict(detections.data or {})
    detections.data["label"] = np.array(labels, dtype=object)
    return detections


def build_rows(detections, session_id: str, width: int, height: int) -> list[dict]:
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
                "flight_session_id": session_id,
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


# --------------------------------------------------------------------------- #
# Long-range pass (tiled full-resolution analysis)
# --------------------------------------------------------------------------- #


def tile_offsets(width: int, height: int, cols: int, rows: int, overlap: float):
    """Yield (x0, y0, x1, y1) tile windows covering the WHOLE frame with overlap.

    The tiles are sized so that `cols` overlapping windows span the full width
    (and `rows` the full height) — otherwise the right/bottom edge of the frame
    is never analysed by the range pass.
    """
    overlap = min(max(overlap, 0.0), 0.5)
    tile_w = max(1, int(round(width / (cols - (cols - 1) * overlap)))) if cols > 1 else width
    tile_h = max(1, int(round(height / (rows - (rows - 1) * overlap)))) if rows > 1 else height
    step_x = max(1, int(round(tile_w * (1.0 - overlap))))
    step_y = max(1, int(round(tile_h * (1.0 - overlap))))
    for r in range(rows):
        for c in range(cols):
            x0 = min(c * step_x, max(0, width - tile_w))
            y0 = min(r * step_y, max(0, height - tile_h))
            yield x0, y0, min(width, x0 + tile_w), min(height, y0 + tile_h)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between xyxy box arrays a (n) and b (m) -> (n, m)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 0, None)
    area_b = np.clip((bx2 - bx1) * (by2 - by1), 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def containment_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection over the SMALLER box area -> (n, m).

    Catches the tile artefact IoU misses: a partial box (torso) fully inside a
    full-body box scores ~1.0 here but only ~0.3 on IoU. Two distinct objects
    next to each other still score near 0, so they are never merged.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 0, None)
    area_b = np.clip((bx2 - bx1) * (by2 - by1), 0, None)
    smaller = np.minimum(area_a, area_b)
    return np.where(smaller > 0, inter / np.maximum(smaller, 1e-9), 0.0)


def is_duplicate(box_a: np.ndarray, box_b: np.ndarray) -> bool:
    """True when two same-class boxes describe the same physical object."""
    a = box_a.reshape(1, 4)
    b = box_b.reshape(1, 4)
    if iou_matrix(a, b)[0, 0] > RANGE_DEDUPE_IOU:
        return True
    return containment_matrix(a, b)[0, 0] > RANGE_CONTAINMENT


def dedupe_class_aware(detections, labels: list[str], iou_thr: float | None = None):
    """Keep the highest-confidence box per physical object, per class.

    Suppression is IoU *and* containment based — see `is_duplicate`. Applied to
    the merged fast+range set right before the tracker, so one object can only
    ever hand the tracker one box, no matter which pass found it.
    """
    if len(detections) <= 1:
        return detections, labels
    confidence = (
        detections.confidence
        if detections.confidence is not None
        else np.zeros(len(detections))
    )
    order = np.argsort(-confidence)
    keep: list[int] = []
    boxes = detections.xyxy
    for idx in order:
        duplicate = False
        for kept in keep:
            if labels[kept] != labels[idx]:
                continue
            if is_duplicate(boxes[idx], boxes[kept]):
                duplicate = True
                if RANGE_DEBUG:
                    log.info(
                        "dedupe: dropped %s %.2f (iou %.2f / containment %.2f vs %s %.2f)",
                        labels[idx],
                        confidence[idx],
                        iou_matrix(boxes[idx].reshape(1, 4), boxes[kept].reshape(1, 4))[0, 0],
                        containment_matrix(boxes[idx].reshape(1, 4), boxes[kept].reshape(1, 4))[0, 0],
                        labels[kept],
                        confidence[kept],
                    )
                break
        if not duplicate:
            keep.append(int(idx))
    keep.sort()
    if len(keep) == len(detections):
        return detections, labels
    return detections[keep], [labels[i] for i in keep]


class RangeScanner:
    """Background thread analysing the full-resolution frame in tiles.

    Small, distant objects disappear when the fast pass downscales the frame.
    This scanner tiles the full frame, runs YOLO per tile at a low cadence and
    hands the boxes (full-resolution coordinates) to the stream worker, which
    merges them into the shared tracker — one object still gets one box.
    """

    def __init__(self, path: str, detector, frame_source) -> None:
        self._path = path
        self._detector = detector
        self._frame_source = frame_source  # callable -> full-res frame or None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._detections: sv.Detections | None = None
        self._labels: list[str] = []
        self._updated_at: float = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"range-{path}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[sv.Detections | None, list[str], float]:
        with self._lock:
            return self._detections, list(self._labels), self._updated_at

    def _run(self) -> None:
        log.info(
            "[%s] range pass: %dx%d tiles every %.1fs",
            self._path,
            RANGE_TILE_COLS,
            RANGE_TILE_ROWS,
            RANGE_PASS_INTERVAL_SECONDS,
        )
        while not self._stop.is_set():
            frame = self._frame_source()
            if frame is None:
                if self._stop.wait(0.2):
                    break
                continue
            started = time.time()
            try:
                self._scan(frame)
            except Exception as exc:
                log.warning("[%s] range pass error: %s", self._path, exc)
            elapsed = time.time() - started
            remaining = RANGE_PASS_INTERVAL_SECONDS - elapsed
            if self._stop.wait(max(0.1, remaining)):
                break
        log.info("[%s] range pass stopped", self._path)

    def _scan(self, frame) -> None:
        height, width = frame.shape[:2]
        all_xyxy: list[np.ndarray] = []
        all_conf: list[float] = []
        all_labels: list[str] = []
        for x0, y0, x1, y1 in tile_offsets(
            width, height, RANGE_TILE_COLS, RANGE_TILE_ROWS, RANGE_TILE_OVERLAP
        ):
            tile = frame[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            detections, labels = self._detector.detect(tile, conf=RANGE_CONFIDENCE)
            if len(detections) == 0:
                continue
            boxes = detections.xyxy.copy()
            confidence = (
                detections.confidence
                if detections.confidence is not None
                else np.zeros(len(detections))
            )
            # A tile can cut an object in half — the resulting torso/head box is
            # a fragment, not an object. Drop boxes that touch a tile edge which
            # is not also a frame edge; the overlapping neighbour tile sees the
            # whole object anyway.
            margin = 2.0
            keep_tile: list[int] = []
            for i in range(len(boxes)):
                bx1, by1, bx2, by2 = boxes[i]
                truncated = (
                    (bx1 <= margin and x0 > 0)
                    or (by1 <= margin and y0 > 0)
                    or (bx2 >= (x1 - x0) - margin and x1 < width)
                    or (by2 >= (y1 - y0) - margin and y1 < height)
                )
                if truncated:
                    if RANGE_DEBUG:
                        log.info(
                            "[%s] range: dropped truncated %s %.2f at tile %d,%d",
                            self._path,
                            labels[i],
                            confidence[i],
                            x0,
                            y0,
                        )
                    continue
                keep_tile.append(i)
            if not keep_tile:
                continue
            boxes = boxes[keep_tile]
            boxes[:, 0] += x0
            boxes[:, 2] += x0
            boxes[:, 1] += y0
            boxes[:, 3] += y0
            all_xyxy.append(boxes)
            all_conf.append(confidence[keep_tile])
            all_labels.extend([labels[i] for i in keep_tile])
        if all_xyxy:
            merged = sv.Detections(
                xyxy=np.concatenate(all_xyxy),
                confidence=np.concatenate(all_conf),
            )
            merged, all_labels = dedupe_class_aware(merged, all_labels)
        else:
            merged = sv.Detections.empty()
            all_labels = []
        with self._lock:
            self._detections = merged
            self._labels = all_labels
            self._updated_at = time.time()
        log.info(
            "[%s] range pass: %d object(s) in full frame", self._path, len(merged)
        )


# --------------------------------------------------------------------------- #
# One worker per live stream
# --------------------------------------------------------------------------- #


class StreamWorker(threading.Thread):
    """Analyses a single RTSP stream until it is asked to stop."""

    def __init__(self, session_id: str, path: str, detector, store: DetectionStore) -> None:
        super().__init__(daemon=True, name=f"stream-{path}")
        self.session_id = session_id
        self.path = path
        self.detector = detector
        self.store = store
        self.url = RTSP_URL if RTSP_URL else rtsp_url_for(path)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    # -- main loop ---------------------------------------------------------- #

    def run(self) -> None:
        log.info("[%s] worker started (%s)", self.path, self.session_id)
        status.update(self.session_id, path=self.path, connected=False, reconnects=0)
        backoff = BACKOFF_MIN
        reconnects = 0
        while not self._stop.is_set():
            try:
                self._run_once()
                log.warning("[%s] stream ended", self.path)
            except Exception as exc:
                log.warning("[%s] stream error: %s", self.path, exc)

            # Drop stale boxes immediately so the UI never shows frozen overlays.
            self.store.clear(self.session_id)
            reconnects += 1
            status.update(
                self.session_id, connected=False, active_tracks=0, reconnects=reconnects
            )
            if self._stop.wait(backoff):
                break
            backoff = min(BACKOFF_MAX, backoff * 2)

        self.store.clear(self.session_id)
        status.remove(self.session_id)
        log.info("[%s] worker stopped", self.path)

    def _run_once(self) -> None:
        cap = open_capture(self.url)
        if cap is None:
            raise ConnectionError("could not open RTSP stream")

        status.update(self.session_id, connected=True)
        log.info("[%s] connected", self.path)

        # Fast-reacting tracker: short memory for lost tracks and a frame rate
        # that matches the actual analysis rate.
        tracker = sv.ByteTrack(
            lost_track_buffer=TRACKER_LOST_BUFFER,
            frame_rate=max(1, int(round(DETECTION_FPS))),
            track_activation_threshold=min(0.25, DETECTION_CONFIDENCE),
        )
        grabber = FrameGrabber(cap)
        last_seq = 0
        last_inference = 0.0

        # Latest full-resolution frame, shared with the long-range scanner.
        full_frame_slot: dict = {"frame": None, "seq": -1}

        def latest_full_frame():
            return full_frame_slot["frame"]

        scanner = (
            RangeScanner(self.path, self.detector, latest_full_frame)
            if RANGE_PASS_ENABLED
            else None
        )

        try:
            while not self._stop.is_set():
                frame, seq, error = grabber.latest()
                if error:
                    raise ConnectionError(error)
                if frame is None or seq == last_seq:
                    time.sleep(0.02)
                    continue

                now = time.time()
                if now - last_inference < MIN_FRAME_INTERVAL:
                    time.sleep(min(0.02, MIN_FRAME_INTERVAL))
                    continue
                last_seq = seq
                last_inference = now
                status.update(self.session_id, last_frame_at=now)

                src_h, src_w = frame.shape[:2]
                if not src_w or not src_h:
                    continue

                # Share the full-resolution frame with the range scanner before
                # downscaling — that is where the small/distant objects live.
                full_frame_slot["frame"] = frame
                full_frame_slot["seq"] = seq

                # Downscale before inference — boxes stay correct because they
                # are normalised against the frame we actually analysed.
                scale = 1.0
                if INFER_MAX_SIDE and max(src_w, src_h) > INFER_MAX_SIDE:
                    scale = INFER_MAX_SIDE / float(max(src_w, src_h))
                    frame = cv2.resize(
                        frame,
                        (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
                        interpolation=cv2.INTER_LINEAR,
                    )
                height, width = frame.shape[:2]

                started = time.time()
                detections, labels = self.detector.detect(frame)
                infer_ms = (time.time() - started) * 1000.0
                raw_count = len(detections)

                # Merge long-range detections (full-res coords -> fast-frame
                # coords) and suppress duplicates ONCE, over the combined set,
                # so the tracker only ever sees one box per physical object.
                # Stale range results are skipped: their coordinates describe
                # where the object was, and feeding them spawns a ghost track.
                range_count = 0
                if scanner is not None:
                    range_dets, range_labels, range_at = scanner.latest()
                    fresh = (now - range_at) <= RANGE_RESULT_MAX_AGE_SECONDS
                    if range_dets is not None and len(range_dets) > 0 and fresh:
                        range_count = len(range_dets)
                        detections = sv.Detections(
                            xyxy=np.concatenate(
                                [detections.xyxy, range_dets.xyxy * scale]
                            ),
                            confidence=np.concatenate(
                                [
                                    detections.confidence
                                    if detections.confidence is not None
                                    else np.zeros(len(detections)),
                                    range_dets.confidence
                                    if range_dets.confidence is not None
                                    else np.zeros(len(range_dets)),
                                ]
                            ),
                        )
                        labels = labels + list(range_labels)

                merged_count = len(detections)
                detections, labels = dedupe_class_aware(detections, labels)
                if RANGE_DEBUG:
                    log.info(
                        "[%s] sources: fast=%d range=%d merged=%d after-dedupe=%d",
                        self.path,
                        raw_count,
                        range_count,
                        merged_count,
                        len(detections),
                    )

                detections = tracker.update_with_detections(
                    attach_labels(detections, labels)
                )
                rows = build_rows(detections, self.session_id, width, height)

                log.info(
                    "[%s] %d raw -> %d tracked -> %d row(s) in %.0f ms",
                    self.path,
                    raw_count,
                    len(detections),
                    len(rows),
                    infer_ms,
                )

                status.update(self.session_id, active_tracks=len(rows))
                self.store.upsert(rows)
        finally:
            if scanner is not None:
                scanner.stop()
            grabber.stop()
            cap.release()
            status.update(self.session_id, connected=False, active_tracks=0)


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #


def cleanup_loop(store: DetectionStore, workers: dict[str, StreamWorker]) -> None:
    while True:
        # Sweep at least twice per TTL so boxes vanish quickly after an object
        # leaves the frame, with a 0.4s floor to keep write volume sane.
        time.sleep(max(0.4, TRACK_TTL_SECONDS / 2))
        store.prune(list(workers.keys()))


def supervise(detector, store: DetectionStore) -> None:
    """Start a worker per live stream, stop workers whose flight ended."""
    workers: dict[str, StreamWorker] = {}
    threading.Thread(
        target=cleanup_loop, args=(store, workers), daemon=True, name="cleanup"
    ).start()

    if RTSP_URL and FLIGHT_SESSION_ID:
        log.info("Pinned to %s (MEDIAMTX_RTSP_URL override)", FLIGHT_SESSION_ID)
        worker = StreamWorker(FLIGHT_SESSION_ID, "manual", detector, store)
        workers[FLIGHT_SESSION_ID] = worker
        worker.start()
        while True:
            time.sleep(DISCOVERY_INTERVAL_SECONDS)

    while True:
        streams = store.live_streams()
        wanted = {s["flight_session_id"]: s["path"] for s in streams}

        for session_id, worker in list(workers.items()):
            if session_id not in wanted or not worker.is_alive():
                log.info("Stopping worker for %s", session_id)
                worker.stop()
                workers.pop(session_id, None)

        for session_id, path in wanted.items():
            if session_id in workers:
                continue
            if len(workers) >= MAX_STREAMS:
                log.warning(
                    "MAX_STREAMS=%d reached — not analysing %s", MAX_STREAMS, path
                )
                break
            worker = StreamWorker(session_id, path, detector, store)
            workers[session_id] = worker
            worker.start()

        time.sleep(DISCOVERY_INTERVAL_SECONDS)


def main() -> None:
    missing = [
        name
        for name, value in (
            ("SUPABASE_URL", SUPABASE_URL),
            ("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")
    if RTSP_URL and not FLIGHT_SESSION_ID:
        raise SystemExit("MEDIAMTX_RTSP_URL requires FLIGHT_SESSION_ID")

    start_health_server()

    detector = YoloDetector()
    log.info("Detector config: %s", detector.describe())
    log.info(
        "Auto-discovery every %.0fs from %s (max %d stream(s))",
        DISCOVERY_INTERVAL_SECONDS,
        RTSP_BASE_URL,
        MAX_STREAMS,
    )

    store = DetectionStore()
    supervise(detector, store)


if __name__ == "__main__":
    main()


    store = DetectionStore()
    supervise(detector, store)


if __name__ == "__main__":
    main()

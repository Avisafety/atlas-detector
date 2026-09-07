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

# --- Motion pass (third detection source) ---------------------------------- #
# YOLO can only report an object it recognises. A distant object is often just
# a few pixels of "something that moves differently than the ground" long
# before its shape is classifiable. This pass finds those, labels them
# `unknown`, and feeds them into the SAME tracker — so the track keeps its id
# when YOLO later manages to classify it.
UNKNOWN_CLASS = "unknown"
MOTION_PASS_ENABLED = os.environ.get("MOTION_PASS_ENABLED", "true").lower() != "false"
# Analyses per second. Low on purpose: this runs beside the fast + range pass.
MOTION_FPS = float(os.environ.get("MOTION_FPS", "3") or 3)
# Frames are compared at this longest side — enough for motion, cheap on CPU.
MOTION_MAX_SIDE = int(os.environ.get("MOTION_MAX_SIDE", "960") or 960)
# Pixel difference (0-255) that counts as movement after camera compensation.
MOTION_DIFF_THRESHOLD = int(os.environ.get("MOTION_DIFF_THRESHOLD", "18") or 18)
# Area bounds on the analysis frame: kills sensor noise and "the whole picture
# moved" (failed compensation).
MOTION_MIN_AREA_PX = int(os.environ.get("MOTION_MIN_AREA_PX", "12") or 12)
MOTION_MAX_AREA_FRAC = float(os.environ.get("MOTION_MAX_AREA_FRAC", "0.08") or 0.08)
# A candidate must reappear in roughly the same place this many analyses in a
# row before it is published — removes parallax flicker and single-frame blobs.
MOTION_CONFIRM_HITS = int(os.environ.get("MOTION_CONFIRM_HITS", "3") or 3)
# How close two candidates must be (centre distance / box size) to count as the
# same candidate between analyses.
MOTION_MATCH_DISTANCE = float(os.environ.get("MOTION_MATCH_DISTANCE", "2.0") or 2.0)
# Minimum inliers for the camera-motion estimate; below this the frame is
# skipped rather than published as noise.
MOTION_MIN_INLIERS = int(os.environ.get("MOTION_MIN_INLIERS", "12") or 12)
# Fixed confidence written for unknown candidates.
MOTION_CONFIDENCE = float(os.environ.get("MOTION_CONFIDENCE", "0.10") or 0.10)
# Candidates are re-fed between analyses only while fresh.
MOTION_RESULT_MAX_AGE_SECONDS = float(
    os.environ.get("MOTION_RESULT_MAX_AGE_SECONDS", "0") or 0
) or (1.0 / MOTION_FPS + 0.5 if MOTION_FPS > 0 else 1.0)

# --- Locked track ----------------------------------------------------------- #
# A user can lock one object in the UI. A locked track gets a padded ROI
# analysed at native resolution on every fast round (lower threshold) and a
# longer TTL, so it survives shake and short occlusions better than the rest.
LOCK_POLL_SECONDS = float(os.environ.get("LOCK_POLL_SECONDS", "2.0") or 2.0)
LOCK_ROI_PADDING = float(os.environ.get("LOCK_ROI_PADDING", "0.6") or 0.6)
LOCK_ROI_CONFIDENCE = float(os.environ.get("LOCK_ROI_CONFIDENCE", "0.08") or 0.08)
LOCK_TTL_SECONDS = float(os.environ.get("LOCK_TTL_SECONDS", "5.0") or 5.0)

MODEL_PATH = os.environ.get("MODEL_PATH", "yolo26n.pt")

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
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(seconds=TRACK_TTL_SECONDS)).isoformat()
        locked_cutoff = (now - timedelta(seconds=LOCK_TTL_SECONDS)).isoformat()
        try:
            # Unlocked tracks disappear fast so the overlay never lags reality.
            self.client.table("atlas_detections").delete().in_(
                "flight_session_id", session_ids
            ).lt("updated_at", cutoff).eq("is_locked", False).execute()
            # A locked track is what the user is watching — it gets a longer
            # grace period so a moment of occlusion does not drop the box.
            self.client.table("atlas_detections").delete().in_(
                "flight_session_id", session_ids
            ).lt("updated_at", locked_cutoff).eq("is_locked", True).execute()
        except Exception as exc:
            log.warning("Prune failed: %s", exc)

    def locked_tracks(self, session_id: str) -> set[int]:
        """track_ids the user has locked in the UI for this flight session."""
        try:
            rows = (
                self.client.table("atlas_detections")
                .select("track_id")
                .eq("flight_session_id", session_id)
                .eq("is_locked", True)
                .execute()
                .data
                or []
            )
            return {int(r["track_id"]) for r in rows}
        except Exception as exc:
            log.warning("Locked-track lookup failed: %s", exc)
            return set()

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


def build_rows(
    detections,
    session_id: str,
    width: int,
    height: int,
) -> list[dict]:
    """Turn tracked detections into atlas_detections rows (normalised boxes).

    `is_locked` is deliberately NOT written here: it is owned by the UI, and
    the upsert only touches the keys present in the payload.
    """
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
        # A track can exist before any pass could classify it (motion source).
        # It is published as `unknown` and upgraded in place later.
        if not name:
            name = UNKNOWN_CLASS
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


def merge_source(detections, labels, extra, extra_labels, scale: float):
    """Append another source's boxes (full-res coords) to the fast-pass set."""
    if extra is None or len(extra) == 0:
        return detections, labels
    extra_conf = (
        extra.confidence if extra.confidence is not None else np.zeros(len(extra))
    )
    base_conf = (
        detections.confidence
        if detections.confidence is not None
        else np.zeros(len(detections))
    )
    merged = sv.Detections(
        xyxy=np.concatenate([detections.xyxy, extra.xyxy * scale]),
        confidence=np.concatenate([base_conf, extra_conf]),
    )
    return merged, list(labels) + list(extra_labels)


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
    # Classified boxes are considered first, then by confidence: an `unknown`
    # motion candidate must always lose against a real class on the same
    # object, never the other way around.
    unknown = np.array([1 if l == UNKNOWN_CLASS else 0 for l in labels])
    order = np.lexsort((-np.asarray(confidence, dtype=float), unknown))
    keep: list[int] = []
    boxes = detections.xyxy
    for idx in order:
        duplicate = False
        for kept in keep:
            # Same class = same object candidate. `unknown` is compared against
            # every class, so a motion box on an object YOLO also found is
            # suppressed instead of becoming a second box.
            if (
                labels[kept] != labels[idx]
                and UNKNOWN_CLASS not in (labels[kept], labels[idx])
            ):
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


class MotionScanner:
    """Third detection source: objects that MOVE, before they are classifiable.

    YOLO needs to recognise a shape. Something far away is often just a handful
    of pixels moving against the background long before its shape says "boat"
    or "person". This scanner finds those pixels and hands the tracker an
    `unknown` box, which is upgraded to a real class in place the moment the
    fast or range pass can name it.

    The camera itself moves (the drone flies and pans), so a naive frame diff
    lights up the whole picture. Global motion is estimated with ORB features +
    a partial affine fit (RANSAC); the previous frame is warped into the
    current one before differencing, leaving only motion that disagrees with
    the camera.

    False positives are handled with: area bounds (noise / failed
    compensation), a short accumulator so slow movers still build up signal,
    and a confirmation requirement — a candidate must reappear in roughly the
    same place N analyses in a row (kills parallax flicker from terrain edges).
    """

    def __init__(self, path: str, frame_source) -> None:
        self._path = path
        self._frame_source = frame_source
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._detections: sv.Detections | None = None
        self._updated_at: float = 0.0
        # Confirmation state: list of [cx, cy, w, h, hits] in analysis coords.
        self._candidates: list[list[float]] = []
        self._prev_gray = None
        self._accum = None
        self._interval = 1.0 / MOTION_FPS if MOTION_FPS > 0 else 0.33
        self._orb = cv2.ORB_create(nfeatures=600)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"motion-{path}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[sv.Detections | None, float]:
        with self._lock:
            return self._detections, self._updated_at

    # -- internals ---------------------------------------------------------- #

    def _run(self) -> None:
        log.info("[%s] motion pass: %.1f analyses/s", self._path, MOTION_FPS)
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
                log.warning("[%s] motion pass error: %s", self._path, exc)
            remaining = self._interval - (time.time() - started)
            if self._stop.wait(max(0.05, remaining)):
                break
        log.info("[%s] motion pass stopped", self._path)

    def _align(self, prev_gray, gray):
        """Warp `prev_gray` into `gray` using estimated camera motion.

        Returns None when the estimate is not trustworthy (too few matches,
        heavy blur) — skipping a frame beats publishing a screen full of noise.
        """
        kp1, des1 = self._orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self._orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None
        matches = self._matcher.match(des1, des2)
        if len(matches) < MOTION_MIN_INLIERS:
            return None
        matches = sorted(matches, key=lambda m: m.distance)[:200]
        src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if matrix is None or inliers is None or int(inliers.sum()) < MOTION_MIN_INLIERS:
            return None
        h, w = gray.shape[:2]
        return cv2.warpAffine(prev_gray, matrix, (w, h), flags=cv2.INTER_LINEAR)

    def _scan(self, frame) -> None:
        full_h, full_w = frame.shape[:2]
        scale = 1.0
        if MOTION_MAX_SIDE and max(full_w, full_h) > MOTION_MAX_SIDE:
            scale = MOTION_MAX_SIDE / float(max(full_w, full_h))
        small = (
            cv2.resize(
                frame,
                (max(1, int(full_w * scale)), max(1, int(full_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale != 1.0
            else frame
        )
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

        prev = self._prev_gray
        self._prev_gray = gray
        if prev is None or prev.shape != gray.shape:
            self._accum = None
            return

        aligned = self._align(prev, gray)
        if aligned is None:
            return

        diff = cv2.absdiff(aligned, gray).astype(np.float32)
        # Short accumulator: a slow object moves few pixels per analysis, so a
        # single difference is weak. Decaying sum keeps that signal alive.
        if self._accum is None or self._accum.shape != diff.shape:
            self._accum = diff
        else:
            self._accum = self._accum * 0.5 + diff
        mask = (self._accum > MOTION_DIFF_THRESHOLD).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        frame_area = float(h * w)
        raw: list[tuple[float, float, float, float]] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = float(bw * bh)
            if area < MOTION_MIN_AREA_PX:
                continue
            if area > frame_area * MOTION_MAX_AREA_FRAC:
                continue
            # Warp artefacts hug the frame border and are extremely elongated.
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            if touches_border and aspect > 4.0:
                continue
            raw.append((float(x), float(y), float(bw), float(bh)))

        confirmed = self._confirm(raw)

        if confirmed:
            boxes = np.array(
                [
                    [x / scale, y / scale, (x + bw) / scale, (y + bh) / scale]
                    for x, y, bw, bh in confirmed
                ],
                dtype=np.float32,
            )
            detections = sv.Detections(
                xyxy=boxes,
                confidence=np.full(len(boxes), MOTION_CONFIDENCE, dtype=np.float32),
            )
        else:
            detections = sv.Detections.empty()

        with self._lock:
            self._detections = detections
            self._updated_at = time.time()
        if RANGE_DEBUG:
            log.info(
                "[%s] motion: %d blob(s) -> %d confirmed",
                self._path,
                len(raw),
                len(confirmed),
            )

    def _confirm(self, raw) -> list[tuple[float, float, float, float]]:
        """Only publish candidates seen in the same place several times."""
        next_state: list[list[float]] = []
        confirmed: list[tuple[float, float, float, float]] = []
        used: set[int] = set()
        for x, y, bw, bh in raw:
            cx, cy = x + bw / 2.0, y + bh / 2.0
            reach = MOTION_MATCH_DISTANCE * max(bw, bh)
            hits = 1
            for i, prev in enumerate(self._candidates):
                if i in used:
                    continue
                if abs(prev[0] - cx) <= reach and abs(prev[1] - cy) <= reach:
                    hits = int(prev[4]) + 1
                    used.add(i)
                    break
            next_state.append([cx, cy, bw, bh, hits])
            if hits >= MOTION_CONFIRM_HITS:
                confirmed.append((x, y, bw, bh))
        self._candidates = next_state
        return confirmed


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
            # Motion candidates carry a deliberately low, fixed confidence —
            # the tracker must still be allowed to open a track for them.
            track_activation_threshold=min(
                0.25,
                DETECTION_CONFIDENCE,
                MOTION_CONFIDENCE if MOTION_PASS_ENABLED else 1.0,
            ),
        )
        grabber = FrameGrabber(cap)
        last_seq = 0
        last_inference = 0.0

        # Latest full-resolution frame, shared with the background scanners.
        full_frame_slot: dict = {"frame": None, "seq": -1}

        def latest_full_frame():
            return full_frame_slot["frame"]

        scanner = (
            RangeScanner(self.path, self.detector, latest_full_frame)
            if RANGE_PASS_ENABLED
            else None
        )
        motion = (
            MotionScanner(self.path, latest_full_frame)
            if MOTION_PASS_ENABLED
            else None
        )

        # Locked track: polled from Supabase (the UI owns the flag) plus the
        # last known normalised box per track, used to build the priority ROI.
        locked_ids: set[int] = set()
        last_locked_poll = 0.0
        last_boxes: dict[int, tuple[float, float, float, float]] = {}

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

                # Motion candidates: objects that move against the compensated
                # background but are not classifiable yet. Same coordinate
                # conversion, same dedupe, same tracker — a motion box that
                # lands on an object YOLO already found is suppressed, so one
                # object never gets two boxes.
                motion_count = 0
                if motion is not None:
                    motion_dets, motion_at = motion.latest()
                    fresh = (now - motion_at) <= MOTION_RESULT_MAX_AGE_SECONDS
                    if motion_dets is not None and len(motion_dets) > 0 and fresh:
                        motion_count = len(motion_dets)
                        detections, labels = merge_source(
                            detections,
                            labels,
                            motion_dets,
                            [UNKNOWN_CLASS] * motion_count,
                            scale,
                        )

                # Locked track priority: analyse a padded crop around the last
                # known box at native resolution and with a lower threshold, so
                # the object the user is watching keeps being found through
                # shake and partial occlusion.
                lock_count = 0
                if locked_ids and full_frame_slot["frame"] is not None:
                    lock_dets, lock_labels = self._detect_locked_rois(
                        full_frame_slot["frame"], locked_ids, last_boxes
                    )
                    if lock_dets is not None and len(lock_dets) > 0:
                        lock_count = len(lock_dets)
                        detections, labels = merge_source(
                            detections, labels, lock_dets, lock_labels, scale
                        )

                merged_count = len(detections)
                detections, labels = dedupe_class_aware(detections, labels)
                if RANGE_DEBUG:
                    log.info(
                        "[%s] sources: fast=%d range=%d motion=%d lock=%d merged=%d after-dedupe=%d",
                        self.path,
                        raw_count,
                        range_count,
                        motion_count,
                        lock_count,
                        merged_count,
                        len(detections),
                    )

                detections = tracker.update_with_detections(
                    attach_labels(detections, labels)
                )
                rows = build_rows(detections, self.session_id, width, height)

                # Remember where each track was (normalised) for the lock ROI.
                last_boxes = {
                    int(r["track_id"]): (
                        r["bbox"]["x"],
                        r["bbox"]["y"],
                        r["bbox"]["width"],
                        r["bbox"]["height"],
                    )
                    for r in rows
                }

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

                # The lock flag is written by the UI — poll it at a low rate.
                if now - last_locked_poll >= LOCK_POLL_SECONDS:
                    last_locked_poll = now
                    new_locked = self.store.locked_tracks(self.session_id)
                    if new_locked != locked_ids:
                        log.info("[%s] locked tracks: %s", self.path, sorted(new_locked))
                    locked_ids = new_locked
        finally:
            if scanner is not None:
                scanner.stop()
            if motion is not None:
                motion.stop()
            grabber.stop()
            cap.release()
            status.update(self.session_id, connected=False, active_tracks=0)

    def _detect_locked_rois(
        self,
        full_frame,
        locked_ids: set[int],
        last_boxes: dict[int, tuple[float, float, float, float]],
    ):
        """Native-resolution detection inside a padded crop per locked track."""
        height, width = full_frame.shape[:2]
        boxes: list[np.ndarray] = []
        confs: list[float] = []
        labels: list[str] = []
        for track_id in locked_ids:
            box = last_boxes.get(track_id)
            if not box:
                continue
            nx, ny, nw, nh = box
            pad_w = nw * LOCK_ROI_PADDING
            pad_h = nh * LOCK_ROI_PADDING
            x0 = int(max(0, (nx - pad_w) * width))
            y0 = int(max(0, (ny - pad_h) * height))
            x1 = int(min(width, (nx + nw + pad_w) * width))
            y1 = int(min(height, (ny + nh + pad_h) * height))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = full_frame[y0:y1, x0:x1]
            try:
                dets, det_labels = self.detector.detect(crop, conf=LOCK_ROI_CONFIDENCE)
            except Exception as exc:
                log.warning("[%s] lock ROI failed: %s", self.path, exc)
                continue
            if len(dets) == 0:
                continue
            xyxy = dets.xyxy.copy()
            xyxy[:, 0] += x0
            xyxy[:, 2] += x0
            xyxy[:, 1] += y0
            xyxy[:, 3] += y0
            confidence = (
                dets.confidence
                if dets.confidence is not None
                else np.zeros(len(dets))
            )
            boxes.append(xyxy)
            confs.append(confidence)
            labels.extend(det_labels)
        if not boxes:
            return None, []
        return (
            sv.Detections(
                xyxy=np.concatenate(boxes), confidence=np.concatenate(confs)
            ),
            labels,
        )


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

"""atlas-detector — continuous object detection and tracking on Atlas drone video.

Reads one MediaMTX stream over RTSP, runs YOLOv8n + ByteTrack on a subset of the
frames, and upserts one row per tracked object into the Supabase table
`atlas_detections` (unique on flight_session_id + track_id). A cleanup loop
deletes tracks that stopped being updated.

CPU only. Everything is configured through environment variables:

  MEDIAMTX_RTSP_URL          rtsp://live-video.internal:8554/<serial>/<sensor>?detector=<secret>
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  FLIGHT_SESSION_ID          active_flights.id the detections belong to
  DETECTION_FPS              analysis rate, default 5
  DETECTION_CLASSES          default person,car,truck,bus,motorcycle,boat
  DETECTION_CONFIDENCE       default 0.35
  TRACK_TTL_SECONDS          default 3
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
from ultralytics import YOLO

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

DETECTION_FPS = float(os.environ.get("DETECTION_FPS", "5") or 5)
DETECTION_CONFIDENCE = float(os.environ.get("DETECTION_CONFIDENCE", "0.35") or 0.35)
TRACK_TTL_SECONDS = float(os.environ.get("TRACK_TTL_SECONDS", "3") or 3)
DETECTION_CLASSES = [
    c.strip().lower()
    for c in os.environ.get(
        "DETECTION_CLASSES", "person,car,truck,bus,motorcycle,boat"
    ).split(",")
    if c.strip()
]

MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n.pt")

# Backoff bounds for reconnecting to MediaMTX.
BACKOFF_MIN = 1.0
BACKOFF_MAX = 30.0

# Force TCP for RTSP — UDP is unreliable across the Fly private network.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

MIN_FRAME_INTERVAL = 1.0 / DETECTION_FPS if DETECTION_FPS > 0 else 0.2


# --------------------------------------------------------------------------- #
# Shared status (for /health)
# --------------------------------------------------------------------------- #


class Status:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.connected = False
        self.last_frame_at: float | None = None
        self.active_tracks = 0
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
                "reconnects": self.reconnects,
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "flight_session_id": FLIGHT_SESSION_ID or None,
                "detection_fps": DETECTION_FPS,
                "classes": DETECTION_CLASSES,
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
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=TRACK_TTL_SECONDS)).isoformat()
        try:
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
        time.sleep(max(1.0, TRACK_TTL_SECONDS / 2))
        store.prune()


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


def open_capture(url: str) -> cv2.VideoCapture | None:
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


def run_stream(model: YOLO, class_ids: dict[int, str], store: DetectionStore) -> None:
    """One connection attempt. Returns when the stream drops."""
    cap = open_capture(RTSP_URL)
    if cap is None:
        raise ConnectionError("could not open RTSP stream")

    with status.lock:
        status.connected = True
    log.info("Connected to stream")

    tracker = sv.ByteTrack()
    wanted = set(class_ids.keys())
    last_inference = 0.0
    empty_reads = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                empty_reads += 1
                if empty_reads > 60:
                    raise ConnectionError("stream returned no frames")
                time.sleep(0.05)
                continue
            empty_reads = 0

            now = time.time()
            with status.lock:
                status.last_frame_at = now

            # Frame skipping: only run inference at DETECTION_FPS.
            if now - last_inference < MIN_FRAME_INTERVAL:
                continue
            last_inference = now

            height, width = frame.shape[:2]
            if not width or not height:
                continue

            result = model.predict(
                frame,
                conf=DETECTION_CONFIDENCE,
                classes=sorted(wanted),
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            rows: list[dict] = []
            timestamp = datetime.now(timezone.utc).isoformat()
            for xyxy, conf, class_id, track_id in zip(
                detections.xyxy,
                detections.confidence if detections.confidence is not None else np.zeros(len(detections)),
                detections.class_id if detections.class_id is not None else np.full(len(detections), -1),
                detections.tracker_id if detections.tracker_id is not None else np.full(len(detections), -1),
            ):
                if track_id is None or int(track_id) < 0:
                    continue
                name = class_ids.get(int(class_id))
                if name is None:
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

            with status.lock:
                status.active_tracks = len(rows)

            store.upsert(rows)
    finally:
        cap.release()
        with status.lock:
            status.connected = False
            status.active_tracks = 0


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

    log.info("Loading YOLOv8n (%s)", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    # Map the configured class names to the model's class ids.
    names: dict[int, str] = {int(k): str(v).lower() for k, v in model.names.items()}
    class_ids = {cid: name for cid, name in names.items() if name in DETECTION_CLASSES}
    unknown = set(DETECTION_CLASSES) - set(class_ids.values())
    if unknown:
        log.warning("Unknown classes ignored: %s", ", ".join(sorted(unknown)))
    if not class_ids:
        raise SystemExit("No valid classes in DETECTION_CLASSES")
    log.info("Detecting: %s", ", ".join(sorted(class_ids.values())))

    store = DetectionStore()
    store.clear()
    threading.Thread(target=cleanup_loop, args=(store,), daemon=True, name="cleanup").start()

    backoff = BACKOFF_MIN
    while True:
        try:
            run_stream(model, class_ids, store)
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

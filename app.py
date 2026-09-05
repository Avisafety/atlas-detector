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
  INFER_MAX_SIDE                downscale longest side before inference, default 480
  TRACKER_LOST_BUFFER           analysed frames a lost track survives, default 5
  TRACK_TTL_SECONDS             default 0.8
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
                "engine": DETECTION_ENGINE,
                "detection_fps": ANALYSIS_FPS,
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

    def describe(self) -> str:
        return (
            f"engine=grounding_dino checkpoint={GROUNDING_DINO_CHECKPOINT} "
            f"prompt=\"{self.prompt}\" box_threshold={DETECTION_BOX_THRESHOLD} "
            f"text_threshold={DETECTION_TEXT_THRESHOLD} fps={ANALYSIS_FPS}"
        )

    def detect(self, frame) -> tuple[sv.Detections, list[str]]:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, text=self.prompt, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=DETECTION_BOX_THRESHOLD,
            text_threshold=DETECTION_TEXT_THRESHOLD,
            target_sizes=[(height, width)],
        )[0]

        boxes = results["boxes"].cpu().numpy().astype(np.float32)
        scores = results["scores"].cpu().numpy().astype(np.float32)
        labels = [str(t).strip().lower() for t in results.get("text_labels", results["labels"])]

        if len(boxes) == 0:
            return sv.Detections.empty(), []

        detections = sv.Detections(
            xyxy=boxes.reshape(-1, 4),
            confidence=scores,
            class_id=np.zeros(len(boxes), dtype=int),
        )
        return detections, labels


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
        track_activation_threshold=min(0.25, DETECTION_CONFIDENCE),
    )
    grabber = FrameGrabber(cap)
    last_seq = 0
    last_inference = 0.0

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
            src_h, src_w = frame.shape[:2]
            if not src_w or not src_h:
                continue

            # Downscale before inference — boxes stay correct because they are
            # normalised against the frame we actually analysed.
            if INFER_MAX_SIDE and max(src_w, src_h) > INFER_MAX_SIDE:
                scale = INFER_MAX_SIDE / float(max(src_w, src_h))
                frame = cv2.resize(
                    frame,
                    (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
                    interpolation=cv2.INTER_LINEAR,
                )
            height, width = frame.shape[:2]

            detections, labels = detector.detect(frame)
            # Keep the label list index-aligned with the detections we track.
            detections.data = dict(detections.data or {})
            detections.data["label"] = np.array(labels, dtype=object)
            detections = tracker.update_with_detections(detections)

            tracked_labels = detections.data.get("label") if detections.data else None

            rows: list[dict] = []
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

            with status.lock:
                status.active_tracks = len(rows)

            store.upsert(rows)
    finally:
        grabber.stop()
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

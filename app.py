"""atlas-detector — continuous object detection and tracking on Atlas drone video.

Discovers every active flight that has a live Atlas video stream, reads each
stream over RTSP from MediaMTX, runs YOLO26n (ONNX Runtime) + ByteTrack on a
subset of the frames, and upserts one row per tracked object into the Supabase table
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
  TRACKER_LOST_BUFFER           analysed frames a lost track survives, default 15
  TRACK_TTL_SECONDS             default 0.8
  RANGE_PASS_ENABLED            tiled full-res pass for small/distant objects
  RANGE_PASS_INTERVAL_SECONDS   cadence of the range pass, default 2.0
  RANGE_TILE_COLS / _ROWS       tile grid, default 3x2 with 15% overlap
  RANGE_TILE_OVERLAP
  RANGE_CONFIDENCE              separate threshold for the range pass, 0.15
  RANGE_DEDUPE_IOU              overlap at which duplicate boxes are merged, 0.5
  MOTION_PASS_ENABLED           motion-based pass for unclassifiable objects
  MOTION_FPS                    analyses per second, default 3
  MOTION_MAX_SIDE               analysis resolution, default 960
  MOTION_DIFF_THRESHOLD         pixel diff counted as movement, default 18
  MOTION_MIN_AREA_PX            smallest blob accepted, default 12
  MOTION_MAX_AREA_FRAC          largest blob as fraction of frame, default 0.08
  MOTION_CONFIRM_HITS           analyses in a row before publishing, default 3
  MOTION_MIN_INLIERS            RANSAC inliers required, default 12
  MOTION_CONFIDENCE             fixed confidence for unknown boxes, default 0.10
  MODEL_PATH                    default yolo26n.onnx (falls back to FALLBACK_MODEL_PATH)
  FALLBACK_MODEL_PATH           default yolo26n.pt
  LOG_SUMMARY_SECONDS           per-stream summary log + /health metrics, default 10
  LOG_EVERY_FRAME               one log line per analysed frame, default false
  DB_WRITER_THREADS             parallel Supabase writers, default MAX_STREAMS
  LOCK_POLL_SECONDS             how often the UI lock flag is read, default 2.0
  LOCK_ROI_PADDING              padding around the locked box, default 0.6
  LOCK_ROI_CONFIDENCE           threshold inside the locked ROI, default 0.08
  LOCK_TTL_SECONDS              grace period for a locked track, default 5.0
"""

from __future__ import annotations

import asyncio
import collections
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
# httpx logs every Supabase request at INFO — ~5 lines/s from the prune loop
# alone — which buried the per-stream summaries and shrank Fly's log buffer to
# ~20 seconds. Failed requests still surface through our own warnings.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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
# How many analysed frames a lost track survives inside the tracker. Kept
# generous: when the camera pans, a track can miss a couple of rounds before
# the boxes line up again — a short buffer turns that into a brand new id.
TRACKER_LOST_BUFFER = int(os.environ.get("TRACKER_LOST_BUFFER", "15") or 15)
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

# --- Crop pass (track-guided, native resolution) ---------------------------- #
# The range pass FINDS small, distant objects (every few seconds). The crop pass
# KEEPS them: a few times per second, a small window is cut from the
# full-resolution frame around every small track (where the tracker expects it)
# and analysed 1:1, all windows in one batched inference. Without it a distant
# object only the range pass can see lives for ~0.5 s after each range scan.
CROP_PASS_ENABLED = os.environ.get("CROP_PASS_ENABLED", "true").lower() != "false"
CROP_FPS = float(os.environ.get("CROP_FPS", "2") or 2)
# Window side in full-resolution px, analysed at exactly this size (imgsz).
CROP_SIZE = int(os.environ.get("CROP_SIZE", "320") or 320)
MAX_CROPS = int(os.environ.get("MAX_CROPS", "4") or 4)
# A track counts as small (gets a window) below this longest side, in px of the
# fast-pass frame.
SMALL_TRACK_MAX_SIDE = float(os.environ.get("SMALL_TRACK_MAX_SIDE", "48") or 48)
CROP_CONFIDENCE = float(os.environ.get("CROP_CONFIDENCE", "0.15") or 0.15)
CROP_RESULT_MAX_AGE_SECONDS = (1.0 / CROP_FPS if CROP_FPS > 0 else 1.0) + 0.3
# The crop pass only runs while the model has room to spare: when a stream's
# fast-pass frames wait longer than this for the model on average (several
# streams sharing it), its crops pause until the wait is back under half.
CROP_MAX_QUEUE_MS = float(os.environ.get("CROP_MAX_QUEUE_MS", "25") or 25)

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
MOTION_MIN_AREA_PX = int(os.environ.get("MOTION_MIN_AREA_PX", "8") or 8)
MOTION_MAX_AREA_FRAC = float(os.environ.get("MOTION_MAX_AREA_FRAC", "0.12") or 0.12)
# A candidate must reappear in roughly the same place this many analyses in a
# row before it is published — removes parallax flicker and single-frame blobs.
# Two is enough at 3 analyses/second: three meant slow movers (leaves, a distant
# boat) were dropped before they ever reached the tracker.
MOTION_CONFIRM_HITS = int(os.environ.get("MOTION_CONFIRM_HITS", "2") or 2)
# How close two candidates must be (centre distance / box size) to count as the
# same candidate between analyses.
MOTION_MATCH_DISTANCE = float(os.environ.get("MOTION_MATCH_DISTANCE", "3.0") or 3.0)
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
# A locked object is followed by a PIXEL tracker (CSRT) instead of by the
# detector: it keeps its box and its id while the camera moves, even when YOLO
# loses the object for a moment, and it is published outside ByteTrack so the
# lock can never turn into two competing boxes.
LOCK_POLL_SECONDS = float(os.environ.get("LOCK_POLL_SECONDS", "2.0") or 2.0)
LOCK_TTL_SECONDS = float(os.environ.get("LOCK_TTL_SECONDS", "5.0") or 5.0)
# Overlap at which a detection is accepted as "this is the locked object" and
# used to re-centre the pixel tracker and name the class.
LOCK_MATCH_IOU = float(os.environ.get("LOCK_MATCH_IOU", "0.3") or 0.3)
# How long the lock survives with neither pixel tracking nor a detection before
# it is released and the UI clears it.
LOCK_GRACE_SECONDS = float(os.environ.get("LOCK_GRACE_SECONDS", "4.0") or 4.0)
# A matched detection only re-initialises the lock's pixel tracker when the two
# boxes have drifted apart (IoU below this). Re-initialising CSRT on every
# matched frame cost ~25 ms per lock per frame.
LOCK_REINIT_IOU = float(os.environ.get("LOCK_REINIT_IOU", "0.6") or 0.6)

# --- Tracker ---------------------------------------------------------------- #
# botsort   — Ultralytics BoT-SORT with global motion compensation: the camera
#             movement between frames (drone flying, gimbal panning) is
#             estimated with sparse optical flow and removed from every track's
#             predicted position before matching, so a pan no longer breaks
#             tracks into new ids. New tracks must be seen on two consecutive
#             frames before they are shown.
# bytetrack — the previous supervision ByteTrack (rollback).
TRACKER_IMPL = os.environ.get("TRACKER_IMPL", "botsort").strip().lower()
# Detections at or above this start and extend tracks directly ...
TRACK_HIGH_THRESH = float(os.environ.get("TRACK_HIGH_THRESH", "0.25") or 0.25)
# ... detections between LOW and HIGH may only extend an existing track (the
# ByteTrack idea: weak detections keep occluded/distant objects alive).
TRACK_LOW_THRESH = float(os.environ.get("TRACK_LOW_THRESH", "0.10") or 0.10)
# Minimum score for an unmatched detection to open a new track.
NEW_TRACK_THRESH = float(os.environ.get("NEW_TRACK_THRESH", "0.25") or 0.25)
TRACK_MATCH_THRESH = float(os.environ.get("TRACK_MATCH_THRESH", "0.8") or 0.8)
# Motion compensation method: maskedFlow (ours, default) | sparseOptFlow | orb
# | ecc | none. maskedFlow is sparse optical flow on the BACKGROUND only.
TRACKER_GMC = os.environ.get("TRACKER_GMC", "maskedFlow").strip()
# The published class of a track is a confidence-weighted vote over its last
# N detections, so one object no longer flips car -> truck -> car.
CLASS_VOTE_WINDOW = int(os.environ.get("CLASS_VOTE_WINDOW", "10") or 10)
# A new range / motion result may open tracks on this many consecutive frames
# (a new track needs two to be confirmed); later re-fed copies only extend.
NEW_RESULT_FRAMES = 2
# The range pass exists for SMALL, distant objects; anything larger than this
# (longest side, px in the fast-pass frame) is found by the fast pass itself, so
# a range box that big may extend a track but never open a second one on the
# same object.
RANGE_NEW_TRACK_MAX_SIDE = float(os.environ.get("RANGE_NEW_TRACK_MAX_SIDE", "64") or 64)

# Classes YOLO regularly swaps between on the same object. Treated as one class
# during duplicate suppression, so a car does not also get a truck box.
CONFUSABLE_CLASS_GROUPS = (
    {"car", "truck", "bus", "train", "boat"},
    {"person", "bicycle", "motorcycle"},
    {"bird", "airplane", "kite"},
    {"dog", "sheep", "cow", "horse"},
)


def class_group(label: str) -> str:
    """Group name used for duplicate suppression (see CONFUSABLE_CLASS_GROUPS)."""
    for group in CONFUSABLE_CLASS_GROUPS:
        if label in group:
            return next(iter(sorted(group)))
    return label

# ONNX Runtime export of YOLO26n (baked into the image): ~3x faster than
# PyTorch on CPU with the same boxes. If it is missing or fails to load, the
# detector falls back to the PyTorch weights, so a bad export never takes the
# service down.
MODEL_PATH = os.environ.get("MODEL_PATH", "yolo26n.onnx")
# Inference size for the fast and range passes (the model's native 640).
DETECTOR_IMGSZ = 640
FALLBACK_MODEL_PATH = os.environ.get("FALLBACK_MODEL_PATH", "yolo26n.pt")

# Observability: one summary line per stream every LOG_SUMMARY_SECONDS instead
# of one line per analysed frame (40+ lines/s made the live log unreadable).
LOG_SUMMARY_SECONDS = float(os.environ.get("LOG_SUMMARY_SECONDS", "10") or 10)
LOG_EVERY_FRAME = os.environ.get("LOG_EVERY_FRAME", "false").lower() == "true"

# Parallel Supabase writer threads (one lane per stream slot by default).
DB_WRITER_THREADS = int(os.environ.get("DB_WRITER_THREADS", "0") or 0) or MAX_STREAMS

# How boxes reach the browser:
#   postgres  — upsert every track into atlas_detections (Postgres Changes)
#   broadcast — one Realtime Broadcast snapshot per frame on a private
#               channel; only LOCKED tracks are still written to the table
#   both      — do both (transition period; frontend picks broadcast)
DETECTIONS_TRANSPORT = os.environ.get("DETECTIONS_TRANSPORT", "both").strip().lower()
if DETECTIONS_TRANSPORT not in ("postgres", "broadcast", "both"):
    DETECTIONS_TRANSPORT = "both"
BROADCAST_ENABLED = DETECTIONS_TRANSPORT in ("broadcast", "both")
# Snapshots per second per stream. The frontend extrapolates with vx/vy, so
# 10 Hz looks smooth while keeping Realtime's messages/second quota (counted
# per recipient) far away. Analysis itself still runs at DETECTION_FPS.
BROADCAST_MAX_HZ = float(os.environ.get("BROADCAST_MAX_HZ", "10") or 10)
# With nothing to show, a keep-alive empty snapshot this often lets the
# frontend know Broadcast is alive without spending quota.
BROADCAST_IDLE_SECONDS = float(os.environ.get("BROADCAST_IDLE_SECONDS", "1.0") or 1.0)
BROADCAST_TOPIC_PREFIX = os.environ.get("BROADCAST_TOPIC_PREFIX", "atlas-detections:")
BROADCAST_EVENT = os.environ.get("BROADCAST_EVENT", "tracks")

# Scale to zero. After this many minutes without a single analysed frame the
# process exits cleanly and Fly stops the machine (restart policy on-failure).
# MediaMTX wakes it again with a request to GET /wake on the app's private
# Flycast address when a drone starts publishing or a viewer starts watching;
# Fly starts the stopped machine on that request. "Active flight" alone does
# not count: a flight whose stream is gone (RTSP 404) must not keep it awake.
# 0 = never sleep.
IDLE_EXIT_MINUTES = float(os.environ.get("IDLE_EXIT_MINUTES", "0") or 0)

# Inference priorities: lower value runs first when the model is contended.
PRIORITY_FAST = 0
PRIORITY_CROP = 1
PRIORITY_RANGE = 2

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
        self.model = MODEL_PATH
        self.writer: dict = {}
        # Latest moment anyone needed the detector: an analysed frame or a
        # /wake request. Drives IDLE_EXIT_MINUTES.
        self.last_activity = self.started_at
        self.broadcast: dict = {}

    def update(self, session_id: str, **fields) -> None:
        with self.lock:
            entry = self.streams.setdefault(session_id, {})
            entry.update(fields)
            if "last_frame_at" in fields:
                self.last_activity = max(self.last_activity, fields["last_frame_at"])

    def touch(self) -> None:
        with self.lock:
            self.last_activity = time.time()

    def idle_seconds(self) -> float:
        with self.lock:
            return time.time() - self.last_activity

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
                        # Rolling window, refreshed every LOG_SUMMARY_SECONDS.
                        "metrics": entry.get("metrics", {}),
                    }
                )
            return {
                "ok": True,
                "service": "atlas-detector",
                "engine": "yolo",
                "model": self.model,
                "db_writer": self.writer,
                "transport": DETECTIONS_TRANSPORT,
                "tracker": TRACKER_IMPL,
                "broadcast": self.broadcast,
                "detection_fps": DETECTION_FPS,
                "confidence": DETECTION_CONFIDENCE,
                "classes": DETECTION_CLASSES,
                "max_streams": MAX_STREAMS,
                "active_streams": len(streams),
                "idle_seconds": round(time.time() - self.last_activity, 1),
                "idle_exit_minutes": IDLE_EXIT_MINUTES,
                "streams": streams,
                "uptime_seconds": round(time.time() - self.started_at, 1),
            }


status = Status()


# Set by GET /wake: run discovery now instead of at the next interval.
wake_event = threading.Event()


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] == "/wake":
            # Reaching this line already did the real work: Fly started the
            # machine to deliver the request. Count it as activity and look
            # for the new stream right away.
            status.touch()
            wake_event.set()
            payload = b'{"ok": true, "woken": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
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


class PriorityLock:
    """Mutex that hands the model to the most urgent waiter first.

    A plain Lock lets a range-pass tile grab the model while a fast-pass frame
    is waiting, which shows up directly as box latency. Here waiters are
    served by (priority, arrival order): fast-pass frames of every stream go
    before range tiles, and equal priorities stay first come, first served.

    Anti-starvation: with several streams the fast passes alone can keep the
    model busy all the time, which would stop the range pass completely (and
    with it every distant object). A waiter that has queued longer than
    `max_wait` seconds is therefore served next, whatever its priority.
    """

    def __init__(self, max_wait: float = 0.25) -> None:
        self._cond = threading.Condition()
        self._held = False
        self._queue: list[tuple[int, int, float]] = []
        self._seq = 0
        self._max_wait = max_wait

    def _next(self) -> tuple[int, int, float]:
        oldest = min(self._queue, key=lambda t: t[1])
        if time.monotonic() - oldest[2] >= self._max_wait:
            return oldest
        return min(self._queue)

    def acquire(self, priority: int) -> None:
        with self._cond:
            self._seq += 1
            ticket = (priority, self._seq, time.monotonic())
            self._queue.append(ticket)
            while self._held or self._next() != ticket:
                # Timed wait so an aged waiter is noticed even without a release.
                self._cond.wait(self._max_wait)
            self._queue.remove(ticket)
            self._held = True

    def release(self) -> None:
        with self._cond:
            self._held = False
            self._cond.notify_all()


class YoloDetector:
    """YOLO26n (NMS-free, end-to-end) on a fixed COCO class list. Shared by every stream worker."""

    name = "yolo"

    def __init__(self) -> None:
        self.model, self.model_path = self._load()
        self._lock = PriorityLock()

        names: dict[int, str] = {int(k): str(v).lower() for k, v in self.model.names.items()}
        self.class_ids = {cid: n for cid, n in names.items() if n in DETECTION_CLASSES}
        unknown = set(DETECTION_CLASSES) - set(self.class_ids.values())
        if unknown:
            log.warning("Unknown classes ignored: %s", ", ".join(sorted(unknown)))
        if not self.class_ids:
            raise SystemExit("No valid classes in DETECTION_CLASSES")

    @staticmethod
    def _load():
        """Load MODEL_PATH, falling back to FALLBACK_MODEL_PATH on any failure.

        The candidate must also survive one real inference: an export that
        loads but cannot run is caught here, at startup, not mid-flight.
        """
        from ultralytics import YOLO

        candidates = [MODEL_PATH]
        if FALLBACK_MODEL_PATH and FALLBACK_MODEL_PATH != MODEL_PATH:
            candidates.append(FALLBACK_MODEL_PATH)
        last_exc: Exception | None = None
        for path in candidates:
            if not path.endswith(".pt") and not os.path.exists(path):
                log.warning("Model %s not found, trying next", path)
                continue
            try:
                log.info("Loading YOLO26 (%s)", path)
                model = YOLO(path, task="detect")
                model.predict(np.zeros((360, 640, 3), dtype=np.uint8), verbose=False)
                return model, path
            except Exception as exc:
                last_exc = exc
                log.warning("Model %s failed to load (%s), trying next", path, exc)
        raise SystemExit(f"No usable model (last error: {last_exc})")

    def describe(self) -> str:
        return (
            f"engine=yolo model={self.model_path} "
            f"classes={','.join(sorted(self.class_ids.values()))} "
            f"confidence={DETECTION_CONFIDENCE} fps={DETECTION_FPS}"
        )

    def detect(
        self,
        frame,
        conf: float | None = None,
        priority: int = PRIORITY_FAST,
        timing: dict | None = None,
    ) -> tuple[sv.Detections, list[str]]:
        # One model instance shared by all workers: serialise inference so two
        # streams cannot corrupt each other's state. Fast-pass frames jump the
        # queue ahead of range tiles (see PriorityLock).
        queued = time.perf_counter()
        self._lock.acquire(priority)
        try:
            started = time.perf_counter()
            result = self.model.predict(
                frame,
                conf=conf if conf is not None else DETECTION_CONFIDENCE,
                classes=sorted(self.class_ids.keys()),
                # Always explicit: Ultralytics keeps the last call's imgsz, so
                # a crop-pass call at 320 would otherwise shrink every later
                # fast/range inference too.
                imgsz=DETECTOR_IMGSZ,
                verbose=False,
            )[0]
            finished = time.perf_counter()
        finally:
            self._lock.release()
        if timing is not None:
            timing["wait_ms"] = (started - queued) * 1000.0
            timing["infer_ms"] = (finished - started) * 1000.0
        return self._convert(result)

    def detect_batch(
        self,
        images: list,
        conf: float,
        imgsz: int,
        priority: int = PRIORITY_CROP,
        timing: dict | None = None,
    ) -> list[tuple[sv.Detections, list[str]]]:
        """Several images in ONE inference (the ONNX export has a dynamic
        batch dimension), each analysed at `imgsz`."""
        if not images:
            return []
        queued = time.perf_counter()
        self._lock.acquire(priority)
        try:
            started = time.perf_counter()
            results = self.model.predict(
                images,
                conf=conf,
                classes=sorted(self.class_ids.keys()),
                imgsz=imgsz,
                verbose=False,
            )
            finished = time.perf_counter()
        finally:
            self._lock.release()
        if timing is not None:
            timing["wait_ms"] = (started - queued) * 1000.0
            timing["infer_ms"] = (finished - started) * 1000.0
        return [self._convert(r) for r in results]

    def _convert(self, result) -> tuple[sv.Detections, list[str]]:
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
        # Writes happen on background threads: a network round-trip inside the
        # detection loop costs 50-200 ms per frame, which shows up directly as
        # boxes lagging behind the video. Only the newest rows per session are
        # kept, so a slow write is skipped instead of queued.
        #
        # One writer "lane" per stream slot, each with its own client: a single
        # writer serialises every stream behind one ~50 ms round trip, which
        # caps the whole machine at ~20 writes/s. A session always maps to the
        # same lane, so its writes stay in order; sessions are spread over the
        # lanes round-robin (hashing can put every session in one lane).
        self._stats_lock = threading.Lock()
        self._lane_of: dict[str, dict] = {}
        self._next_lane = 0
        self._stats = self._empty_stats()
        self.broadcaster = Broadcaster() if BROADCAST_ENABLED else None
        self._lanes = []
        for idx in range(max(1, DB_WRITER_THREADS)):
            lane = {
                "client": self.client if idx == 0 else create_client(
                    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
                ),
                "pending": {},
                "lock": threading.Lock(),
                "event": threading.Event(),
            }
            self._lanes.append(lane)
            threading.Thread(
                target=self._writer_loop, args=(lane, idx == 0), daemon=True,
                name=f"db-writer-{idx}",
            ).start()

    # -- detections --------------------------------------------------------- #

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        key = str(rows[0].get("flight_session_id") or "")
        lane = self._lane_for(key)
        with lane["lock"]:
            skipped = key in lane["pending"]
            lane["pending"][key] = rows
        if skipped:
            # The writer did not get to the previous batch in time.
            with self._stats_lock:
                self._stats["skipped"] += 1
        lane["event"].set()

    def _lane_for(self, key: str) -> dict:
        with self._stats_lock:
            lane = self._lane_of.get(key)
            if lane is None:
                lane = self._lanes[self._next_lane % len(self._lanes)]
                self._next_lane += 1
                self._lane_of[key] = lane
            return lane

    def _writer_loop(self, lane: dict, reports: bool) -> None:
        window_start = time.time()
        while True:
            lane["event"].wait(0.1)
            lane["event"].clear()
            with lane["lock"]:
                batches = list(lane["pending"].values())
                lane["pending"].clear()
            for rows in batches:
                self._write(lane["client"], rows)
            if reports:
                now = time.time()
                if now - window_start >= LOG_SUMMARY_SECONDS:
                    self._report(now - window_start)
                    window_start = now

    def _write(self, client, rows: list[dict]) -> None:
        started = time.perf_counter()
        failed = False
        try:
            client.table("atlas_detections").upsert(
                rows, on_conflict="flight_session_id,track_id"
            ).execute()
        except Exception as exc:  # never let a write error kill the loop
            failed = True
            log.warning("Upsert failed: %s", exc)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._stats_lock:
            self._stats["writes"] += 1
            self._stats["failed"] += int(failed)
            self._stats["total_ms"] += elapsed_ms
            self._stats["max_ms"] = max(self._stats["max_ms"], elapsed_ms)

    def _report(self, window: float) -> None:
        """Publish and log the writers' throughput for the last window."""
        with self._stats_lock:
            s = self._stats
            self._stats = self._empty_stats()
        writes = s["writes"]
        summary = {
            "writes_per_second": round(writes / window, 1),
            "avg_write_ms": round(s["total_ms"] / writes, 1) if writes else None,
            "max_write_ms": round(s["max_ms"], 1) if writes else None,
            "skipped_batches": s["skipped"],
            "failed_writes": s["failed"],
            "writer_threads": len(self._lanes),
        }
        status.writer = summary
        if writes or s["skipped"] or s["failed"]:
            log.info(
                "db writer: %.1f writes/s, avg %s ms, max %s ms, %d skipped, %d failed",
                summary["writes_per_second"],
                summary["avg_write_ms"],
                summary["max_write_ms"],
                s["skipped"],
                s["failed"],
            )

    @staticmethod
    def _empty_stats() -> dict:
        return {"writes": 0, "total_ms": 0.0, "max_ms": 0.0, "skipped": 0, "failed": 0}

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

    def locked_tracks(self, session_id: str) -> dict[int, dict]:
        """Locks the user made in the UI: track_id -> row (bbox, class, conf).

        A negative track_id is a manual lock: the user clicked somewhere in the
        picture where nothing was detected, and the UI wrote the click box
        itself. Both kinds are handled identically from here on.
        """
        try:
            rows = (
                self.client.table("atlas_detections")
                .select("track_id,bbox,object_class,confidence")
                .eq("flight_session_id", session_id)
                .eq("is_locked", True)
                .execute()
                .data
                or []
            )
            return {int(r["track_id"]): r for r in rows}
        except Exception as exc:
            log.warning("Locked-track lookup failed: %s", exc)
            return {}

    def clear_lock(self, session_id: str, track_id: int) -> None:
        """Release a lock the detector can no longer follow."""
        try:
            table = self.client.table("atlas_detections")
            if track_id < 0:
                # Manual click lock: nothing else owns the row, so drop it.
                table.delete().eq("flight_session_id", session_id).eq(
                    "track_id", track_id
                ).execute()
            else:
                table.update({"is_locked": False}).eq(
                    "flight_session_id", session_id
                ).eq("track_id", track_id).execute()
        except Exception as exc:
            log.warning("Lock release failed: %s", exc)

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
        through /atlas-video-endpoint. EO (sensor 1) wins when the
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

        try:
            sensors = (
                self.client.table("atlas_drone_sensors")
                .select("drone_id, serial, sensor")
                .in_("drone_id", drone_ids)
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


class RealtimeSocket:
    """One persistent Supabase Realtime websocket shared by every stream.

    Speaks the Phoenix channel protocol Realtime uses: join
    `realtime:<topic>` as a private channel with the service-role key, push
    `broadcast` events, heartbeat every 25 s. Channels are joined on first use
    and left after a minute without snapshots. Broadcast acks are requested so
    delivery is confirmed and its round trip measured, but never awaited by the
    sender. While the socket is down `submit` returns False and the caller
    falls back to the REST endpoint.
    """

    HEARTBEAT_SECONDS = 25.0
    IDLE_LEAVE_SECONDS = 60.0

    def __init__(self, broadcaster) -> None:
        self._broadcaster = broadcaster
        base = SUPABASE_URL.rstrip("/")
        base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        self._url = f"{base}/realtime/v1/websocket?apikey={SUPABASE_SERVICE_ROLE_KEY}&vsn=1.0.0"
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._loop = asyncio.new_event_loop()
        self._wake: asyncio.Event | None = None
        self.connected = False
        threading.Thread(target=self._thread, daemon=True, name="realtime-ws").start()

    # -- called from worker threads ------------------------------------------ #

    def submit(self, session_id: str, payload: dict) -> tuple[bool, bool]:
        """Queue a snapshot. Returns (accepted, replaced_an_unsent_one)."""
        if not self.connected or self._wake is None:
            return False, False
        with self._lock:
            skipped = session_id in self._pending
            self._pending[session_id] = payload
        self._loop.call_soon_threadsafe(self._wake.set)
        return True, skipped

    # -- event loop ------------------------------------------------------------ #

    def _thread(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._run_forever())

    async def _run_forever(self) -> None:
        import websockets

        self._wake = asyncio.Event()
        backoff = BACKOFF_MIN
        while True:
            try:
                async with websockets.connect(
                    self._url, ping_interval=None, open_timeout=10, max_size=None
                ) as ws:
                    backoff = BACKOFF_MIN
                    await self._session(ws)
            except Exception as exc:
                # The URL carries the service-role key: never let it reach a log.
                reason = str(exc).replace(SUPABASE_SERVICE_ROLE_KEY, "***") if SUPABASE_SERVICE_ROLE_KEY else str(exc)
                log.warning("Realtime websocket lost (%s); using REST until it is back", reason)
            finally:
                self.connected = False
            await asyncio.sleep(backoff)
            backoff = min(BACKOFF_MAX, backoff * 2)

    async def _session(self, ws) -> None:
        refs = iter(range(1, 1 << 62))
        joined: dict[str, float] = {}           # topic -> last snapshot time
        join_waiters: dict[str, asyncio.Future] = {}
        sent_at: dict[str, float] = {}          # broadcast ref -> perf_counter

        async def reader() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                event, ref, topic = msg.get("event"), msg.get("ref"), msg.get("topic")
                if event == "phx_reply":
                    ok = (msg.get("payload") or {}).get("status") == "ok"
                    if ref in join_waiters:
                        join_waiters.pop(ref).set_result(msg.get("payload"))
                    elif ref in sent_at:
                        ms = (time.perf_counter() - sent_at.pop(ref)) * 1000.0
                        error = None if ok else f"broadcast rejected: {msg.get('payload')}"
                        self._broadcaster.record("ws", ms, error)
                elif event in ("phx_error", "phx_close") and topic in joined:
                    joined.pop(topic, None)  # rejoin on next snapshot

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(self.HEARTBEAT_SECONDS)
                await ws.send(json.dumps(
                    {"topic": "phoenix", "event": "heartbeat", "payload": {}, "ref": str(next(refs))}
                ))

        async def join(topic: str) -> None:
            ref = str(next(refs))
            waiter = asyncio.get_running_loop().create_future()
            join_waiters[ref] = waiter
            await ws.send(json.dumps({
                "topic": topic, "event": "phx_join", "ref": ref, "join_ref": ref,
                "payload": {
                    "config": {
                        "broadcast": {"ack": True, "self": False},
                        "presence": {"key": ""},
                        "private": True,
                    },
                    "access_token": SUPABASE_SERVICE_ROLE_KEY,
                },
            }))
            reply = await asyncio.wait_for(waiter, timeout=10)
            if (reply or {}).get("status") != "ok":
                raise RuntimeError(f"join {topic} refused: {reply}")

        tasks = [asyncio.ensure_future(reader()), asyncio.ensure_future(heartbeat())]
        self.connected = True
        log.info("Realtime websocket connected")
        try:
            while True:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                for task in tasks:
                    if task.done():
                        raise task.exception() or ConnectionError("socket closed")
                with self._lock:
                    batch = list(self._pending.items())
                    self._pending.clear()
                now = time.time()
                for session_id, payload in batch:
                    topic = f"realtime:{Broadcaster.topic(session_id)}"
                    if topic not in joined:
                        await join(topic)
                    joined[topic] = now
                    ref = str(next(refs))
                    sent_at[ref] = time.perf_counter()
                    await ws.send(json.dumps({
                        "topic": topic, "event": "broadcast", "ref": ref, "join_ref": ref,
                        "payload": {"type": "broadcast", "event": BROADCAST_EVENT, "payload": payload},
                    }))
                # Housekeeping: leave idle channels, forget acks that never came.
                for topic in [t for t, last in joined.items() if now - last > self.IDLE_LEAVE_SECONDS]:
                    joined.pop(topic, None)
                    await ws.send(json.dumps(
                        {"topic": topic, "event": "phx_leave", "payload": {}, "ref": str(next(refs))}
                    ))
                stale = time.perf_counter() - 10.0
                for ref in [r for r, t in sent_at.items() if t < stale]:
                    sent_at.pop(ref, None)
                    self._broadcaster.record("ws", None, "broadcast not acknowledged within 10 s")
        finally:
            self.connected = False
            for task in tasks:
                task.cancel()


class Broadcaster:
    """Pushes one Realtime Broadcast snapshot per frame to a private channel.

    Postgres Changes turns every changed row into a Realtime message for every
    viewer, after a database write, WAL decoding and an RLS check. A snapshot
    is one message per frame no matter how many tracks it holds, and never
    touches the database.

    Primary path: one persistent Realtime websocket (RealtimeSocket), ~10 ms
    per message from Fly. Fallback while the socket is down: the Realtime REST
    endpoint (~110 ms per request), through one lane (own HTTP client) per
    stream slot. Both use the service-role key, which may publish to private
    channels; viewers are authorised by the RLS policy on realtime.messages.
    Newest snapshot wins everywhere: a slow send is skipped, never queued.
    """

    def __init__(self) -> None:
        import httpx

        self._url = f"{SUPABASE_URL.rstrip('/')}/realtime/v1/api/broadcast"
        self._headers = {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json",
        }
        self._stats_lock = threading.Lock()
        self._stats = self._empty_stats()
        self._last_error: str | None = None
        self._lane_of: dict[str, dict] = {}
        self._next_lane = 0
        self._lanes = []
        for idx in range(max(1, DB_WRITER_THREADS)):
            lane = {
                "client": httpx.Client(timeout=5.0),
                "pending": {},
                "lock": threading.Lock(),
                "event": threading.Event(),
            }
            self._lanes.append(lane)
            threading.Thread(
                target=self._sender_loop, args=(lane, idx == 0), daemon=True,
                name=f"broadcast-{idx}",
            ).start()
        self._socket = RealtimeSocket(self)

    @staticmethod
    def topic(session_id: str) -> str:
        return f"{BROADCAST_TOPIC_PREFIX}{session_id}"

    def publish(self, session_id: str, tracks: list[dict]) -> None:
        """Queue a snapshot (the complete track list) for one flight."""
        payload = {
            "v": 1,
            "flight_session_id": session_id,
            "sent_at": int(time.time() * 1000),
            "tracks": tracks,
        }
        accepted, skipped = self._socket.submit(session_id, payload)
        if not accepted:
            lane = self._lane_for(session_id)
            with lane["lock"]:
                skipped = session_id in lane["pending"]
                lane["pending"][session_id] = payload
            lane["event"].set()
        if skipped:
            with self._stats_lock:
                self._stats["skipped"] += 1

    def _lane_for(self, key: str) -> dict:
        with self._stats_lock:
            lane = self._lane_of.get(key)
            if lane is None:
                lane = self._lanes[self._next_lane % len(self._lanes)]
                self._next_lane += 1
                self._lane_of[key] = lane
            return lane

    def record(self, via: str, ms: float | None = None, error: str | None = None) -> None:
        """Account one sent message (ms = delivery round trip when known)."""
        with self._stats_lock:
            self._stats["sent_" + via] += 1
            if ms is not None:
                self._stats["timed"] += 1
                self._stats["total_ms"] += ms
                self._stats["max_ms"] = max(self._stats["max_ms"], ms)
            if error:
                self._stats["failed"] += 1
                first = self._last_error is None
                self._last_error = error
        if error and first:
            # Logged once; afterwards the count shows up in the summary line.
            log.warning("Broadcast failed: %s (further failures are counted)", error)

    def _sender_loop(self, lane: dict, reports: bool) -> None:
        window_start = time.time()
        while True:
            lane["event"].wait(0.1)
            lane["event"].clear()
            with lane["lock"]:
                payloads = list(lane["pending"].values())
                lane["pending"].clear()
            for payload in payloads:
                self._send(lane["client"], payload)
            if reports:
                now = time.time()
                if now - window_start >= LOG_SUMMARY_SECONDS:
                    self._report(now - window_start)
                    window_start = now

    def _post(self, client, body: dict) -> int:
        return client.post(self._url, headers=self._headers, json=body).status_code

    def _send(self, client, payload: dict) -> None:
        body = {
            "messages": [
                {
                    "topic": self.topic(payload["flight_session_id"]),
                    "event": BROADCAST_EVENT,
                    "payload": payload,
                    "private": True,
                }
            ]
        }
        started = time.perf_counter()
        error = None
        try:
            code = self._post(client, body)
            if code >= 300:
                error = f"HTTP {code}"
        except Exception as exc:  # never let a send error kill the loop
            error = str(exc)
        self.record("rest", (time.perf_counter() - started) * 1000.0, error)

    def _report(self, window: float) -> None:
        with self._stats_lock:
            s = self._stats
            self._stats = self._empty_stats()
            last_error = self._last_error
            if not s["failed"]:
                self._last_error = None
        sent = s["sent_ws"] + s["sent_rest"]
        timed = s["timed"]
        summary = {
            "messages_per_second": round(sent / window, 1),
            "via_websocket": s["sent_ws"],
            "via_rest": s["sent_rest"],
            "avg_ack_ms": round(s["total_ms"] / timed, 1) if timed else None,
            "max_ack_ms": round(s["max_ms"], 1) if timed else None,
            "skipped": s["skipped"],
            "failed": s["failed"],
            "last_error": last_error if s["failed"] else None,
            "websocket_connected": self._socket.connected,
        }
        status.broadcast = summary
        if sent or s["skipped"] or s["failed"]:
            log.info(
                "broadcast: %.1f msg/s (%d websocket, %d rest), ack avg %s ms, "
                "max %s ms, %d skipped, %d failed%s",
                summary["messages_per_second"],
                s["sent_ws"],
                s["sent_rest"],
                summary["avg_ack_ms"],
                summary["max_ack_ms"],
                s["skipped"],
                s["failed"],
                f" ({last_error})" if s["failed"] else "",
            )

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "sent_ws": 0, "sent_rest": 0, "timed": 0, "total_ms": 0.0,
            "max_ms": 0.0, "skipped": 0, "failed": 0,
        }


class VelocityEstimator:
    """Smoothed per-track velocity in normalised units per second.

    Sent with every snapshot so the frontend can glide boxes between
    snapshots instead of letting them jump at 10 Hz.
    """

    def __init__(self, alpha: float = 0.5, max_gap: float = 1.0) -> None:
        self._alpha = alpha
        self._max_gap = max_gap
        self._state: dict[int, tuple[float, float, float, float, float]] = {}

    def update(self, track_id: int, cx: float, cy: float, now: float) -> tuple[float, float]:
        prev = self._state.get(track_id)
        vx = vy = 0.0
        if prev is not None:
            px, py, pt, pvx, pvy = prev
            dt = now - pt
            if 0.0 < dt <= self._max_gap:
                a = self._alpha
                vx = a * (cx - px) / dt + (1.0 - a) * pvx
                vy = a * (cy - py) / dt + (1.0 - a) * pvy
        self._state[track_id] = (cx, cy, now, vx, vy)
        return vx, vy

    def forget_older_than(self, now: float, seconds: float = 2.0) -> None:
        for tid in [t for t, st in self._state.items() if now - st[2] > seconds]:
            self._state.pop(tid, None)


def snapshot_tracks(rows: list[dict], locked_ids, velocity: VelocityEstimator, now: float) -> list[dict]:
    """atlas_detections rows -> the compact track list of a broadcast snapshot."""
    tracks = []
    for row in rows:
        box = row["bbox"]
        tid = int(row["track_id"])
        vx, vy = velocity.update(
            tid, box["x"] + box["width"] / 2.0, box["y"] + box["height"] / 2.0, now
        )
        tracks.append(
            {
                "id": tid,
                "cls": row["object_class"],
                "conf": row["confidence"],
                "x": box["x"],
                "y": box["y"],
                "w": box["width"],
                "h": box["height"],
                "vx": round(vx, 4),
                "vy": round(vy, 4),
                "locked": tid in locked_ids,
            }
        )
    velocity.forget_older_than(now)
    return tracks


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
        # A Condition instead of a plain Lock so the worker wakes the moment a
        # frame is decoded instead of polling (polling added up to 20 ms).
        self._lock = threading.Condition()
        self._frame = None
        self._seq = 0
        self._grabbed_at = 0.0
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
                        self._lock.notify_all()
                    return
                time.sleep(0.05)
                continue
            empty_reads = 0
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._grabbed_at = time.time()
                self._lock.notify_all()

    def wait_newer(self, seq: int, timeout: float):
        """Block until a frame newer than `seq` exists (or an error/timeout).

        Returns (frame, seq, error, grabbed_at) — the caller detects a timeout
        by getting its own `seq` back.
        """
        with self._lock:
            self._lock.wait_for(
                lambda: self._seq != seq or self._error is not None, timeout
            )
            return self._frame, self._seq, self._error, self._grabbed_at

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


IDENTITY_WARP = np.eye(2, 3, dtype=np.float32)


def compose_warp(later: np.ndarray, earlier: np.ndarray) -> np.ndarray:
    """2x3 affine for "earlier, then later"."""
    a = np.vstack([later, [0.0, 0.0, 1.0]])
    b = np.vstack([earlier, [0.0, 0.0, 1.0]])
    return (a @ b)[:2].astype(np.float32)


def warp_boxes(xyxy: np.ndarray, warp: np.ndarray) -> np.ndarray:
    """Move xyxy boxes by a (similarity) affine; axis-aligned result."""
    if len(xyxy) == 0:
        return xyxy
    pts = np.concatenate([xyxy[:, :2], xyxy[:, 2:]], axis=0)
    moved = pts @ warp[:, :2].T + warp[:, 2]
    n = len(xyxy)
    p1, p2 = moved[:n], moved[n:]
    return np.concatenate([np.minimum(p1, p2), np.maximum(p1, p2)], axis=1).astype(np.float32)


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


def with_sources(detections, sources: list[str], past: list[bool] | None = None):
    """Tag every box with where it came from (fast / range_new / range_old /
    crop_new / crop_old / motion_new / motion_old) and whether it was computed
    on an earlier frame; survives dedupe and filtering (sv.Detections slices
    data arrays)."""
    detections.data = dict(detections.data or {})
    detections.data["src"] = np.array(sources, dtype=object)
    detections.data["past"] = np.array(past if past is not None else [False] * len(sources), dtype=bool)
    return detections


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
    # Re-fed copies of an older range/motion result describe where an object
    # WAS; on the same object a fresh detection must win whatever its score.
    src = (detections.data or {}).get("src")
    stale = (
        np.array([1 if str(x).endswith("_old") else 0 for x in src])
        if src is not None and len(src) == len(labels)
        else np.zeros(len(labels), dtype=int)
    )
    order = np.lexsort((-np.asarray(confidence, dtype=float), stale, unknown))
    keep: list[int] = []
    boxes = detections.xyxy
    for idx in order:
        duplicate = False
        for kept in keep:
            # Same class group = same object candidate. `unknown` is compared
            # against every class, so a motion box on an object YOLO also found
            # is suppressed instead of becoming a second box. Confusable classes
            # (car/truck, person/bicycle, ...) share a group, so the same object
            # cannot keep one box per guess.
            if (
                class_group(labels[kept]) != class_group(labels[idx])
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
        self._scan_seq = -1
        self._result_seq = -1
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

    def latest(self) -> tuple[sv.Detections | None, float, int]:
        """(detections, finished at, seq of the frame they describe)."""
        with self._lock:
            return self._detections, self._updated_at, self._result_seq

    # -- internals ---------------------------------------------------------- #

    def _run(self) -> None:
        log.info("[%s] motion pass: %.1f analyses/s", self._path, MOTION_FPS)
        while not self._stop.is_set():
            frame, self._scan_seq = self._frame_source()
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
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0
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
            self._result_seq = self._scan_seq
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
        self._frame_source = frame_source  # callable -> (full-res frame or None, seq)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._detections: sv.Detections | None = None
        self._labels: list[str] = []
        self._updated_at: float = 0.0
        self._scan_seq = -1
        self._result_seq = -1
        self.last_scan_ms: float = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"range-{path}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[sv.Detections | None, list[str], float, int]:
        """(detections, labels, finished at, seq of the frame they describe)."""
        with self._lock:
            return self._detections, list(self._labels), self._updated_at, self._result_seq

    def _run(self) -> None:
        log.info(
            "[%s] range pass: %dx%d tiles every %.1fs",
            self._path,
            RANGE_TILE_COLS,
            RANGE_TILE_ROWS,
            RANGE_PASS_INTERVAL_SECONDS,
        )
        while not self._stop.is_set():
            frame, self._scan_seq = self._frame_source()
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
        scan_started = time.perf_counter()
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
            detections, labels = self._detector.detect(
                tile, conf=RANGE_CONFIDENCE, priority=PRIORITY_RANGE
            )
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
            self._result_seq = self._scan_seq
        self.last_scan_ms = (time.perf_counter() - scan_started) * 1000.0
        if RANGE_DEBUG:
            log.info(
                "[%s] range pass: %d object(s) in full frame", self._path, len(merged)
            )


class CropScanner:
    """Background thread that re-finds small tracks at native resolution.

    `targets_source()` returns the full-resolution boxes of the current small
    tracks (where the tracker expects them). A few times per second a
    CROP_SIZE window is cut around each (nearby targets share a window, at
    most MAX_CROPS), all windows go through the model in one batched call at
    their own size, and the boxes come back in full-resolution coordinates.
    """

    def __init__(self, path: str, detector, frame_source, targets_source) -> None:
        self._path = path
        self._detector = detector
        self._frame_source = frame_source
        self._targets_source = targets_source
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._detections: sv.Detections | None = None
        self._labels: list[str] = []
        self._updated_at: float = 0.0
        self._scan_seq = -1
        self._result_seq = -1
        self.last_scan_ms: float = 0.0
        self.last_crops: int = 0
        self._interval = 1.0 / CROP_FPS if CROP_FPS > 0 else 0.25
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"crop-{path}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> tuple[sv.Detections | None, list[str], float, int]:
        """(detections, labels, finished at, seq of the frame they describe)."""
        with self._lock:
            return self._detections, list(self._labels), self._updated_at, self._result_seq

    def _run(self) -> None:
        log.info(
            "[%s] crop pass: up to %d x %dpx windows %.0f/s around small tracks",
            self._path, MAX_CROPS, CROP_SIZE, CROP_FPS,
        )
        while not self._stop.is_set():
            started = time.time()
            frame, self._scan_seq = self._frame_source()
            targets = self._targets_source()
            if frame is not None and targets is not None and len(targets):
                try:
                    self._scan(frame, targets)
                except Exception as exc:
                    log.warning("[%s] crop pass error: %s", self._path, exc)
            else:
                self.last_crops = 0
            remaining = self._interval - (time.time() - started)
            if self._stop.wait(max(0.02, remaining)):
                break
        log.info("[%s] crop pass stopped", self._path)

    @staticmethod
    def windows(targets: np.ndarray, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """CROP_SIZE windows covering the targets; a target whose centre sits
        in the inner part of an earlier window shares it."""
        side_w, side_h = min(CROP_SIZE, width), min(CROP_SIZE, height)
        out: list[tuple[int, int, int, int]] = []
        for x1, y1, x2, y2 in targets:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if any(
                wx0 + side_w * 0.2 <= cx <= wx1 - side_w * 0.2
                and wy0 + side_h * 0.2 <= cy <= wy1 - side_h * 0.2
                for wx0, wy0, wx1, wy1 in out
            ):
                continue
            x0 = int(min(max(0, cx - side_w / 2), width - side_w))
            y0 = int(min(max(0, cy - side_h / 2), height - side_h))
            out.append((x0, y0, x0 + side_w, y0 + side_h))
            if len(out) >= MAX_CROPS:
                break
        return out

    def _scan(self, frame, targets: np.ndarray) -> None:
        started = time.perf_counter()
        height, width = frame.shape[:2]
        wins = self.windows(np.asarray(targets, dtype=float).reshape(-1, 4), width, height)
        crops = [frame[y0:y1, x0:x1] for x0, y0, x1, y1 in wins]
        results = self._detector.detect_batch(
            crops, conf=CROP_CONFIDENCE, imgsz=CROP_SIZE, priority=PRIORITY_CROP
        )
        all_xyxy: list[np.ndarray] = []
        all_conf: list[np.ndarray] = []
        all_labels: list[str] = []
        margin = 2.0
        for (x0, y0, x1, y1), (dets, labels) in zip(wins, results):
            if len(dets) == 0:
                continue
            conf = dets.confidence if dets.confidence is not None else np.zeros(len(dets))
            keep = []
            for i, (bx1, by1, bx2, by2) in enumerate(dets.xyxy):
                # A box cut by a window edge that is not a frame edge is a
                # fragment; the full object is (or will be) seen elsewhere.
                cut = (
                    (bx1 <= margin and x0 > 0)
                    or (by1 <= margin and y0 > 0)
                    or (bx2 >= (x1 - x0) - margin and x1 < width)
                    or (by2 >= (y1 - y0) - margin and y1 < height)
                )
                if not cut:
                    keep.append(i)
            if not keep:
                continue
            boxes = dets.xyxy[keep].copy()
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0
            all_xyxy.append(boxes)
            all_conf.append(conf[keep])
            all_labels.extend(labels[i] for i in keep)
        if all_xyxy:
            merged = sv.Detections(xyxy=np.concatenate(all_xyxy), confidence=np.concatenate(all_conf))
            merged, all_labels = dedupe_class_aware(merged, all_labels)
        else:
            merged, all_labels = sv.Detections.empty(), []
        with self._lock:
            self._detections = merged
            self._labels = all_labels
            self._updated_at = time.time()
            self._result_seq = self._scan_seq
        self.last_crops = len(wins)
        self.last_scan_ms = (time.perf_counter() - started) * 1000.0


# --------------------------------------------------------------------------- #
# Locked objects (pixel tracking)
# --------------------------------------------------------------------------- #


def _create_cv_tracker():
    """Best available OpenCV pixel tracker (CSRT preferred, KCF as fallback)."""
    factories = (
        getattr(cv2, "TrackerCSRT_create", None),
        getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None),
        getattr(cv2, "TrackerKCF_create", None),
        getattr(getattr(cv2, "legacy", None), "TrackerKCF_create", None),
        # Last resort: always present, less accurate, but keeps a lock alive.
        getattr(cv2, "TrackerMIL_create", None),
    )
    for factory in factories:
        if factory is None:
            continue
        try:
            return factory()
        except Exception:  # pragma: no cover - build without contrib modules
            continue
    return None


class LockTracker:
    """Follows ONE locked object by its pixels, independent of the detector.

    The detector loses and re-finds objects constantly, and every re-find risks
    a new track id — which is exactly how a lock used to jump off the object
    when the camera moved. A locked object is therefore followed by an OpenCV
    pixel tracker and published on its own, outside ByteTrack: one lock, one
    box, one id, for as long as the user keeps it.

    Whenever the detector produces a box that clearly overlaps the lock, that
    box wins: it re-centres the pixel tracker (removing drift) and names the
    class, so a manual click on "some pixels" becomes "person 84%" by itself.
    """

    def __init__(self, track_id: int, frame, box, label: str, confidence: float):
        self.track_id = int(track_id)
        self.label = (label or UNKNOWN_CLASS).strip().lower() or UNKNOWN_CLASS
        self.confidence = float(confidence or 0.0)
        self.box = np.asarray(box, dtype=float)
        self.last_seen = time.time()
        self.impl = None
        self._init_impl(frame)

    def _rect(self, frame) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in self.box)
        x1 = max(0.0, min(w - 2.0, x1))
        y1 = max(0.0, min(h - 2.0, y1))
        x2 = max(x1 + 2.0, min(float(w), x2))
        y2 = max(y1 + 2.0, min(float(h), y2))
        # Integers: some OpenCV builds reject float bounding boxes in init().
        return (int(x1), int(y1), int(max(2.0, x2 - x1)), int(max(2.0, y2 - y1)))

    def _init_impl(self, frame) -> None:
        impl = _create_cv_tracker()
        if impl is None:
            return
        try:
            impl.init(frame, self._rect(frame))
            self.impl = impl
        except Exception as exc:
            log.warning("Lock %s: pixel tracker init failed: %s", self.track_id, exc)
            self.impl = None

    def update(self, frame) -> bool:
        """Advance the box one frame. False = the pixels were lost."""
        if self.impl is None:
            return False
        try:
            ok, rect = self.impl.update(frame)
        except Exception:
            return False
        if not ok:
            return False
        x, y, w, h = (float(v) for v in rect)
        if w < 2 or h < 2:
            return False
        self.box = np.array([x, y, x + w, y + h], dtype=float)
        self.last_seen = time.time()
        return True

    def correct(self, frame, box, label: str, confidence: float) -> None:
        """Take class/confidence from a matching detector box; re-anchor the
        pixel tracker on it only when the two have drifted apart."""
        box = np.asarray(box, dtype=float)
        self.last_seen = time.time()
        if label:
            self.label = label
        self.confidence = float(confidence or 0.0)
        drift_iou = float(iou_matrix(self.box.reshape(1, 4), box.reshape(1, 4))[0, 0])
        if self.impl is not None and drift_iou >= LOCK_REINIT_IOU:
            return
        self.box = box
        self._init_impl(frame)


def lock_row(
    session_id: str,
    track_id: int,
    box,
    width: int,
    height: int,
    label: str,
    confidence: float,
) -> dict:
    """Row for a locked object (same shape as build_rows, exactly one box)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    nx = max(0.0, min(1.0, x1 / width))
    ny = max(0.0, min(1.0, y1 / height))
    nw = max(0.0, min(1.0 - nx, (x2 - x1) / width))
    nh = max(0.0, min(1.0 - ny, (y2 - y1) / height))
    return {
        "flight_session_id": session_id,
        "track_id": int(track_id),
        "object_class": label or UNKNOWN_CLASS,
        "confidence": round(float(confidence or 0.0), 4),
        "bbox": {
            "x": round(nx, 5),
            "y": round(ny, 5),
            "width": round(nw, 5),
            "height": round(nh, 5),
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# One worker per live stream
# --------------------------------------------------------------------------- #


class _TrackerInput:
    """The minimal Results-like view Ultralytics trackers read (boolean
    indexing plus xyxy / xywh / conf / cls)."""

    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        x1, y1, x2, y2 = self.xyxy.T
        self.xywh = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_TrackerInput":
        return _TrackerInput(self.xyxy[idx], self.conf[idx], self.cls[idx])


class MaskedFlowGMC:
    """Camera motion between consecutive frames, from the background only.

    Ultralytics' sparseOptFlow picks corners anywhere — including on a large
    moving object — and then reports that object's motion as camera motion,
    which shifts every track the wrong way (measured: static camera, one big
    moving person -> 12-24 px phantom pans and ~5x more track ids). Here
    detected boxes (padded) and the frame border are masked out before corners
    are chosen, and when too few background points agree (open sea, fog,
    blur) no compensation is applied at all rather than a bad one.

    Interface matches what Ultralytics' BYTETracker expects from `self.gmc`.
    """

    method = "maskedFlow"
    MIN_POINTS = 12
    MIN_INLIER_RATIO = 0.35
    # Real camera motion moves the whole scene: the agreeing points must span
    # at least this fraction of the frame in both directions. A large moving
    # object (or a moving billboard) agrees with itself in one region only.
    MIN_SPREAD = 0.45
    # A drone camera does not zoom or roll more than this between two analysed
    # frames; a larger fitted scale/rotation means a few moving points were
    # folded into the model (static points near the origin, moving ones far
    # away fit a small "zoom" plus shift).
    MAX_SCALE_STEP = 0.02
    MAX_ROTATION_STEP = 0.035  # radians, ~2 degrees
    # When the fitted model moves the scene but a good share of the tracked
    # points did not move at all, the camera is still and something large is
    # moving (billboard, trailer, waves): apply no compensation.
    STATIC_FLOW_PX = 0.25
    STATIC_SHARE = 0.2

    # 640x360 fast frame -> 160x90: ~3 ms per frame worst case, mean error
    # ~0.1 px on a synthetic pan (2x/400 corners cost ~9-23 ms for 0.05 px).
    def __init__(self, downscale: int = 4) -> None:
        self.downscale = max(1, downscale)
        self._prev = None
        self._prev_pts = None
        self.last_ok = False
        self.last_H = IDENTITY_WARP

    def reset_params(self) -> None:
        self._prev = None
        self._prev_pts = None

    def _mask(self, shape, detections) -> np.ndarray:
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(0.03 * h): int(0.97 * h), int(0.03 * w): int(0.97 * w)] = 255
        if detections is not None:
            for det in np.asarray(detections, dtype=float).reshape(-1, 4):
                x1, y1, x2, y2 = det / self.downscale
                pw, ph = (x2 - x1) * 0.15, (y2 - y1) * 0.15
                mask[
                    max(0, int(y1 - ph)): max(0, int(y2 + ph)),
                    max(0, int(x1 - pw)): max(0, int(x2 + pw)),
                ] = 0
        return mask

    def apply(self, raw_frame, detections=None) -> np.ndarray:
        H = np.eye(2, 3, dtype=np.float32)
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            h0, w0 = gray.shape
            gray = cv2.resize(gray, (w0 // self.downscale, h0 // self.downscale))
        self.last_ok = False
        if self._prev is not None and self._prev_pts is not None and len(self._prev_pts) >= self.MIN_POINTS:
            curr, status, _ = cv2.calcOpticalFlowPyrLK(self._prev, gray, self._prev_pts, None)
            good = status.reshape(-1) == 1
            if int(good.sum()) >= self.MIN_POINTS:
                M, inliers = cv2.estimateAffinePartial2D(
                    self._prev_pts[good], curr[good], method=cv2.RANSAC, ransacReprojThreshold=1.0
                )
                accept = False
                if M is not None and inliers is not None and inliers.any():
                    pts = curr[good][inliers.reshape(-1) == 1].reshape(-1, 2)
                    h1, w1 = gray.shape
                    lo, hi = np.percentile(pts, [5, 95], axis=0)  # one stray point cannot fake a spread
                    accept = hi[0] - lo[0] >= self.MIN_SPREAD * w1 and hi[1] - lo[1] >= self.MIN_SPREAD * h1
                if accept:
                    scale = float(np.hypot(M[0, 0], M[1, 0]))
                    angle = float(np.arctan2(M[1, 0], M[0, 0]))
                    accept = abs(scale - 1.0) <= self.MAX_SCALE_STEP and abs(angle) <= self.MAX_ROTATION_STEP
                if accept:
                    flow = (curr[good] - self._prev_pts[good]).reshape(-1, 2)
                    static = int((np.abs(flow).max(axis=1) < self.STATIC_FLOW_PX).sum())
                    moves = float(np.hypot(M[0, 2], M[1, 2])) > 1.0 or abs(scale - 1.0) > 0.005
                    if moves and static >= max(4, self.STATIC_SHARE * int(good.sum())):
                        accept = False
                if (
                    accept
                    and int(inliers.sum()) >= max(self.MIN_POINTS, self.MIN_INLIER_RATIO * int(good.sum()))
                ):
                    H = M.astype(np.float32)
                    H[:, 2] *= self.downscale
                    self.last_ok = True
        self.last_H = H
        self._prev = gray
        self._prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=150, qualityLevel=0.01, minDistance=3,
            mask=self._mask(gray.shape, detections),
        )
        return H


class _FixedWarp:
    """Stands in for BoT-SORT's GMC for one update and returns a camera
    motion that was already measured (see StreamTracker.update)."""

    method = "precomputed"

    def __init__(self, warp: np.ndarray) -> None:
        self.last_H = warp

    def apply(self, raw_frame, detections=None) -> np.ndarray:
        return self.last_H

    def reset_params(self) -> None:
        pass


def _botsort_class():
    from ultralytics.trackers.bot_sort import BOTSORT

    class StreamBOTSORT(BOTSORT):
        """BoT-SORT that never resets the track-id counter.

        Ultralytics keeps one class-level id counter and resets it whenever a
        tracker is constructed. With several streams (or one stream
        reconnecting) that would hand out ids a running stream still uses. Ids
        simply keep counting up for the life of the process instead.
        """

        @staticmethod
        def reset_id() -> None:
            return None

    return StreamBOTSORT


class StreamTracker:
    """One tracker per stream: detections in, tracked detections out.

    Output is an sv.Detections with tracker_id, the detection's own confidence
    and data["label"] holding the voted class — exactly what build_rows needs.
    """

    def __init__(self) -> None:
        self._votes: dict[int, dict] = {}
        self._label_ids: dict[str, int] = {}
        self._frame = 0
        self._last_step = IDENTITY_WARP
        self.impl = TRACKER_IMPL if TRACKER_IMPL in ("botsort", "bytetrack") else "botsort"
        if self.impl == "bytetrack":
            # Previous behaviour, kept as a rollback (TRACKER_IMPL=bytetrack).
            self._sv = sv.ByteTrack(
                lost_track_buffer=TRACKER_LOST_BUFFER,
                frame_rate=30,  # makes the buffer exactly TRACKER_LOST_BUFFER frames
                track_activation_threshold=min(
                    0.25,
                    DETECTION_CONFIDENCE,
                    MOTION_CONFIDENCE if MOTION_PASS_ENABLED else 1.0,
                ),
            )
            self._bot = None
            return
        from types import SimpleNamespace

        self._sv = None
        self._bot = _botsort_class()(
            SimpleNamespace(
                tracker_type="botsort",
                track_high_thresh=TRACK_HIGH_THRESH,
                track_low_thresh=TRACK_LOW_THRESH,
                new_track_thresh=NEW_TRACK_THRESH,
                track_buffer=TRACKER_LOST_BUFFER,  # frames, not scaled
                match_thresh=TRACK_MATCH_THRESH,
                fuse_score=True,
                gmc_method=(
                    None if TRACKER_GMC.lower() in ("none", "maskedflow") else TRACKER_GMC
                ),
                proximity_thresh=0.5,
                appearance_thresh=0.8,
                with_reid=False,
                model="auto",
            )
        )
        if TRACKER_GMC.lower() == "maskedflow":
            self._bot.gmc = MaskedFlowGMC()

    def update(self, detections, labels: list[str], frame):
        """Advance one analysed frame. `frame` is the image the boxes live in
        (needed for motion compensation)."""
        if self._bot is None:
            return self._sv.update_with_detections(attach_labels(detections, labels))

        self._frame += 1
        n = len(detections)
        conf = (
            np.asarray(detections.confidence, dtype=np.float32)
            if detections.confidence is not None and n
            else np.zeros(n, dtype=np.float32)
        )
        feed = conf.copy()
        if n:
            src = (detections.data or {}).get("src")
            src = list(src) if src is not None and len(src) == n else [""] * n
            unknown = np.array([label == UNKNOWN_CLASS for label in labels], dtype=bool)
            stale = np.array([str(x).endswith("_old") for x in src], dtype=bool)
            sides = np.maximum(
                detections.xyxy[:, 2] - detections.xyxy[:, 0],
                detections.xyxy[:, 3] - detections.xyxy[:, 1],
            )
            big_range = np.array([x == "range_new" for x in src], dtype=bool) & (
                sides > RANGE_NEW_TRACK_MAX_SIDE
            )
            stale = stale | big_range
            # Confirmed motion candidates carry a deliberately low confidence
            # for display; a fresh one may still open a track.
            bump = unknown & ~stale
            feed[bump] = np.maximum(feed[bump], NEW_TRACK_THRESH)
            # Re-fed copies may only extend a track (low band), never open one.
            feed[stale] = np.minimum(feed[stale], max(TRACK_LOW_THRESH + 0.01, TRACK_HIGH_THRESH - 0.01))
        cls = [self._label_ids.setdefault(label, len(self._label_ids)) for label in labels]
        xyxy = np.asarray(detections.xyxy, dtype=np.float32) if n else np.zeros((0, 4), np.float32)
        gmc = self._bot.gmc
        if getattr(gmc, "method", None) is not None and frame is not None:
            # Measure this frame's camera motion first (same input BoT-SORT
            # would use), so boxes computed on an earlier frame (background
            # passes, re-fed copies) can be moved into this frame before they
            # are matched — BoT-SORT moves its tracks by the same step. Without
            # it every re-fed box trails a panning camera by one frame, which
            # is more than a small box is wide.
            try:
                step = gmc.apply(frame, xyxy[feed >= TRACK_HIGH_THRESH])
            except Exception as exc:
                log.warning("camera motion estimate failed: %s", exc)
                step = IDENTITY_WARP
            past = (detections.data or {}).get("past") if n else None
            if past is not None and len(past) == n and np.any(past):
                xyxy = xyxy.copy()
                xyxy[past] = warp_boxes(xyxy[past], step)
            self._last_step = step
            self._bot.gmc = _FixedWarp(step)
            try:
                out = self._bot.update(_TrackerInput(xyxy, feed, cls), frame)
            finally:
                self._bot.gmc = gmc
        else:
            out = self._bot.update(_TrackerInput(xyxy, feed, cls), frame)

        if out is None or len(out) == 0:
            self._prune()
            return sv.Detections.empty()
        out = np.asarray(out, dtype=np.float64)
        idx = out[:, 7].astype(int)
        tids = out[:, 4].astype(int)
        voted = [self._vote(int(t), labels[i], float(conf[i])) for t, i in zip(tids, idx)]
        tracked = sv.Detections(
            xyxy=out[:, :4].astype(np.float32),
            confidence=conf[idx],
            tracker_id=tids,
        )
        self._prune()
        return attach_labels(tracked, voted)

    def small_targets(self, max_side: float, limit: int) -> np.ndarray:
        """Boxes (tracker coordinates) of small tracks worth a native-resolution
        look: confirmed tracks and ones lost only moments ago, freshest first."""
        if self._bot is None:
            return np.zeros((0, 4), dtype=np.float32)
        tracks = [t for t in self._bot.tracked_stracks if t.is_activated]
        tracks += list(self._bot.lost_stracks)
        boxes = []
        for t in sorted(tracks, key=lambda t: -t.frame_id):
            x1, y1, x2, y2 = (float(v) for v in t.xyxy)
            if max(x2 - x1, y2 - y1) <= max_side:
                boxes.append((x1, y1, x2, y2))
            if len(boxes) >= limit * 3:  # windows merge nearby targets
                break
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    def last_warp(self) -> np.ndarray:
        """Camera motion measured on the last update (identity if unknown)."""
        return self._last_step

    def _vote(self, track_id: int, label: str, confidence: float) -> str:
        entry = self._votes.get(track_id)
        if entry is None:
            entry = self._votes[track_id] = {
                "hist": collections.deque(maxlen=max(1, CLASS_VOTE_WINDOW)),
                "seen": self._frame,
            }
        entry["hist"].append((label, confidence))
        entry["seen"] = self._frame
        scores: dict[str, float] = {}
        for lab, c in entry["hist"]:
            if lab and lab != UNKNOWN_CLASS:
                scores[lab] = scores.get(lab, 0.0) + max(c, 0.01)
        return max(scores, key=scores.get) if scores else UNKNOWN_CLASS

    def _prune(self) -> None:
        if self._frame % 50:
            return
        stale = [t for t, e in self._votes.items() if self._frame - e["seen"] > 300]
        for t in stale:
            self._votes.pop(t, None)


class StreamStats:
    """Rolling per-stream latency counters, flushed every LOG_SUMMARY_SECONDS.

    frame_age = decode -> analysis start, queue = waiting for the shared model,
    infer = model time, loop = analysis start -> rows handed to the writer.
    """

    FIELDS = ("frame_age_ms", "wait_ms", "infer_ms", "loop_ms", "tracks")

    def __init__(self) -> None:
        self._reset(time.time())

    def _reset(self, now: float) -> None:
        self._started = now
        self._count = 0
        self._sums = dict.fromkeys(self.FIELDS, 0.0)
        self._max_loop = 0.0

    def add(self, **values: float) -> None:
        self._count += 1
        for key in self.FIELDS:
            self._sums[key] += values.get(key, 0.0)
        self._max_loop = max(self._max_loop, values.get("loop_ms", 0.0))

    def due(self) -> bool:
        return time.time() - self._started >= LOG_SUMMARY_SECONDS

    def flush(
        self,
        range_scan_ms: float | None,
        locks: int,
        crop_scan_ms: float | None = None,
        crops: int = 0,
    ) -> dict:
        now = time.time()
        n = max(1, self._count)
        metrics = {
            "fps": round(self._count / max(1e-6, now - self._started), 1),
            "avg_frame_age_ms": round(self._sums["frame_age_ms"] / n, 1),
            "avg_queue_ms": round(self._sums["wait_ms"] / n, 1),
            "avg_infer_ms": round(self._sums["infer_ms"] / n, 1),
            "avg_loop_ms": round(self._sums["loop_ms"] / n, 1),
            "max_loop_ms": round(self._max_loop, 1),
            "avg_tracks": round(self._sums["tracks"] / n, 1),
            "range_scan_ms": None if range_scan_ms is None else round(range_scan_ms),
            "crop_scan_ms": None if crop_scan_ms is None else round(crop_scan_ms),
            "crops": crops,
            "locks": locks,
            "window_seconds": round(now - self._started, 1),
        }
        self._reset(now)
        return metrics




class StreamWorker(threading.Thread):
    """Analyses a single RTSP stream until it is asked to stop."""

    def __init__(self, session_id: str, path: str, detector, store: DetectionStore) -> None:
        super().__init__(daemon=True, name=f"stream-{path}")
        self.session_id = session_id
        self.path = path
        self.detector = detector
        self.store = store
        self.url = RTSP_URL if RTSP_URL else rtsp_url_for(path)
        self._stop_event = threading.Event()  # not `_stop`: Thread uses that name internally (join)

    def stop(self) -> None:
        self._stop_event.set()

    # -- main loop ---------------------------------------------------------- #

    def run(self) -> None:
        log.info("[%s] worker started (%s)", self.path, self.session_id)
        status.update(self.session_id, path=self.path, connected=False, reconnects=0)
        backoff = BACKOFF_MIN
        reconnects = 0
        while not self._stop_event.is_set():
            try:
                self._run_once()
                log.warning("[%s] stream ended", self.path)
            except Exception as exc:
                log.warning("[%s] stream error: %s", self.path, exc)

            # Drop stale boxes immediately so the UI never shows frozen overlays.
            self.store.clear(self.session_id)
            self._broadcast_empty()
            reconnects += 1
            status.update(
                self.session_id, connected=False, active_tracks=0, reconnects=reconnects
            )
            if self._stop_event.wait(backoff):
                break
            backoff = min(BACKOFF_MAX, backoff * 2)

        self.store.clear(self.session_id)
        self._broadcast_empty()
        status.remove(self.session_id)
        log.info("[%s] worker stopped", self.path)

    def _broadcast_empty(self) -> None:
        """Tell viewers right away that this stream has no boxes any more."""
        if self.store.broadcaster is not None:
            self.store.broadcaster.publish(self.session_id, [])

    def _run_once(self) -> None:
        cap = open_capture(self.url)
        if cap is None:
            raise ConnectionError("could not open RTSP stream")

        status.update(self.session_id, connected=True)
        log.info("[%s] connected", self.path)

        # One tracker per connection (BoT-SORT with motion compensation by
        # default, see StreamTracker).
        tracker = StreamTracker()
        # Camera motion since the current range / motion result was produced,
        # used to move its re-fed copies along with the scene.
        motion_compensated = tracker.impl == "botsort"
        range_fed_at = motion_fed_at = -1.0
        range_warp = motion_warp = IDENTITY_WARP
        range_new_left = motion_new_left = 0
        crop_fed_at = -1.0
        crop_warp = IDENTITY_WARP
        crop_new_left = 0
        crop_targets: dict = {"boxes": None}
        crop_active = True
        queue_ema = 0.0
        grabber = FrameGrabber(cap)
        last_seq = 0
        last_inference = 0.0
        stats = StreamStats()
        velocity = VelocityEstimator()
        broadcaster = self.store.broadcaster
        last_broadcast = 0.0
        last_broadcast_empty = False
        # Token bucket for BROADCAST_MAX_HZ: frames arrive with jitter, and a
        # plain "at least 1/hz since the last send" check skipped every other
        # frame whenever a frame came a few ms early (10 fps gave ~5 msg/s).
        broadcast_budget = 1.0
        budget_at = 0.0

        # Latest full-resolution frame and its seq, shared with the background
        # scanners (one tuple, so frame and seq always belong together).
        full_frame_slot: dict = {"pair": (None, -1)}

        def latest_full_frame():
            return full_frame_slot["pair"]

        # Camera motion per analysed frame (seq, step), so a background result
        # can be moved from the frame it was computed on to the current one —
        # a range scan takes ~0.3 s, a crop scan ~0.05 s, and a panning camera
        # moves the scene several px per frame meanwhile.
        warp_log: collections.deque = collections.deque(maxlen=256)

        def warp_since(result_seq: int) -> np.ndarray:
            warp = IDENTITY_WARP
            for step_seq, step in warp_log:
                if step_seq > result_seq:
                    warp = compose_warp(step, warp)
            return warp

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
        # Needs the BoT-SORT tracker (small_targets reads its track states).
        cropper = (
            CropScanner(self.path, self.detector, latest_full_frame, lambda: crop_targets["boxes"])
            if CROP_PASS_ENABLED and tracker.impl == "botsort"
            else None
        )

        # Locked track: polled from Supabase (the UI owns the flag) plus the
        # last known normalised box per track, used to build the priority ROI.
        lock_trackers: dict[int, LockTracker] = {}
        last_locked_poll = 0.0

        try:
            while not self._stop_event.is_set():
                # Pace to DETECTION_FPS: sleep the exact remainder of the frame
                # interval, then take the newest frame the moment it exists.
                pause = MIN_FRAME_INTERVAL - (time.time() - last_inference)
                if pause > 0:
                    self._stop_event.wait(pause)
                    continue
                frame, seq, error, grabbed_at = grabber.wait_newer(last_seq, 0.5)
                if error:
                    raise ConnectionError(error)
                if frame is None or seq == last_seq:
                    continue

                now = time.time()
                last_seq = seq
                last_inference = now
                status.update(self.session_id, last_frame_at=now)

                src_h, src_w = frame.shape[:2]
                if not src_w or not src_h:
                    continue

                # Share the full-resolution frame with the range scanner before
                # downscaling — that is where the small/distant objects live.
                full_frame_slot["pair"] = (frame, seq)

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

                timing: dict = {}
                detections, labels = self.detector.detect(frame, timing=timing)
                infer_ms = timing.get("infer_ms", 0.0)
                raw_count = len(detections)

                # Merge long-range detections (full-res coords -> fast-frame
                # coords) and suppress duplicates ONCE, over the combined set,
                # so the tracker only ever sees one box per physical object.
                # Stale range results are skipped: their coordinates describe
                # where the object was, and feeding them spawns a ghost track.
                range_count = 0
                sources = ["fast"] * len(detections)
                # True for boxes computed on an earlier frame: the tracker moves
                # them by this frame's camera motion too (see StreamTracker).
                past = [False] * len(detections)
                if scanner is not None:
                    range_dets, range_labels, range_at, range_seq = scanner.latest()
                    fresh = (now - range_at) <= RANGE_RESULT_MAX_AGE_SECONDS
                    if range_dets is not None and len(range_dets) > 0 and fresh:
                        range_count = len(range_dets)
                        range_boxes = range_dets.xyxy * scale
                        if motion_compensated:
                            # A range result may open tracks only the first
                            # time it is fed. Later copies are moved along with
                            # the camera since then and may only keep an
                            # existing track alive (see StreamTracker).
                            if range_at != range_fed_at:
                                range_fed_at, range_warp = range_at, warp_since(range_seq)
                                range_new_left = NEW_RESULT_FRAMES
                            if range_new_left > 0:
                                # New tracks need two consecutive frames to be
                                # confirmed, so a result may open tracks on its
                                # first NEW_RESULT_FRAMES frames.
                                range_new_left -= 1
                                range_src = "range_new"
                            else:
                                range_src = "range_old"
                            if range_warp is not IDENTITY_WARP:
                                range_boxes = warp_boxes(range_boxes, range_warp)
                        else:
                            range_src = "range_new"
                        sources += [range_src] * range_count
                        past += [motion_compensated and range_seq != seq] * range_count
                        detections = sv.Detections(
                            xyxy=np.concatenate(
                                [detections.xyxy, range_boxes]
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

                # Crop pass: native-resolution looks at the small tracks. Same
                # freshness rules as the range pass (new result may open tracks
                # for NEW_RESULT_FRAMES frames, later copies follow the camera
                # and only extend).
                crop_count = 0
                if cropper is not None:
                    crop_dets, crop_labels, crop_at, crop_seq = cropper.latest()
                    fresh = (now - crop_at) <= CROP_RESULT_MAX_AGE_SECONDS
                    if crop_dets is not None and len(crop_dets) > 0 and fresh:
                        crop_count = len(crop_dets)
                        if crop_at != crop_fed_at:
                            crop_fed_at, crop_warp = crop_at, warp_since(crop_seq)
                            crop_new_left = NEW_RESULT_FRAMES
                        if crop_new_left > 0:
                            crop_new_left -= 1
                            crop_src = "crop_new"
                        else:
                            crop_src = "crop_old"
                        crop_view = crop_dets
                        if crop_warp is not IDENTITY_WARP:
                            crop_view = sv.Detections(
                                xyxy=warp_boxes(crop_dets.xyxy * scale, crop_warp) / scale,
                                confidence=crop_dets.confidence,
                            )
                        sources += [crop_src] * crop_count
                        past += [crop_seq != seq] * crop_count
                        detections, labels = merge_source(
                            detections, labels, crop_view, crop_labels, scale
                        )

                # Motion candidates: objects that move against the compensated
                # background but are not classifiable yet. Same coordinate
                # conversion, same dedupe, same tracker — a motion box that
                # lands on an object YOLO already found is suppressed, so one
                # object never gets two boxes.
                motion_count = 0
                if motion is not None:
                    motion_dets, motion_at, motion_seq = motion.latest()
                    fresh = (now - motion_at) <= MOTION_RESULT_MAX_AGE_SECONDS
                    if motion_dets is not None and len(motion_dets) > 0 and fresh:
                        motion_count = len(motion_dets)
                        motion_src = "motion_new"
                        if motion_compensated:
                            if motion_at != motion_fed_at:
                                motion_fed_at, motion_warp = motion_at, warp_since(motion_seq)
                                motion_new_left = NEW_RESULT_FRAMES
                            if motion_new_left > 0:
                                motion_new_left -= 1
                            else:
                                motion_src = "motion_old"
                            if motion_warp is not IDENTITY_WARP:
                                motion_dets = sv.Detections(
                                    xyxy=warp_boxes(motion_dets.xyxy * scale, motion_warp) / scale,
                                    confidence=motion_dets.confidence,
                                )
                        sources += [motion_src] * motion_count
                        past += [motion_compensated and motion_seq != seq] * motion_count
                        detections, labels = merge_source(
                            detections,
                            labels,
                            motion_dets,
                            [UNKNOWN_CLASS] * motion_count,
                            scale,
                        )

                merged_count = len(detections)
                detections = with_sources(detections, sources, past)
                detections, labels = dedupe_class_aware(detections, labels)

                # ---- Locked objects ---------------------------------------- #
                # Handled BEFORE the tracker and outside it: each lock is moved
                # by its own pixel tracker, snapped to the best matching
                # detection, and every other box sitting on the same object is
                # removed — so a lock is always exactly one box that stays put
                # while the camera moves.
                lock_rows: list[dict] = []
                if lock_trackers:
                    keep = np.ones(len(detections), dtype=bool)
                    for tid, lock in list(lock_trackers.items()):
                        moved = lock.update(frame)
                        ious = (
                            iou_matrix(lock.box.reshape(1, 4), detections.xyxy)[0]
                            if len(detections)
                            else np.zeros(0)
                        )
                        matched = False
                        if len(ious) and float(ious.max()) >= LOCK_MATCH_IOU:
                            best = int(np.argmax(ious))
                            conf = (
                                float(detections.confidence[best])
                                if detections.confidence is not None
                                else 0.0
                            )
                            lock.correct(
                                frame, detections.xyxy[best], labels[best], conf
                            )
                            matched = True
                        if not matched and not moved:
                            if now - lock.last_seen > LOCK_GRACE_SECONDS:
                                log.info("[%s] lock %s released (lost)", self.path, tid)
                                self.store.clear_lock(self.session_id, tid)
                                lock_trackers.pop(tid, None)
                                continue
                        if len(detections):
                            cont = containment_matrix(
                                lock.box.reshape(1, 4), detections.xyxy
                            )[0]
                            overlapping = (cont > RANGE_CONTAINMENT) | (
                                ious > RANGE_DEDUPE_IOU
                            )
                            keep &= ~overlapping
                        lock_rows.append(
                            lock_row(
                                self.session_id,
                                tid,
                                lock.box,
                                width,
                                height,
                                lock.label,
                                lock.confidence,
                            )
                        )
                    if not keep.all():
                        idxs = np.flatnonzero(keep).tolist()
                        labels = [labels[i] for i in idxs]
                        detections = detections[idxs]

                if RANGE_DEBUG:
                    log.info(
                        "[%s] sources: fast=%d range=%d motion=%d locks=%d merged=%d after-dedupe=%d",
                        self.path,
                        raw_count,
                        range_count,
                        motion_count,
                        len(lock_trackers),
                        merged_count,
                        len(detections),
                    )

                detections = tracker.update(detections, labels, frame)
                if motion_compensated:
                    step = tracker.last_warp()
                    warp_log.append((seq, step))
                    range_warp = compose_warp(step, range_warp)
                    motion_warp = compose_warp(step, motion_warp)
                    crop_warp = compose_warp(step, crop_warp)
                if cropper is not None:
                    # Pause the crop pass while the model is contended (the
                    # fast pass of every stream comes first), with hysteresis.
                    queue_ema += 0.1 * (timing.get("wait_ms", 0.0) - queue_ema)
                    if crop_active and queue_ema > CROP_MAX_QUEUE_MS:
                        crop_active = False
                        log.info("[%s] crop pass paused (model queue %.0f ms)", self.path, queue_ema)
                    elif not crop_active and queue_ema < CROP_MAX_QUEUE_MS / 2:
                        crop_active = True
                        log.info("[%s] crop pass resumed (model queue %.0f ms)", self.path, queue_ema)
                    # Where the small tracks are now, in full-resolution px,
                    # for the crop pass's next round.
                    crop_targets["boxes"] = (
                        tracker.small_targets(SMALL_TRACK_MAX_SIDE, MAX_CROPS) / scale
                        if crop_active
                        else None
                    )
                rows = build_rows(detections, self.session_id, width, height)
                rows = [r for r in rows if r["track_id"] not in lock_trackers]
                rows.extend(lock_rows)

                if LOG_EVERY_FRAME:
                    log.info(
                        "[%s] %d raw -> %d tracked -> %d row(s) in %.0f ms",
                        self.path,
                        raw_count,
                        len(detections),
                        len(rows),
                        infer_ms,
                    )

                status.update(self.session_id, active_tracks=len(rows))
                if DETECTIONS_TRANSPORT == "broadcast":
                    # Only locks live in the table now: the UI owns the flag and
                    # the lock poll below reads it back from there.
                    self.store.upsert(lock_rows)
                else:
                    self.store.upsert(rows)

                if broadcaster is not None:
                    # Throttled to BROADCAST_MAX_HZ on average; with nothing to
                    # show, only a keep-alive empty snapshot every
                    # BROADCAST_IDLE_SECONDS.
                    hz = max(0.1, BROADCAST_MAX_HZ)
                    broadcast_budget = min(2.0, broadcast_budget + (now - budget_at) * hz)
                    budget_at = now
                    if not rows and last_broadcast_empty:
                        due = now - last_broadcast >= BROADCAST_IDLE_SECONDS
                    else:
                        due = broadcast_budget >= 1.0
                    if due:
                        broadcaster.publish(
                            self.session_id,
                            snapshot_tracks(rows, set(lock_trackers), velocity, now),
                        )
                        broadcast_budget = max(0.0, broadcast_budget - 1.0)
                        last_broadcast = now
                        last_broadcast_empty = not rows

                stats.add(
                    frame_age_ms=(now - grabbed_at) * 1000.0,
                    wait_ms=timing.get("wait_ms", 0.0),
                    infer_ms=infer_ms,
                    loop_ms=(time.time() - now) * 1000.0,
                    tracks=len(rows),
                )
                if stats.due():
                    metrics = stats.flush(
                        range_scan_ms=scanner.last_scan_ms if scanner else None,
                        crop_scan_ms=cropper.last_scan_ms if cropper else None,
                        crops=cropper.last_crops if cropper else 0,
                        locks=len(lock_trackers),
                    )
                    status.update(self.session_id, metrics=metrics)
                    log.info(
                        "[%s] %.1f fps | infer %.0f ms (+%.0f ms queue) | "
                        "frame age %.0f ms | loop %.0f ms (max %.0f) | "
                        "%.1f tracks | range scan %s ms | crops %d (%s ms) | %d lock(s)",
                        self.path,
                        metrics["fps"],
                        metrics["avg_infer_ms"],
                        metrics["avg_queue_ms"],
                        metrics["avg_frame_age_ms"],
                        metrics["avg_loop_ms"],
                        metrics["max_loop_ms"],
                        metrics["avg_tracks"],
                        metrics["range_scan_ms"],
                        metrics["crops"],
                        metrics["crop_scan_ms"],
                        metrics["locks"],
                    )

                # Locks are created by the UI (on a box, or by clicking anywhere
                # in the picture) — poll them at a low rate and start a pixel
                # tracker for each new one.
                if now - last_locked_poll >= LOCK_POLL_SECONDS:
                    last_locked_poll = now
                    locked = self.store.locked_tracks(self.session_id)
                    for tid in list(lock_trackers):
                        if tid not in locked:
                            log.info("[%s] lock %s released by user", self.path, tid)
                            lock_trackers.pop(tid, None)
                    for tid, row in locked.items():
                        if tid in lock_trackers:
                            continue
                        bbox = row.get("bbox") or {}
                        try:
                            bx = float(bbox.get("x", 0.0)) * width
                            by = float(bbox.get("y", 0.0)) * height
                            bw = max(4.0, float(bbox.get("width", 0.0)) * width)
                            bh = max(4.0, float(bbox.get("height", 0.0)) * height)
                        except (TypeError, ValueError):
                            continue
                        lock_trackers[tid] = LockTracker(
                            tid,
                            frame,
                            [bx, by, bx + bw, by + bh],
                            str(row.get("object_class") or UNKNOWN_CLASS),
                            float(row.get("confidence") or 0.0),
                        )
                        log.info("[%s] lock %s acquired", self.path, tid)
        finally:
            if scanner is not None:
                scanner.stop()
            if cropper is not None:
                cropper.stop()
            if motion is not None:
                motion.stop()
            grabber.stop()
            cap.release()
            status.update(self.session_id, connected=False, active_tracks=0)


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #


def cleanup_loop(store: DetectionStore, workers: dict[str, StreamWorker]) -> None:
    while True:
        # Sweep at least twice per TTL so boxes vanish quickly after an object
        # leaves the frame, with a 0.4s floor to keep write volume sane. In
        # broadcast mode the table only holds locks (5 s grace), so a slow
        # sweep is enough and saves ~4 DELETEs per second.
        if DETECTIONS_TRANSPORT == "broadcast":
            time.sleep(2.0)
        else:
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

        if IDLE_EXIT_MINUTES > 0 and status.idle_seconds() >= IDLE_EXIT_MINUTES * 60:
            log.info(
                "No video for %.0f min — exiting so the machine can sleep "
                "(MediaMTX wakes it through /wake)",
                IDLE_EXIT_MINUTES,
            )
            for worker in workers.values():
                worker.stop()
            for worker in workers.values():
                worker.join(timeout=5)  # clears their rows / sends empty snapshots
            time.sleep(0.5)  # let a final empty broadcast leave the socket
            raise SystemExit(0)

        # Sleep until the next discovery round, or until /wake says a stream
        # (or a viewer) just appeared.
        wake_event.wait(DISCOVERY_INTERVAL_SECONDS)
        wake_event.clear()


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
    status.model = detector.model_path
    log.info("Detector config: %s", detector.describe())
    log.info(
        "Auto-discovery every %.0fs from %s (max %d stream(s))%s",
        DISCOVERY_INTERVAL_SECONDS,
        RTSP_BASE_URL,
        MAX_STREAMS,
        f", sleeps after {IDLE_EXIT_MINUTES:g} min without video" if IDLE_EXIT_MINUTES > 0 else "",
    )

    store = DetectionStore()
    supervise(detector, store)


if __name__ == "__main__":
    main()

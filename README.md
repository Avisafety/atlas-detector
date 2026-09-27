# atlas-detector

Continuous object detection and tracking (person / vehicle / vessel / aircraft
and more) on Atlas drone video. It discovers every active flight with a live
Atlas stream from Supabase, reads each stream over RTSP from MediaMTX on the
Fly private network, runs YOLO26n (ONNX Runtime) + ByteTrack on CPU, and writes tracked
objects to the Supabase table `atlas_detections` so the Avisafe frontend can
draw a live overlay through Supabase Realtime.

## Architecture

```text
Atlas drone --RTMPS--> live-video (MediaMTX) --RTSP (6PN, :8554)--> atlas-detector
                                                                          |
                                                          upsert atlas_detections
                                                                          |
                                                     Supabase Realtime -> frontend
```

RTSP is enabled in `atlas-video/mediamtx.yml` but **not** published in
`atlas-video/fly.toml`, so port 8554 is only reachable at
`live-video.internal` inside the Fly organisation.

No serial number is configured: a supervisor loop polls Supabase every
`DISCOVERY_INTERVAL_SECONDS` for active flights whose drones have a live
sensor registered in `atlas_drone_sensors`, and starts one worker per stream
(up to `MAX_STREAMS`). Workers stop — and their boxes are deleted — when the
flight ends or the stream goes stale.

## Getting boxes to the browser

`DETECTIONS_TRANSPORT` selects the path:

| Value | What happens |
|---|---|
| `postgres` | Every track is upserted into `atlas_detections`; the frontend listens to Postgres Changes (the original behaviour). |
| `broadcast` | One Supabase Realtime **Broadcast** snapshot per frame on the private channel `atlas-detections:<flight_session_id>` (event `tracks`). Only locked tracks are still written to the table. |
| `both` | Both at once — for the transition; the frontend prefers Broadcast. |

Snapshot payload (a track missing from a snapshot is gone):

```json
{ "v": 1, "flight_session_id": "uuid", "sent_at": 1727330000123,
  "tracks": [ { "id": 17, "cls": "person", "conf": 0.84,
                "x": 0.41, "y": 0.22, "w": 0.05, "h": 0.12,
                "vx": 0.012, "vy": -0.003, "locked": false } ] }
```

Coordinates are normalised (top-left + size), `vx`/`vy` in normalised units
per second so the frontend can glide boxes between snapshots. Snapshots are
throttled to `BROADCAST_MAX_HZ`; with nothing detected an empty keep-alive is
sent every `BROADCAST_IDLE_SECONDS`, and an empty snapshot is sent right away
when a stream drops. Messages go over one persistent Realtime websocket
(~10 ms each from Fly); while it is down they fall back to the Realtime REST
endpoint (~110 ms). Viewers are authorised by the RLS policy on
`realtime.messages`.

## Two cooperating passes, one tracker

- **Fast pass** — the whole frame downscaled to `INFER_MAX_SIDE` (default 640),
  analysed `DETECTION_FPS` times per second. This drives box responsiveness.
- **Range pass** — the full-resolution frame split into a
  `RANGE_TILE_COLS` × `RANGE_TILE_ROWS` grid with `RANGE_TILE_OVERLAP`,
  analysed every `RANGE_PASS_INTERVAL_SECONDS` (default 2 s) at
  `RANGE_CONFIDENCE`. Catches small, distant objects the downscale loses.

Range boxes are mapped to full-frame coordinates and merged with the fast-pass
boxes into **one** detection list, which is deduped in a single class-aware
pass before it reaches the single ByteTrack instance — one object is always one
box. Suppression uses IoU (`RANGE_DEDUPE_IOU`) **and** containment
(`RANGE_CONTAINMENT`, intersection over the smaller box): a tile often sees only
part of a large object, and that fragment box sits inside the full-frame box
where IoU stays low. Tile-truncated boxes (touching a tile edge that is not a
frame edge) are dropped at the source, and range results older than
`RANGE_RESULT_MAX_AGE_SECONDS` are not re-fed, so a moving object never leaves a
ghost box behind. `RANGE_DEBUG=true` logs per-source counts and every
suppression.


## Deploy

```bash
cd atlas-detector
fly launch --no-deploy --name atlas-detector --org <same org as live-video>
fly secrets set \
  SUPABASE_URL="https://wazxzyygflomhyoomxcc.supabase.co" \
  SUPABASE_SERVICE_ROLE_KEY="<service role key>" \
  DETECTOR_SHARED_SECRET="<shared secret>"
fly deploy
```

`DETECTOR_SHARED_SECRET` is appended as `?detector=<secret>` when reading RTSP;
`atlas-video-auth` only accepts it from the Fly private network (`fdaa::/16`).
It must also be set as a Supabase edge-function secret.

For manual testing against one fixed stream, pin it instead of discovery:

```bash
fly secrets set \
  MEDIAMTX_RTSP_URL="rtsp://live-video.internal:8554/<serial>/<sensor>?detector=<secret>" \
  FLIGHT_SESSION_ID="<active_flights.id>"
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SUPABASE_URL` | – | Required |
| `SUPABASE_SERVICE_ROLE_KEY` | – | Required |
| `DETECTOR_SHARED_SECRET` | – | Required for RTSP auth |
| `RTSP_BASE_URL` | `rtsp://live-video.internal:8554` | Stream base (path = `<serial>/<sensor>`) |
| `MEDIAMTX_RTSP_URL` | – | Optional: pin one stream (manual testing) |
| `FLIGHT_SESSION_ID` | – | Required together with `MEDIAMTX_RTSP_URL` |
| `MAX_STREAMS` | `2` | Simultaneous streams analysed |
| `DISCOVERY_INTERVAL_SECONDS` | `10` | How often live streams are (re)discovered |
| `SENSOR_STALE_SECONDS` | `300` | A sensor counts as live for this long |
| `DETECTION_FPS` | `10` | Fast-pass frames analysed per second |
| `MODEL_PATH` | `yolo26n.onnx` | Model file; the ONNX export is baked into the image |
| `FALLBACK_MODEL_PATH` | `yolo26n.pt` | Used when `MODEL_PATH` is missing or fails to load |
| `DETECTION_CLASSES` | `person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,bird,dog,horse,sheep,cow,kite,surfboard` | COCO class names; the UI filters which are drawn |
| `DETECTION_CONFIDENCE` | `0.20` | Fast-pass minimum score (UI filters further) |
| `INFER_MAX_SIDE` | `640` | Fast-pass downscale, longest side (0 = off) |
| `TRACKER_LOST_BUFFER` | `15` | Analysed frames a lost track survives |
| `TRACK_TTL_SECONDS` | `0.8` | Stale tracks are deleted after this |
| `RANGE_PASS_ENABLED` | `true` | Tiled full-resolution range pass on/off |
| `RANGE_PASS_INTERVAL_SECONDS` | `2.0` | Range-pass cadence |
| `RANGE_TILE_COLS` / `RANGE_TILE_ROWS` | `3` / `2` | Range-pass tile grid |
| `RANGE_TILE_OVERLAP` | `0.15` | Tile overlap fraction |
| `RANGE_CONFIDENCE` | `0.15` | Range-pass minimum score (small objects score lower) |
| `RANGE_DEDUPE_IOU` | `0.5` | IoU at which duplicate boxes are merged |
| `RANGE_CONTAINMENT` | `0.7` | Containment (overlap / smaller box) at which duplicates are merged |
| `RANGE_RESULT_MAX_AGE_SECONDS` | `0` | Max age of re-fed range boxes (0 = interval + 0.5 s) |
| `RANGE_DEBUG` | `false` | Temporary per-source / per-suppression logging |
| `LOG_SUMMARY_SECONDS` | `10` | Interval of the per-stream summary log line and `/health` metrics |
| `LOG_EVERY_FRAME` | `false` | Also log one line per analysed frame (very verbose) |
| `DB_WRITER_THREADS` | `MAX_STREAMS` | Parallel Supabase writers (one per stream slot) |
| `IDLE_EXIT_MINUTES` | `0` (fly.toml: `10`) | Exit after this long without an analysed frame so the Fly machine sleeps; `0` = never |
| `DETECTIONS_TRANSPORT` | `both` | `postgres`, `broadcast` or `both` (see above) |
| `BROADCAST_MAX_HZ` | `10` | Broadcast snapshots per second per stream |
| `BROADCAST_IDLE_SECONDS` | `1.0` | Keep-alive interval for empty snapshots |
| `BROADCAST_TOPIC_PREFIX` | `atlas-detections:` | Channel name prefix (+ flight_session_id) |
| `BROADCAST_EVENT` | `tracks` | Broadcast event name |

## Sleeping and waking (scale to zero)

With `IDLE_EXIT_MINUTES` set, the detector exits cleanly after that many
minutes without a single analysed frame, and Fly stops the machine (restart
policy `on-failure`; `min_machines_running = 0`). An active flight whose stream
is gone does not keep it awake — only real video (or a wake request) does.

It is woken by `GET /wake` on its private Flycast address
(`http://atlas-detector.flycast/wake`): Fly's proxy starts the stopped machine
to deliver the request, and the detector runs discovery immediately. MediaMTX
(`live-video`) sends that request from its `runOnReady` hook (a drone started
publishing) and `runOnRead` hook (someone started watching). The Flycast IP is
allocated once with `fly ips allocate-v6 --private -a atlas-detector`. The
proxy never stops the machine itself (`auto_stop_machines = "off"`), because
the detector receives no inbound traffic while it works.

## Health

`GET /health` returns connection state, last frame age, active track count and
reconnect count — used by the Fly health check in `fly.toml`. It also carries a
rolling `metrics` block per stream and a `db_writer` block, refreshed every
`LOG_SUMMARY_SECONDS`:

| Field | Meaning |
|---|---|
| `fps` | Fast-pass frames actually analysed per second |
| `avg_frame_age_ms` | Decode → analysis start (how stale the analysed frame is) |
| `avg_queue_ms` | Time spent waiting for the shared model (contention) |
| `avg_infer_ms` | Model time per fast-pass frame |
| `avg_loop_ms` / `max_loop_ms` | Analysis start → rows handed to the writer |
| `range_scan_ms` | Duration of the last full range pass |
| `db_writer.skipped_batches` | Frames whose rows were replaced before being written (writer too slow) |

The same numbers are logged as one line per stream every `LOG_SUMMARY_SECONDS`,
so the live log in the Fly dashboard stays readable.

Inference is shared by all streams through a priority queue: fast-pass frames
go before range-pass tiles, and anything that has waited 250 ms goes next, so
the range pass is never starved when many streams are busy.

## Benchmark

`python bench.py [image] [runs]` times the fast pass and a range tile on every
model file present (`yolo26n.pt`, `yolo26n.onnx`) and checks that the backends
agree on the boxes. Run it on the target machine to size `MAX_STREAMS`.

## Notes

- CPU only; runs on a dedicated-CPU Fly machine (`performance-2x`). The ONNX
  Runtime export is ~3x faster than PyTorch with the same boxes.
- Bounding boxes are normalised to 0–1 relative to frame width/height.
- One row per `(flight_session_id, track_id)`; stale rows are pruned by a
  background loop and cleared entirely when the stream drops.
- The frontend draws boxes filtered by a per-user sensitivity slider and a
  per-class on/off filter — both are display filters only, so the detector
  always writes at the low raw thresholds above.

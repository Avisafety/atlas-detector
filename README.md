# atlas-detector

Continuous object detection and tracking (person / vehicle / vessel / aircraft
and more) on Atlas drone video. It discovers every active flight with a live
Atlas stream from Supabase, reads each stream over RTSP from MediaMTX on the
Fly private network, runs YOLO26n + ByteTrack on CPU, and writes tracked
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
| `DETECTION_CLASSES` | `person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,bird,dog,horse,sheep,cow,kite,surfboard` | COCO class names; the UI filters which are drawn |
| `DETECTION_CONFIDENCE` | `0.20` | Fast-pass minimum score (UI filters further) |
| `INFER_MAX_SIDE` | `640` | Fast-pass downscale, longest side (0 = off) |
| `TRACKER_LOST_BUFFER` | `5` | Analysed frames a lost track survives |
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

## Health

`GET /health` returns connection state, last frame age, active track count and
reconnect count — used by the Fly health check in `fly.toml`.

## Notes

- CPU only; runs on a dedicated-CPU Fly machine (`performance-2x`).
- Bounding boxes are normalised to 0–1 relative to frame width/height.
- One row per `(flight_session_id, track_id)`; stale rows are pruned by a
  background loop and cleared entirely when the stream drops.
- The frontend draws boxes filtered by a per-user sensitivity slider and a
  per-class on/off filter — both are display filters only, so the detector
  always writes at the low raw thresholds above.

# atlas-detector

Continuous object detection and tracking (person / vehicle / vessel) on Atlas
drone video. Reads one MediaMTX stream over RTSP on the Fly private network,
runs YOLOv8n + ByteTrack on CPU, and writes tracked objects to the Supabase
table `atlas_detections` so the Avisafe frontend can draw a live overlay
through Supabase Realtime.

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

## Deploy

```bash
cd atlas-detector
fly launch --no-deploy --name atlas-detector --org <same org as live-video>
fly secrets set \
  MEDIAMTX_RTSP_URL="rtsp://live-video.internal:8554/APA006SILVERRAVEN/1?detector=<DETECTOR_SHARED_SECRET>" \
  SUPABASE_URL="https://wazxzyygflomhyoomxcc.supabase.co" \
  SUPABASE_SERVICE_ROLE_KEY="<service role key>" \
  FLIGHT_SESSION_ID="<active_flights.id>"
fly deploy
```

`DETECTOR_SHARED_SECRET` must also be set as a Supabase edge-function secret —
`atlas-video-auth` only accepts it when the request comes from the Fly private
network (`fdaa::/16`).

Remember to redeploy MediaMTX after the RTSP change:

```bash
cd ../atlas-video && fly deploy
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MEDIAMTX_RTSP_URL` | – | Stream to analyse (required) |
| `SUPABASE_URL` | – | Required |
| `SUPABASE_SERVICE_ROLE_KEY` | – | Required |
| `FLIGHT_SESSION_ID` | – | `active_flights.id` the detections belong to (required) |
| `DETECTION_FPS` | `5` | Frames analysed per second |
| `DETECTION_CLASSES` | `person,car,truck,bus,motorcycle,boat` | COCO class names |
| `DETECTION_CONFIDENCE` | `0.35` | Minimum score |
| `TRACK_TTL_SECONDS` | `3` | Tracks without updates are deleted after this |

## Health

`GET /health` returns connection state, last frame age, active track count and
reconnect count — used by the Fly health check in `fly.toml`.

## Notes

- CPU only; torch is installed from the CPU wheel index.
- Bounding boxes are normalised to 0–1 relative to frame width/height.
- One row per `(flight_session_id, track_id)`; stale rows are pruned by a
  background loop and cleared entirely when the stream drops.

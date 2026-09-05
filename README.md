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
| `DETECTION_ENGINE` | `yolo` | `yolo` or `grounding_dino` |
| `DETECTION_FPS` | `5` | Frames analysed per second (YOLO) |
| `DETECTION_CLASSES` | `person,car,truck,bus,motorcycle,boat` | COCO class names (YOLO) |
| `DETECTION_CONFIDENCE` | `0.35` | Minimum score (YOLO) |
| `DETECTION_FPS_GROUNDING_DINO` | `0.2` | Frames per second (Grounding DINO) |
| `DETECTION_TEXT_PROMPT` | `person . car . boat . truck . bus . motorcycle` | Prompt (Grounding DINO) |
| `DETECTION_BOX_THRESHOLD` | `0.25` | Box confidence (Grounding DINO) |
| `DETECTION_TEXT_THRESHOLD` | `0.20` | Text-match threshold (Grounding DINO) |
| `TRACK_TTL_SECONDS` | `3` | Tracks without updates are deleted after this |

## Detection engines

Both engines live in `app.py` and are selected with `DETECTION_ENGINE`:

- **`yolo` (default)** — YOLOv8n on the fixed COCO class list. Fast, ~5 fps.
- **`grounding_dino`** — `IDEA-Research/grounding-dino-tiny` (Swin-T) through
  Hugging Face `transformers`. Open vocabulary: categories come from
  `DETECTION_TEXT_PROMPT`, separated by ` . `. Slower on CPU, so it defaults to
  one frame every 5 seconds. `object_class` is the text label the model matched.

Both engines feed the same ByteTrack tracker and write the same rows to
`atlas_detections`. The startup log prints the active engine, checkpoint,
prompt/classes, thresholds and effective fps — check `fly logs` to confirm.

```bash
fly secrets set DETECTION_ENGINE=grounding_dino \
  DETECTION_TEXT_PROMPT="person . boat . life raft . kayak"
```

## Health

`GET /health` returns connection state, last frame age, active track count and
reconnect count — used by the Fly health check in `fly.toml`.

## Notes

- CPU only; torch is installed from the CPU wheel index.
- Bounding boxes are normalised to 0–1 relative to frame width/height.
- One row per `(flight_session_id, track_id)`; stale rows are pruned by a
  background loop and cleared entirely when the stream drops.

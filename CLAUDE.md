# atlas-detector — notes for Claude

Continuous object detection on live Atlas drone video (YOLO26n via ONNX
Runtime + ByteTrack, CPU only). One Python service, `app.py`, deployed to the
Fly app `atlas-detector` (region `ams`, one `performance-2x` machine). It
reads RTSP from the `live-video` MediaMTX app over Fly's private network and
writes boxes to the Supabase table `atlas_detections`. See README.md for the
architecture and every environment variable.

## Working with the owner

- The owner (Gard, AviSafe) writes in Norwegian and works from an iPad: no
  local terminal. Answer in Norwegian, keep instructions tappable (dashboard
  paths, not shell commands they must run).
- Never deploy, change secrets, scale or restart machines without an explicit
  "yes" in the conversation. A deploy takes the detector down for ~1 minute —
  check whether a flight is being tested first (`fly logs`).
- Work on the session's feature branch; `main` is what production was last
  deployed from. Do not push to `main`.

## Fly.io

`flyctl` is installed by the environment's setup script and authenticates with
`FLY_API_TOKEN` (a deploy token scoped to this app).

```sh
fly status -a atlas-detector                 # machines, version, health check
fly logs -a atlas-detector --no-tail         # recent logs (buffer is short)
fly releases -a atlas-detector --image       # versions + image refs, for rollback
fly secrets list -a atlas-detector           # names only
```

Deploy (always from the repo root, after the checks below):

```sh
fly deploy --remote-only --depot=false -a atlas-detector
```

- `--depot=false` is required here: the default Depot builder uses its own
  TLS certificates, which the session's egress proxy cannot pass. Fly's own
  builder (`fly-builder-*` app) works.
- `fly deploy --build-only` validates the Dockerfile on Fly but does NOT push
  the image, so it cannot be deployed with `--image` afterwards.
- Rollback: `fly deploy -a atlas-detector --image <previous image ref>` using
  a ref from `fly releases --image`.
- Before deploying, diff the live config against `fly.toml`
  (`fly config show -a atlas-detector`) so a deploy never silently changes
  env values that were set another way.

Secrets override `fly.toml` `[env]`: `DETECTION_CONFIDENCE` (0.35 in
production, not the 0.20 in fly.toml) and `SENSOR_STALE_SECONDS` are secrets.

The app has no public IP on purpose (`/health` exposes flight ids and drone
serials). Read it from inside the machine:

```sh
fly ssh console -a atlas-detector -C "python -c \"import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/health').read().decode())\""
```

`/health` → `streams[].metrics` has fps, inference/queue time, frame age, loop
time and range-scan time per stream; `db_writer` has Supabase write stats. The
same numbers are logged every `LOG_SUMMARY_SECONDS` as one line per stream.

## Checking a change before it ships

The session container cannot run the full Docker build (Debian mirrors and
download.pytorch.org are blocked), so:

1. `pyflakes app.py bench.py tools/smoke_test.py`
2. Install requirements in a venv (torch from PyPI works; the CPU index is
   blocked), export the ONNX model the same way the Dockerfile does, then run
   `tools/smoke_test.py` pinned to 2 cores (`taskset -c 0,1`) against both the
   old and the new `app.py` — 1 stream, `H_STREAMS=3` and `H_LOCK=1`. Compare
   fps, rows per upsert, classes and that the range pass still reports.
3. `python bench.py <image>` checks that the PyTorch and ONNX backends agree.
4. `fly deploy --build-only --remote-only --depot=false` to prove the image
   builds on Fly.

## Known state (September 2026)

- Measured on Fly: ~45 ms per fast-pass inference on performance-2x.
- Test streams from the Larix phone app arrive in bursts (~200 ms gaps), which
  caps analysis at ~8 fps. The owner reports that video from the Atlas
  controller looks normal — not yet measured with the frame-gap probe.
- Planned next: Supabase Realtime Broadcast instead of Postgres Changes for
  the overlay (frontend in Lovable), track-guided native-resolution crops for
  range, camera-motion compensation in the tracker, and a lease table so
  several machines can share many streams.

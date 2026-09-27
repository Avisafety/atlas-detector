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

`flyctl` is installed by the environment's setup script. The cloud
environment holds one deploy token per app, each scoped to that app only:

| Variable | App |
|---|---|
| `FLY_TOKEN_ATLAS_DETECTOR` | `atlas-detector` |
| `FLY_TOKEN_LIVE_VIDEO` | `live-video` (MediaMTX, repo Avisafety/live_video) |
| `FLY_TOKEN_DJILOGPARSER` | `djilogparser` |

flyctl only reads `FLY_API_TOKEN`, so pass the right one per command, e.g.
`FLY_API_TOKEN="$FLY_TOKEN_ATLAS_DETECTOR" fly status -a atlas-detector`.
Never print token values. Variables appear only in sessions started after
they were saved (one per line, no hyphens in names).

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
   fps, rows per upsert, classes and that the range pass still reports. For
   the broadcast path add `H_WS=1` (fake Realtime websocket server) and
   `H_WS_DROP=1` (connection drop + reconnect), and run each
   `DETECTIONS_TRANSPORT` value.
3. `python bench.py <image>` checks that the PyTorch and ONNX backends agree.
4. `fly deploy --build-only --remote-only --depot=false` to prove the image
   builds on Fly.

## Known state (September 2026)

- Measured on Fly: ~45 ms per fast-pass inference on performance-2x.
- Test streams from the Larix phone app arrive in bursts (~200 ms gaps), which
  caps analysis at ~8 fps. The owner reports that video from the Atlas
  controller looks normal — not yet measured with the frame-gap probe.
- Production (v58, deployed from `main` at d258eed) runs
  DETECTIONS_TRANSPORT=broadcast: boxes reach the browser only through the
  private Realtime channel (frontend HUD shows BROADCAST · SUBSCRIBED); only
  locked tracks are written to atlas_detections. `both` / `postgres` are the
  rollback. Measured from Fly: websocket ~10-15 ms per message, REST fallback
  ~110 ms. Supabase plan: Pro, micro compute — mind the Realtime messages/s
  quota (counted per recipient).
- Open issue on the frontend side: locking a detected (positive id) box uses
  an upsert that the atlas_detections INSERT policy (track_id < 0) rejects.
  The fix is a SECURITY DEFINER RPC (lock_detection/unlock_detection) in the
  Lovable project, not a detector change.
- Scale to zero (v59): the detector exits after IDLE_EXIT_MINUTES=10 without
  video and the machine stops (min_machines_running = 0). MediaMTX in the
  `live-video` app (repo Avisafety/live_video, `mediamtx.yml` runOnReady /
  runOnRead) wakes it with `wget http://atlas-detector.flycast/wake`; the
  private Flycast IP is fdaa:38:d7df:0:1::2. A stopped machine is normal —
  `fly status` showing "stopped" between flights is not an outage. Test the
  wake path with
  `fly ssh console -a live-video -C "wget -T 20 -O - http://atlas-detector.flycast/wake"`.
  Deploying live-video interrupts all video briefly (publishers reconnect);
  do it only with the owner's yes and preferably when nobody is flying. Use
  `FLY_TOKEN_LIVE_VIDEO` for it.
- The Lovable project must never edit or deploy a copy of this detector; the
  detector is maintained only here.
- Tracker: BoT-SORT + MaskedFlowGMC (README "Tracking with a moving camera").
  `TRACKER_IMPL=bytetrack` is the rollback. Use `H_PAN=1` (camera pan) and
  `H_IDS_DETAIL=1` in tools/smoke_test.py to check id stability; compare
  against the previous app.py. DETECTION_CONFIDENCE secret: owner staged 0.2
  (was 0.35) to go out with the tracker release.
- Known: distant objects seen only by the range pass (every 4 s) still get
  short tracks between scans — round 3 (track-guided native-resolution
  crops) fixes that. Lock labels come from the last matching detection and
  can flip (no voting for locks yet).
- Planned next: round 3 range crops; later a lease table so several machines
  can share many streams; custom training on own footage (not now).

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
- The owner has authorised Claude to push to `main` and to deploy through the
  "Deploy to Fly" GitHub Actions workflow once a change is tested (27 Sep
  2026). Develop on the session branch, run the checks below, then push to
  main and trigger the workflow. Before triggering, check for an active
  stream (`fly logs`): a deploy takes the detector down for ~1 minute — if a
  flight or test is running, ask first. Say in the conversation what is being
  deployed and report the result.
- Still never change secrets, scale or restart machines, or deploy
  `live-video`, without an explicit "yes" in the conversation.
- `main` must always equal production: push to main only what you deploy.
- The owner can also deploy from the Fly dashboard (GitHub integration,
  manual); it is not automatic on push.

## Fly.io

`flyctl` is installed by the environment's setup script. The cloud
environment holds one deploy token per app, each scoped to that app only:

| Variable | App |
|---|---|
| `claude_access` | `atlas-detector` |
| `claude_access_live_video` | `live-video` (MediaMTX, repo Avisafety/live_video) |
| `claude_access_djilogparser` | `djilogparser` |

flyctl only reads `FLY_API_TOKEN`, so pass the right one per command, e.g.
`FLY_API_TOKEN="$TOK" fly status -a atlas-detector`. The saved values are
wrapped in « » quotes and may lack the space after FlyV1, so normalise first:
`TOK=$(python3 -c "import os;v=os.environ['claude_access'].strip().strip('«»\"\' ');v=v[5:].lstrip() if v.startswith('FlyV1') else v;print('FlyV1 '+v)")`.
Never print token values. Variables appear only in sessions started after
they were saved (one per line, no hyphens in names).

```sh
fly status -a atlas-detector                 # machines, version, health check
fly logs -a atlas-detector --no-tail         # recent logs (buffer is short)
fly releases -a atlas-detector --image       # versions + image refs, for rollback
fly secrets list -a atlas-detector           # names only
```

Deploy — preferred path: GitHub Actions workflow "Deploy to Fly"
(`.github/workflows/deploy.yml`, manual `workflow_dispatch`, main only; the
app-scoped deploy token lives in the repo secret `FLY_API_TOKEN`). Push to
main, then trigger it with `actions_run_trigger` (GitHub MCP tools, workflow
`deploy.yml`, ref `main`) and follow the run with `actions_get` /
`get_job_logs`. The owner can also run it from the GitHub app.

Direct deploy from a session only works with a token that can reach Fly's
builder app (an app-scoped token gets "remote builder app unavailable"):

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

Secrets override `fly.toml` `[env]`: `DETECTION_CONFIDENCE` (0.2 in
production since v60) and `SENSOR_STALE_SECONDS` are secrets.

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
  `claude_access_live_video` for it.
- The Lovable project must never edit or deploy a copy of this detector; the
  detector is maintained only here.
- Tracker: BoT-SORT + MaskedFlowGMC (README "Tracking with a moving camera").
  `TRACKER_IMPL=bytetrack` is the rollback. Use `H_PAN=1` (camera pan) and
  `H_IDS_DETAIL=1` in tools/smoke_test.py to check id stability; compare
  against the previous app.py. Live since v60 (deployed via the GitHub Actions
  workflow) with DETECTION_CONFIDENCE=0.2 (secret, was 0.35).
- Fly tokens are "FlyV1 fm2_..." — the space after FlyV1 is part of the
  token; without it Fly answers Unauthorized.
- Crop pass (round 3): native-resolution windows around small tracks,
  `CROP_FPS=2`, pauses itself when the model queue exceeds
  `CROP_MAX_QUEUE_MS` (several streams). Background results carry the seq of
  the frame they were computed on and are warped to the current frame
  (warp_log in the worker + this frame's step in StreamTracker). Check with
  `H_PAN=1 H_IDS_DETAIL=1` (small people keep one id) and `H_STREAMS=3`
  (log shows "crop pass paused"; fps close to the previous app.py).
  `CROP_PASS_ENABLED=false` is the rollback. Lock labels come from the last
  matching detection and can flip (no voting for locks yet).
- MAX_STREAMS=3 (fly.toml, since v64). Discovery counts a flight as live when
  its drone has registered a sensor, not when video arrives; a worker with no
  video for SLOT_RELEASE_SECONDS gives its slot to a waiting flight (log: "no
  video for … giving its slot"). Check with
  `H_DEAD_FIRST=2 H_STREAMS=2 H_MAX_STREAMS=3 SLOT_RELEASE_SECONDS=10`.
- Planned next: a lease table so several machines can share many streams;
  faster ORT threading for multi-stream; custom training on own footage
  (not now).

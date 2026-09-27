"""End-to-end smoke test for app.py without RTSP or Supabase.

Feeds app.py a fake real-time 1080p/25 fps video (the Ultralytics sample
images, one moving and one small "distant" object) and a fake Supabase client
with a 50 ms round trip, runs it for N seconds and prints what would have
been written plus the /health metrics.

    python tools/smoke_test.py app.py 45                  # one pinned stream
    H_STREAMS=3 python tools/smoke_test.py app.py 45      # three discovered streams
    H_LOCK=1 python tools/smoke_test.py app.py 35         # a manual lock (track -1)
    MODEL_PATH=missing.onnx python tools/smoke_test.py app.py 25   # fallback to .pt

Pin it to the target's core count (`taskset -c 0,1 ...` for performance-2x)
and run it against the old and the new app.py to compare. Needs the model
files in the working directory (yolo26n.pt, yolo26n.onnx — see bench.py).
"""
import collections
import importlib.util
import json
import os
import sys
import threading
import time
import urllib.request

import cv2, numpy as np

APP, SECONDS = sys.argv[1], float(sys.argv[2])
A = os.path.join(os.path.dirname(__import__('ultralytics').__file__), 'assets')
bus, zid = cv2.imread(f'{A}/bus.jpg'), cv2.imread(f'{A}/zidane.jpg')

def make_frames(n=100, w=1920, h=1080):
    out = []
    zz = cv2.resize(zid, (640, 360)); bb = cv2.resize(bus, (135, 180))
    rng = np.random.default_rng(0)
    base = rng.integers(90, 130, (h, w, 3), dtype=np.uint8)
    for i in range(n):
        f = base.copy()
        x = 100 + int(900 * (0.5 - 0.5 * np.cos(2 * np.pi * i / n)))  # smooth back and forth
        f[200:560, x:x + 640] = zz
        f[800:980, 1600:1735] = bb   # small, distant
        out.append(f)
    return out

def make_pan_frames(n=100, w=1920, h=1080, amp=400):
    """Static scene, moving camera: a textured canvas (corners for optical
    flow) with the objects fixed in scene coordinates, and a view window that
    sweeps +-amp px sideways (up to ~25 px per frame) — what a panning gimbal
    looks like to the detector."""
    rng = np.random.default_rng(1)
    cw, ch = w + 2 * amp, h + 200
    canvas = np.full((ch, cw, 3), 100, np.uint8)
    for _ in range(900):  # random blocks = lots of trackable corners
        x, y = int(rng.integers(0, cw - 40)), int(rng.integers(0, ch - 40))
        bw, bh = int(rng.integers(8, 60)), int(rng.integers(8, 60))
        canvas[y:y + bh, x:x + bw] = rng.integers(40, 200)
    zz = cv2.resize(zid, (640, 360)); bb = cv2.resize(bus, (135, 180))
    canvas[300:660, amp + 500:amp + 1140] = zz
    canvas[850:1030, amp + 1500:amp + 1635] = bb
    out = []
    for i in range(n):
        ox = amp + int(amp * np.sin(2 * np.pi * i / n))
        out.append(canvas[100:100 + h, ox:ox + w].copy())
    return out

FRAMES = make_pan_frames() if os.environ.get('H_PAN') == '1' else make_frames()

class FakeCap:
    def __init__(self, fps=25):
        self.i, self.dt, self.t = 0, 1.0 / fps, time.time()
    def isOpened(self): return True
    def set(self, *a): return True
    def release(self): pass
    def read(self):
        self.t += self.dt
        d = self.t - time.time()
        if d > 0: time.sleep(d)
        f = FRAMES[self.i % len(FRAMES)]; self.i += 1
        return True, f.copy()

class Store:
    upserts = []  # (t, rows)
    deletes = 0

class Q:
    def __init__(self, op=None, payload=None): self.op, self.payload = op, payload
    def __getattr__(self, name):
        def f(*a, **k):
            if name == 'upsert': return Q('upsert', a[0])
            if name == 'delete': return Q('delete')
            if name == 'select': return Q('select')
            if name == 'update': return Q('update')
            return self
        return f
    @property
    def not_(self): return self
    def execute(self):
        if self.op == 'upsert':
            time.sleep(0.05)  # typical Supabase round trip
            Store.upserts.append((time.time(), self.payload))
        elif self.op == 'delete': Store.deletes += 1
        elif self.op == 'select' and os.environ.get('H_LOCK') == '1':
            return type('R', (), {'data': [{'track_id': -1, 'bbox': {'x': 0.83, 'y': 0.74, 'width': 0.07, 'height': 0.17}, 'object_class': 'unknown', 'confidence': 0}]})()
        return type('R', (), {'data': []})()

LOCK = os.environ.get('H_LOCK') == '1'
class FakeClient:
    def table(self, name):
        q = Q()
        if LOCK:
            orig = q.execute
            def ex():
                r = orig()
                r.data = [{'track_id': -1, 'bbox': {'x': 0.83, 'y': 0.74, 'width': 0.07, 'height': 0.17}, 'object_class': 'unknown', 'confidence': 0}]
                return r
            q.execute = ex
        return q

# H_WS=1: run a fake Supabase Realtime (Phoenix) websocket server locally, so
# the websocket path is exercised; otherwise the socket cannot connect and
# broadcasts go through the (fake) REST fallback.
WS_MESSAGES = []  # (t, decoded frame)
WS_URL = 'http://fake'
if os.environ.get('H_WS') == '1':
    import asyncio
    import websockets

    async def _phoenix(ws):
        if os.environ.get('H_WS_DROP') == '1':  # drop the first connection after 20 s
            asyncio.get_running_loop().call_later(20, lambda: asyncio.ensure_future(ws.close()))
        async for raw in ws:
            msg = json.loads(raw)
            WS_MESSAGES.append((time.time(), msg))
            if msg['event'] in ('phx_join', 'broadcast', 'heartbeat', 'phx_leave'):
                await asyncio.sleep(0.01)
                await ws.send(json.dumps({'topic': msg['topic'], 'event': 'phx_reply', 'ref': msg['ref'],
                                          'payload': {'status': 'ok', 'response': {}}}))

    def _serve():
        loop = asyncio.new_event_loop()
        async def main():
            async with websockets.serve(_phoenix, '127.0.0.1', 8765):
                await asyncio.Future()
        loop.run_until_complete(main())
    threading.Thread(target=_serve, daemon=True).start()
    WS_URL = 'http://127.0.0.1:8765'

os.environ.update(SUPABASE_URL=WS_URL, SUPABASE_SERVICE_ROLE_KEY='x',
                  MEDIAMTX_RTSP_URL='fake://stream', FLIGHT_SESSION_ID='test-session',
                  DETECTION_FPS='20', TRACKER_LOST_BUFFER='10', TRACK_TTL_SECONDS='0.2',
                  RANGE_PASS_INTERVAL_SECONDS='4.0', RANGE_TILE_COLS='2', RANGE_TILE_ROWS='2',
                  LOG_SUMMARY_SECONDS='10')
spec = importlib.util.spec_from_file_location('app', APP)
app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
app.create_client = lambda *a, **k: FakeClient()
BROADCASTS = []  # (t, body)
def fake_post(self, client, body):
    time.sleep(0.04)  # typical Realtime REST round trip
    BROADCASTS.append((time.time(), body))
    return 202
if hasattr(app, "Broadcaster"):  # older app.py versions have no broadcast
    app.Broadcaster._post = fake_post
app.open_capture = lambda url: FakeCap()
N = int(os.environ.get('H_STREAMS', '1'))
# Scale-to-zero checks: H_NO_STREAMS=1 (discovery finds nothing) or
# H_DEAD_STREAM=1 (an active flight whose RTSP stream cannot be opened).
if os.environ.get('H_NO_STREAMS') == '1' or os.environ.get('H_DEAD_STREAM') == '1':
    os.environ.pop('MEDIAMTX_RTSP_URL'); app.RTSP_URL = ''
    if os.environ.get('H_DEAD_STREAM') == '1':
        app.DetectionStore.live_streams = lambda self: [{'flight_session_id': 'dead', 'path': 'gone/1'}]
        app.open_capture = lambda url: None
    else:
        app.DetectionStore.live_streams = lambda self: []
if N > 1:
    os.environ.pop('MEDIAMTX_RTSP_URL'); app.RTSP_URL = ''; app.MAX_STREAMS = N
    app.DetectionStore.live_streams = lambda self: [{'flight_session_id': f's{i}', 'path': f'drone{i}/1'} for i in range(N)]
main_thread = threading.Thread(target=app.main, daemon=True)
main_thread.start()
t0 = time.time()
EXITED = []
def _watch():
    main_thread.join(); EXITED.append(time.time() - t0)
threading.Thread(target=_watch, daemon=True).start()
WAKE_AT = float(os.environ.get('H_WAKE_AT', '0') or 0)  # send GET /wake after N s
if WAKE_AT:
    time.sleep(WAKE_AT)
    print('WAKE', urllib.request.urlopen('http://127.0.0.1:8080/wake', timeout=2).read().decode(), f'at {time.time()-t0:.1f}s')
    time.sleep(max(0, SECONDS - WAKE_AT))
else:
    time.sleep(SECONDS)
print(f"MAIN {'exited after %.1fs' % EXITED[0] if EXITED else 'still running'}")
if not any(u[0] > t0 for u in Store.upserts) and not BROADCASTS and not WS_MESSAGES:
    sys.exit(0)
up = [u for u in Store.upserts if u[0] > t0 + 15]  # skip warm-up
span = max(1e-6, up[-1][0] - up[0][0]) if len(up) > 1 else 1
classes = collections.Counter(r['object_class'] for _, rows in up for r in rows)
ids = collections.Counter(r['track_id'] for _, rows in up for r in rows)
print(f'RESULT upserts/s={len(up)/span:.1f} rows/upsert={np.mean([len(r) for _,r in up]):.2f} '
      f'distinct_track_ids={len(ids)} classes={dict(classes.most_common(6))}')
# Track stability: distinct ids per class and how long an id lives (upserts
# it appears in). Fewer ids / longer lives = fewer identity switches.
if up:
    per_id = collections.Counter(r['track_id'] for _, rows in up for r in rows)
    cls_of = {r['track_id']: r['object_class'] for _, rows in up for r in rows}
    by_cls = collections.defaultdict(list)
    for tid, cnt in per_id.items():
        by_cls[cls_of[tid]].append(cnt)
    if os.environ.get('H_IDS_DETAIL') == '1':
        size = {}
        for _, rows in up:
            for r in rows:
                size.setdefault(r['track_id'], (round(r['bbox']['width'] * 640), round(r['bbox']['height'] * 360)))
        print('IDS_DETAIL', sorted(((cls_of[t], per_id[t], size[t]) for t in per_id), key=lambda x: -x[1])[:12])
    print('IDS ' + ' '.join(f"{c}:{len(v)} ids (median life {int(np.median(v))} frames)" for c, v in sorted(by_cls.items())))
bc = [b for b in BROADCASTS if b[0] > t0 + 15]
if bc:
    msgs = [m for _, body in bc for m in body['messages']]
    tr = [len(m['payload']['tracks']) for m in msgs]
    moving = [t for m in msgs for t in m['payload']['tracks'] if abs(t['vx']) > 0.01]
    print(f"BROADCAST msgs/s={len(bc)/max(1e-6, bc[-1][0]-bc[0][0]):.1f} tracks/msg={np.mean(tr):.2f} "
          f"topics={sorted({m['topic'] for m in msgs})} events={sorted({m['event'] for m in msgs})} "
          f"private={all(m['private'] for m in msgs)} moving_tracks={len(moving)} "
          f"locked_seen={any(t['locked'] for m in msgs for t in m['payload']['tracks'])}")
    print('SAMPLE', json.dumps(msgs[-1]['payload'])[:400])
else:
    print('BROADCAST (rest) none')
ws = [m for t, m in WS_MESSAGES if t > t0 + 15]
if ws:
    b = [m for m in ws if m['event'] == 'broadcast']
    joins = [m for _, m in WS_MESSAGES if m['event'] == 'phx_join']
    span = max(1e-6, ws[-1][0] - ws[0][0]) if False else max(1e-6, SECONDS - 15)
    print(f"WEBSOCKET broadcasts/s={len(b)/span:.1f} joins={len(joins)} "
          f"join_private={all(j['payload']['config']['private'] for j in joins)} "
          f"topics={sorted({m['topic'] for m in b})} "
          f"tracks/msg={np.mean([len(m['payload']['payload']['tracks']) for m in b]):.2f}")
try:
    h = json.load(urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2))
    print('HEALTH', json.dumps({k: h.get(k) for k in ('model', 'transport', 'db_writer', 'broadcast')}))
    for st in h['streams']: print('  ', st['path'], json.dumps(st.get('metrics')))
except Exception as e:
    print('HEALTH ERR', e)

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

FRAMES = make_frames()

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

os.environ.update(SUPABASE_URL='http://fake', SUPABASE_SERVICE_ROLE_KEY='x',
                  MEDIAMTX_RTSP_URL='fake://stream', FLIGHT_SESSION_ID='test-session',
                  DETECTION_FPS='20', TRACKER_LOST_BUFFER='10', TRACK_TTL_SECONDS='0.2',
                  RANGE_PASS_INTERVAL_SECONDS='4.0', RANGE_TILE_COLS='2', RANGE_TILE_ROWS='2',
                  LOG_SUMMARY_SECONDS='10')
spec = importlib.util.spec_from_file_location('app', APP)
app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
app.create_client = lambda *a, **k: FakeClient()
app.open_capture = lambda url: FakeCap()
N = int(os.environ.get('H_STREAMS', '1'))
if N > 1:
    os.environ.pop('MEDIAMTX_RTSP_URL'); app.RTSP_URL = ''; app.MAX_STREAMS = N
    app.DetectionStore.live_streams = lambda self: [{'flight_session_id': f's{i}', 'path': f'drone{i}/1'} for i in range(N)]
threading.Thread(target=app.main, daemon=True).start()
t0 = time.time(); time.sleep(SECONDS)
up = [u for u in Store.upserts if u[0] > t0 + 15]  # skip warm-up
span = max(1e-6, up[-1][0] - up[0][0]) if len(up) > 1 else 1
classes = collections.Counter(r['object_class'] for _, rows in up for r in rows)
ids = collections.Counter(r['track_id'] for _, rows in up for r in rows)
print(f'RESULT upserts/s={len(up)/span:.1f} rows/upsert={np.mean([len(r) for _,r in up]):.2f} '
      f'distinct_track_ids={len(ids)} classes={dict(classes.most_common(6))}')
try:
    h = json.load(urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2))
    print('HEALTH', json.dumps({k: h[k] for k in ('model', 'db_writer')}))
    for st in h['streams']: print('  ', st['path'], json.dumps(st.get('metrics')))
except Exception as e:
    print('HEALTH ERR', e)

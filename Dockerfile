# atlas-detector — continuous object detection/tracking on Atlas drone video.
#
# CPU only. YOLO26n runs through ONNX Runtime (~3x faster than PyTorch on CPU).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/mpl

# OpenCV + ffmpeg runtime bits (RTSP demuxing, H.264 decoding).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libgl1 libsm6 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch CPU wheels only — the CUDA build is several GB and useless here.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Bake the YOLO26n weights into the image so cold start does not download them.
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')" \
 && cp /app/yolo26n.pt /app/model.pt 2>/dev/null || true

# ONNX Runtime export of the same weights (dynamic shapes, so preprocessing and
# boxes match PyTorch exactly) — ~3x faster on CPU. app.py falls back to the
# .pt file if this is missing, but the build fails loudly if the export fails.
RUN python -c "from ultralytics import YOLO; YOLO('/app/yolo26n.pt').export(format='onnx', dynamic=True)" \
 && test -s /app/yolo26n.onnx

COPY app.py bench.py ./

EXPOSE 8080
CMD ["python", "app.py"]

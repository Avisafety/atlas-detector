# atlas-detector — continuous object detection/tracking on Atlas drone video.
#
# CPU only. YOLOv8n is small enough for 5–10 fps on a shared Fly CPU.
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

# Bake the YOLOv8n weights into the image so cold start does not download them.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
 && cp /app/yolov8n.pt /app/model.pt 2>/dev/null || true

# Bake the Grounding DINO (Swin-T) checkpoint too, so switching
# DETECTION_ENGINE=grounding_dino does not trigger a ~700 MB cold-start download.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection; \
ckpt='IDEA-Research/grounding-dino-tiny'; \
AutoProcessor.from_pretrained(ckpt); \
AutoModelForZeroShotObjectDetection.from_pretrained(ckpt)"

COPY app.py .

EXPOSE 8080# atlas-detector — continuous object detection/tracking on Atlas drone video.
#
# CPU only. YOLO26n is small enough for 5–10 fps on a shared Fly CPU.
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

# Bake the YOLOv8n weights into the image so cold start does not download them.
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')" \
 && cp /app/yolo26n.pt /app/model.pt 2>/dev/null || true

COPY app.py .

EXPOSE 8080
CMD ["python", "app.py"]

CMD ["python", "app.py"]

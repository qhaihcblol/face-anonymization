# Backend image cho FaceGuard.
# Build context PHẢI là repo root vì `ai_core` nằm ở root và pipeline.py import nó
# qua parents[4] (webapp/backend/app/processing/pipeline.py -> repo root).
#
# CPU-only image. Muốn dùng GPU: đổi base image sang một image có CUDA/cuDNN
# (vd nvidia/cuda:12.x-cudnn-runtime-ubuntu22.04 + cài python 3.12) và giữ
# onnxruntime-gpu trong requirements.txt, rồi chạy container với `--gpus all`.

FROM python:3.12-slim

# ffmpeg cho audio/video; libgl1 + libglib2.0-0 là runtime của opencv-python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài Python deps trước để tận dụng layer cache.
COPY requirements.txt ./
# Ảnh CPU-only: đổi onnxruntime-gpu -> onnxruntime. Bỏ dòng sed này nếu build ảnh GPU.
RUN sed -i 's/^onnxruntime-gpu.*/onnxruntime/' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# Mã nguồn. ai_core phải ở /app/ai_core để pipeline.py tìm thấy.
COPY ai_core/ ./ai_core/
COPY scripts/ ./scripts/
COPY webapp/backend/ ./webapp/backend/

# Model ONNX: KHÔNG bake vào image (nặng + cần Kaggle creds). Mount qua volume lúc
# chạy (xem docker-compose.yml). Nếu muốn bake luôn, bỏ comment 2 dòng dưới và
# truyền KAGGLE creds qua build secret/env:
# RUN pip install --no-cache-dir kagglehub \
#     && python scripts/download_onnx_files.py

WORKDIR /app/webapp/backend
EXPOSE 8000

# 1 worker: pipeline ONNX nạp 1 lần/process; nhiều worker = nhân RAM/VRAM.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

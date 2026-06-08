# FaceGuard — Face & Voice Anonymization

Nền tảng ẩn danh khuôn mặt và giọng nói cho video, gồm pipeline AI (`ai_core`) chạy
trên ONNX Runtime và một web app full‑stack (FastAPI + Next.js). Hỗ trợ 2 chế độ:

- **Ẩn danh video offline** — upload video, xử lý theo job và tải kết quả về.
- **Live camera (real‑time)** — stream webcam qua WebSocket, ẩn danh khuôn mặt theo từng frame.

Pipeline AI: phát hiện khuôn mặt (RetinaFace) → tracking → parsing (BiSeNet) →
ẩn danh (blur / pixelate / face‑swap BlendSwap) → phục hồi (GFPGAN), và ẩn danh
giọng nói (kNN‑VC: WavLM encoder + HiFi‑GAN vocoder, fallback sang DSP).

---

## Kiến trúc

```
face-anonymization/
├── ai_core/                 # Pipeline AI (ONNX): detection, parsing, swap, restore, voice
│   └── */onnx/              # Model ONNX (tải bằng scripts/download_onnx_files.py, KHÔNG commit)
├── scripts/
│   └── download_onnx_files.py
├── webapp/
│   ├── backend/             # FastAPI (REST + WebSocket), SQLAlchemy async, R2 storage
│   │   ├── app/             # main.py → app.main:app
│   │   └── .env             # cấu hình backend (KHÔNG commit)
│   └── frontend/            # Next.js 16 + React 19 (pnpm)
│       └── .env.local       # cấu hình frontend (KHÔNG commit)
├── requirements.txt
└── README.MD
```

| Thành phần | Công nghệ |
|-----------|-----------|
| AI core   | Python 3.12, onnxruntime(-gpu), OpenCV, librosa, scipy |
| Backend   | FastAPI, Uvicorn, SQLAlchemy async + asyncpg (PostgreSQL), JWT, Cloudflare R2 (S3 API) |
| Frontend  | Next.js 16, React 19, TailwindCSS, pnpm |

Frontend gọi backend qua **proxy server‑side** của Next.js (`/api/*`) — token được
gắn ở phía server, browser không bao giờ thấy URL backend trực tiếp.

---

## 1. Yêu cầu hệ thống

- **Python 3.12** (khuyến nghị dùng conda)
- **Node.js 24** + **pnpm** (qua corepack)
- **FFmpeg**
- **PostgreSQL** (local hoặc managed, ví dụ Neon/Supabase)
- **Cloudflare R2** bucket (hoặc storage tương thích S3) để lưu video
- GPU NVIDIA + CUDA/cuDNN (tùy chọn — để tăng tốc; mặc định chạy được trên CPU)

---

## 2. Cài đặt môi trường

```bash
git clone https://github.com/qhaihcblol/face-anonymization.git
cd face-anonymization

# Tạo env conda
conda create -p ./.conda python=3.12 -y
conda activate ./.conda

# Các framework hệ thống
conda install -c conda-forge ffmpeg -y
conda install -c conda-forge "nodejs=24.*" -y
corepack enable
corepack prepare pnpm@latest --activate
pnpm -v
```

### 2.1. Tải model ONNX

```bash
# Tạo thư mục chứa model
mkdir -p ai_core/face_detection/onnx \
         ai_core/face_parsing/onnx \
         ai_core/face_swapping/onnx \
         ai_core/face_restoration/onnx \
         ai_core/voice_anonymization/onnx

# Tải từ Kaggle dataset (quanghijr/video-anonymization-onnx)
pip install kagglehub
python scripts/download_onnx_files.py
```

> Cần đăng nhập Kaggle (`kaggle.json` trong `~/.kaggle/` hoặc biến môi trường
> `KAGGLE_USERNAME` / `KAGGLE_KEY`). Nếu model voice (`wavlm_encoder.onnx`,
> `hifigan_vocoder.onnx`) thiếu, pipeline tự fallback sang ẩn danh giọng bằng DSP.

### 2.2. Cài thư viện Python

```bash
pip install -r requirements.txt
```

> `requirements.txt` mặc định cài `onnxruntime-gpu`. **Nếu chạy CPU‑only**, hãy đổi
> sang `onnxruntime` (gỡ `onnxruntime-gpu` trước để tránh xung đột). Với GPU, đảm bảo
> CUDA/cuDNN khớp phiên bản onnxruntime, hoặc cài thêm:
> ```bash
> pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
>             nvidia-cufft-cu12 nvidia-curand-cu12
> ```

---

## 3. Cấu hình (.env)

### 3.1. Backend — `webapp/backend/.env`

```env
# --- Auth ---
SECRET_KEY=thay-bang-chuoi-ngau-nhien-dai            # bắt buộc; vd: openssl rand -hex 32
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# --- App ---
PROJECT_NAME=FaceGuard Backend API
API_PREFIX=/api
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000   # thêm domain frontend khi deploy

# --- Database (PostgreSQL) --- bắt buộc
DATABASE_URL=postgresql://user:pass@host:5432/faceguard

# --- Object storage (Cloudflare R2 / S3) ---
R2_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=faceguard-videos
R2_REGION=auto
R2_PRESIGN_EXPIRY_SECONDS=3600

# --- Giới hạn upload (tùy chọn) ---
VIDEO_MAX_UPLOAD_MB=2048
VIDEO_ALLOWED_EXTENSIONS=.mp4,.mov,.webm,.mkv,.avi

# --- Live camera (tùy chọn) ---
LIVE_JPEG_QUALITY=80              # 1-100, chất lượng frame trả về browser
LIVE_DETECT_INTERVAL=2            # chạy detector mỗi N frame, frame còn lại dùng tracker
LIVE_MAX_CONCURRENT_FRAMES=2      # giới hạn frame infer đồng thời (CPU-bound)
```

> `DATABASE_URL` chấp nhận `postgresql://`, `postgres://` hoặc `postgresql+asyncpg://`.
> Schema được tạo tự động lúc khởi động (`Base.metadata.create_all`).

### 3.2. Frontend — `webapp/frontend/.env.local`

```env
# URL gốc của API backend (dùng ở server-side proxy). Mặc định: http://127.0.0.1:8000/api
BACKEND_API_BASE_URL=http://127.0.0.1:8000/api
```

> Live camera dùng WebSocket; proxy tự suy ra URL WS từ `BACKEND_API_BASE_URL`
> (`http→ws`, `https→wss`). Khi deploy HTTPS, dùng URL `https://...` để tự động lên `wss`.

---

## 4. Chạy ở môi trường DEV

Mở **2 terminal** (đều `conda activate ./.conda` cho terminal backend).

### Terminal 1 — Backend (FastAPI, cổng 8000)

```bash
cd webapp/backend
fastapi dev app/main.py --host 0.0.0.0 --port 8000
# hoặc: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health check: http://127.0.0.1:8000/health
- API docs (Swagger): http://127.0.0.1:8000/docs

### Terminal 2 — Frontend (Next.js, cổng 3000)

```bash
cd webapp/frontend
pnpm install        # lần đầu
pnpm dev
```

- Mở app: http://localhost:3000

---

## 5. Deploy production

Phần deploy production đang được thiết kế lại. Các artifact deploy cũ đã được gỡ để
repo chỉ giữ phần lõi ứng dụng (`ai_core`, `webapp`) và tránh tài liệu trỏ vào một
topology chưa chốt.

Các ràng buộc vẫn cần giữ khi thiết kế deploy mới:

- Backend cần hostname công khai nếu dùng live camera, vì browser nối WebSocket tới
  `/api/live/ws`.
- `BACKEND_API_BASE_URL` của frontend phải trỏ tới URL API production, ví dụ
  `https://api.your-domain.com/api`.
- Backend vẫn cần PostgreSQL, R2/S3 storage, FFmpeg và model ONNX trong
  `ai_core/*/onnx/`.
- Khi chạy GPU, giữ số process backend thấp; mỗi process sẽ nạp một bộ ONNX riêng
  vào RAM/VRAM.

---

## 6. Test

```bash
# Test pipeline AI offline (cần video trong test_videos/)
python test_scripts/test_anonymize_video_full_pipeline.py
python test_scripts/test_anonymize_video_visual.py
python test_scripts/test_anonymize_video_swap.py
python test_scripts/test_anonymize_video_audio.py

# Test backend
cd webapp/backend && pytest
```

---

## 7. API tổng quan

| Prefix | Mô tả |
|--------|-------|
| `POST /api/auth/*`   | Đăng ký / đăng nhập (JWT) |
| `GET  /api/users/*`  | Thông tin người dùng |
| `*    /api/videos/*` | Upload, tạo job ẩn danh (edits), tải kết quả |
| `WS   /api/live/ws`  | Live camera real‑time (frontend cấp ticket qua route `/api/live/ticket`) |
| `GET  /health`       | Health check |

Chi tiết xem Swagger tại `/docs` khi backend đang chạy.

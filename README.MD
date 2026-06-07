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

## 5. Deploy production (tự host full-stack bằng Docker)

Deploy cả frontend + backend trên cùng 1 VM, sau 1 Nginx, dùng 2 subdomain. File sẵn
trong repo: [Dockerfile](Dockerfile) (backend), [webapp/frontend/Dockerfile](webapp/frontend/Dockerfile),
[docker-compose.yml](docker-compose.yml), [deploy/nginx.conf.example](deploy/nginx.conf.example).

### 5.1. Vì sao 2 subdomain

Backend **phải có hostname công khai** vì live camera để browser nối WebSocket *thẳng*
tới `wss://api.../api/live/ws` (route ticket nhúng URL này từ `BACKEND_API_BASE_URL`).
Không thể giấu backend sau địa chỉ nội bộ Docker. Topology:

- `app.your-domain.com` → frontend (`next start`, :3000)
- `api.your-domain.com` → backend (FastAPI, :8000) — có WebSocket + upload lớn
- `BACKEND_API_BASE_URL=https://api.your-domain.com/api`

### 5.2. Các bước

```bash
# 1. Lấy code + model + env lên VM
git clone https://github.com/qhaihcblol/face-anonymization.git
cd face-anonymization
pip install kagglehub && python scripts/download_onnx_files.py   # đổ vào ai_core/*/onnx (mount vào container)
nano webapp/backend/.env          # theo mẫu mục 3.1; CORS_ORIGINS chứa https://app.your-domain.com

# 2. Sửa BACKEND_API_BASE_URL trong docker-compose.yml (service frontend) -> domain api thật

# 3. Build & chạy
docker compose up -d --build
docker compose logs -f
curl http://127.0.0.1:8000/health     # -> {"status":"ok"}

# 4. Reverse proxy + TLS
sudo cp deploy/nginx.conf.example /etc/nginx/sites-available/faceguard
# sửa server_name (app/api), rồi:
sudo ln -s /etc/nginx/sites-available/faceguard /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d app.your-domain.com -d api.your-domain.com
```

> **CPU vs GPU:** ảnh backend mặc định CPU-only ([Dockerfile](Dockerfile) tự đổi
> `onnxruntime-gpu`→`onnxruntime`). Muốn GPU: dùng base image CUDA, bỏ dòng `sed`,
> bỏ comment khối `deploy.resources` trong compose, cài `nvidia-container-toolkit`.
>
> **Workers:** giữ `--workers 1` cho backend — mỗi worker nạp ONNX riêng (nhân RAM/VRAM).
> Scale bằng nhiều container + load balancer.

### 5.3. Checklist trước khi deploy

- [ ] `webapp/backend/.env`: `SECRET_KEY` mạnh, `DATABASE_URL`, R2 đầy đủ, `CORS_ORIGINS` chứa domain `app.*`
- [ ] `docker-compose.yml`: `BACKEND_API_BASE_URL` (service frontend) trỏ `https://api.your-domain.com/api`
- [ ] Đã chạy `download_onnx_files.py` (model ONNX có trong `ai_core/*/onnx/` — được mount vào container)
- [ ] PostgreSQL chạy và truy cập được; R2 bucket tồn tại, key có quyền read/write
- [ ] DNS: `app.*` và `api.*` cùng trỏ về VM; đã cấp TLS cho cả hai
- [ ] Nginx: `client_max_body_size` đủ lớn ở **cả hai** server block; forward WebSocket cho `/api/live/ws`

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

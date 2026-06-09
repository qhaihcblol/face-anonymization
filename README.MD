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

Riêng **upload video đi thẳng browser → R2** bằng presigned PUT (backend chỉ ký URL,
không trung chuyển byte), nên video lớn không bị giới hạn body‑size/timeout của
server; đổi lại bucket R2 cần bật **CORS cho PUT** (xem mục Deploy).

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

### 2.1. Cài thư viện Python

```bash
pip install -r requirements.txt
```

> `requirements.txt` cài sẵn `onnxruntime-gpu[cuda,cudnn]` — đã kèm CUDA + cuDNN runtime,
> không cần cài thêm gói `nvidia-*` thủ công. **Nếu chạy CPU‑only**, đổi dòng đó trong
> `requirements.txt` thành `onnxruntime`.

### 2.2. Tải model ONNX

```bash
# Tạo thư mục chứa model
mkdir -p ai_core/face_detection/onnx \
         ai_core/face_parsing/onnx \
         ai_core/face_swapping/onnx \
         ai_core/face_restoration/onnx \
         ai_core/voice_anonymization/onnx

# Tải từ Kaggle dataset (quanghijr/video-anonymization-onnx); kagglehub đã có trong requirements
python scripts/download_onnx_files.py
```

> Cần đăng nhập Kaggle (`kaggle.json` trong `~/.kaggle/` hoặc biến môi trường
> `KAGGLE_USERNAME` / `KAGGLE_KEY`). Nếu model voice (`wavlm_encoder.onnx`,
> `hifigan_vocoder.onnx`) thiếu, pipeline tự fallback sang ẩn danh giọng bằng DSP.

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
R2_PRESIGN_EXPIRY_SECONDS=3600        # thời hạn link tải (GET)
R2_UPLOAD_URL_EXPIRY_SECONDS=3600     # thời hạn link upload (PUT); nới rộng nếu upload file lớn

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

Topology khuyến nghị — tách 3 phần, mỗi phần ở đúng chỗ mạnh của nó:

```
 Browser ──HTTPS──> Frontend (Vercel)
   │                    │  proxy server-side (BACKEND_API_BASE_URL)
   │                    ▼
   │             Backend (GPU box: vast.ai / VPS) ──> PostgreSQL (Neon/Supabase, managed)
   │                    ▲                          └─> Cloudflare R2 (S3 API)
   ├── upload (PUT) & playback (GET) đi THẲNG ─────> Cloudflare R2
   └── live camera (WSS) đi thẳng ─────────────────> Backend
```

- **Frontend → Vercel** (serverless + CDN). Mọi call tới backend đi qua proxy server‑side.
- **Backend → GPU box** (vast.ai / VPS có GPU): process chạy lâu dài, giữ model ONNX
  trong VRAM, mở WebSocket → **không** deploy lên Vercel/serverless được.
- **DB (PostgreSQL managed) + file (R2)** đều nằm ngoài GPU box → đổi/mất instance
  không mất dữ liệu.
- **Upload đi thẳng browser → R2** (presigned PUT) → video lớn không qua backend.

> Ràng buộc bắt buộc:
> - Backend phải có **HTTPS** (frontend là https → live camera dùng `wss://`; `http`
>   thường bị chặn mixed‑content).
> - Chạy backend **1 process** (`--workers 1`): mỗi process nạp một bộ ONNX riêng vào
>   VRAM, và worker xử lý nền + state live là per‑process.

### 5.1. Backend lên vast.ai

**1) Tạo instance:** GPU NVIDIA ≥ 16GB VRAM (vd RTX 4090), disk ≥ 30GB, image có CUDA 12
+ cuDNN (vd `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` hoặc image PyTorch). Không
cần mở port inbound nếu dùng Cloudflare Tunnel (tunnel xuyên NAT).

**2) SSH vào** (vast.ai cấp sẵn lệnh): `ssh -p <PORT> root@<HOST>`

**3) Lấy code + cài môi trường + model:**
```bash
apt-get update && apt-get install -y git ffmpeg curl
git clone https://github.com/qhaihcblol/face-anonymization.git
cd face-anonymization

# conda (nếu image chưa có Python phù hợp)
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
bash /tmp/mc.sh -b -p ./.conda
./.conda/bin/conda install -p ./.conda python=3.12 -y

./.conda/bin/pip install -r requirements.txt   # onnxruntime-gpu[cuda,cudnn] + kagglehub

# model ONNX (xem mục 2.2):
mkdir -p ai_core/face_detection/onnx ai_core/face_parsing/onnx \
         ai_core/face_swapping/onnx ai_core/face_restoration/onnx \
         ai_core/voice_anonymization/onnx
./.conda/bin/python scripts/download_onnx_files.py
```

**4) Tạo `webapp/backend/.env`** (xem mục 3.1): `SECRET_KEY`, `DATABASE_URL` trỏ Postgres
managed, `R2_*` đầy đủ. `CORS_ORIGINS` không bắt buộc (call đi qua proxy server‑side).

**5) Chạy backend (tmux để sống qua SSH):**
```bash
tmux new -s backend
cd webapp/backend
../../.conda/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
# Ctrl+B rồi D để detach;  tmux attach -t backend để quay lại
```
Kiểm tra: `curl http://localhost:8000/health` → `{"status":"ok"}`.

### 5.2. Expose backend ra internet (Cloudflare Tunnel)

```bash
curl -sL -o ./.conda/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ./.conda/bin/cloudflared

tmux new -s tunnel
cloudflared tunnel --url http://localhost:8000     # → copy URL https://<...>.trycloudflare.com
# Ctrl+B rồi D
```
- Tunnel tự lo **TLS + WebSocket** (live camera chạy được qua `wss`), không cần mở port.
- **URL cố định?** Quick tunnel đổi URL mỗi lần restart. Muốn cố định → **named tunnel +
  domain riêng**: add domain vào Cloudflare, `cloudflared tunnel create`, route DNS
  `api.tenban.com` → tunnel (theo docs Cloudflare Tunnel).

### 5.3. Frontend lên Vercel

- Import repo, đặt **Root Directory = `webapp/frontend`** (Next.js tự nhận diện).
- **Settings → Environment Variables** (Production + Preview), thêm:
  ```
  BACKEND_API_BASE_URL = https://<tunnel-url>/api
  ```
  Nhớ đuôi `/api`. Đây là biến **server‑side** — **không** thêm tiền tố `NEXT_PUBLIC_`.
- **Redeploy** để env có hiệu lực (Vercel không tự áp env mới vào bản deploy cũ).

### 5.4. CORS cho R2 (bắt buộc — nếu thiếu, upload sẽ lỗi mạng)

Upload đi thẳng browser → R2 nên bucket phải cho phép PUT từ domain frontend.
**Cloudflare → R2 → bucket → Settings → CORS Policy:**
```json
[
  {
    "AllowedOrigins": ["https://your-app.vercel.app"],
    "AllowedMethods": ["PUT", "GET"],
    "AllowedHeaders": ["content-type"],
    "MaxAgeSeconds": 3600
  }
]
```

### 5.5. Lưu ý vận hành

- **Instance vast.ai có thể bị thu hồi** → không để dữ liệu trên đó (DB ở Neon, file ở
  R2 — đã đúng). Mất instance thì dựng lại theo các bước trên.
- **Quick tunnel đổi URL** mỗi lần restart → phải sửa lại env Vercel + redeploy. Giữ
  tiến trình `cloudflared` chạy liên tục (tmux), hoặc dùng named tunnel cho URL cố định.
- **Object mồ côi trên R2** (xin URL nhưng không upload xong): đặt **lifecycle rule** xóa
  object cũ dưới prefix `uploads/` sau N ngày.
- **Cold start**: model nạp lần đầu khi có request/live đầu tiên (vài chục giây).

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

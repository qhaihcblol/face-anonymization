# 🚀 FaceGuard Deployment Guide for Vast.ai

Hướng dẫn chi tiết deploy hệ thống FaceGuard lên nền tảng GPU cloud Vast.ai.

## 📋 Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                    Vast.ai GPU Instance                      │
│  ┌─────────────────┐         ┌───────────────────────────┐  │
│  │   Next.js       │  HTTP   │   FastAPI Backend         │  │
│  │   Frontend      │ ◄─────► │   + AI Core (ONNX)        │  │
│  │   Port: 3000    │         │   Port: 8000              │  │
│  └─────────────────┘         └─────────────┬─────────────┘  │
│                                            │                │
│                                  ┌─────────▼─────────────┐  │
│                                  │   PostgreSQL          │  │
│                                  │   (local or managed)  │  │
│                                  └───────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Cloudflare R2       │
              │   (Video Storage)     │
              └───────────────────────┘
```

## ⚡ Quick Start (5 phút)

### Bước 1: Chuẩn bị tài khoản & credentials

1. **Vast.ai**: Đăng ký tại https://vast.ai và nạp tối thiểu $5 credit
2. **Kaggle**: Lấy API key tại https://www.kaggle.com/settings
3. **Cloudflare R2**: Tạo bucket tại https://dash.cloudflare.com

### Bước 2: Chọn GPU instance trên Vast.ai

**Yêu cầu tối thiểu:**
- GPU: RTX 3090 (24GB VRAM) hoặc cao hơn (khuyến nghị RTX 4090/A100)
- RAM: ≥32GB
- Storage: ≥100GB NVMe
- CUDA: 12.x support

**Khuyến nghị:**
- RTX 4090 24GB @ ~$0.25-0.35/giờ
- A100 40GB @ ~$0.60-0.80/giờ (cho xử lý nhanh hơn)

### Bước 3: Tạo file .env

```bash
cd /workspace
cp .env.example .env
```

Chỉnh sửa `.env` với thông tin thực tế:

```bash
# Security
SECRET_KEY=$(openssl rand -hex 32)

# Database (dùng PostgreSQL container chạy cùng)
DATABASE_URL=postgresql+asyncpg://faceguard:faceguard_secret@localhost:5432/faceguard

# Cloudflare R2
R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
R2_BUCKET=faceguard-videos

# Kaggle
KAGGLE_USERNAME=your-username
KAGGLE_KEY=your-api-key
```

### Bước 4: Build Docker image

```bash
# Build image với GPU support
docker build --platform linux/amd64 -t faceguard:latest .

# Hoặc pull từ registry (sau khi push)
# docker pull your-registry/faceguard:latest
```

### Bước 5: Push lên Docker Hub hoặc Registry cá nhân

```bash
# Tag image
docker tag faceguard:latest your-dockerhub-username/faceguard:latest

# Push
docker push your-dockerhub-username/faceguard:latest
```

### Bước 6: Deploy trên Vast.ai

1. Vào https://cloud.vast.ai/create/
2. Tìm GPU phù hợp (RTX 3090/4090/A100)
3. Click "Rent" và cấu hình:
   - **Image**: `your-dockerhub-username/faceguard:latest`
   - **Disk**: 100GB+ NVMe
   - **Ports**: 
     - `8000` → Backend API
     - `3000` → Frontend
     - `5432` → PostgreSQL (nếu chạy trong container)
   - **Environment Variables**: Copy toàn bộ nội dung file `.env`
   - **Extra Args**: `--gpus all`

4. Click "Launch" và chờ instance khởi động (~2-5 phút)

### Bước 7: Truy cập ứng dụng

Sau khi instance chạy:
- **Frontend**: `http://<vast-ip>:3000`
- **Backend API**: `http://<vast-ip>:8000`
- **API Docs**: `http://<vast-ip>:8000/docs`
- **Health Check**: `http://<vast-ip>:8000/health`

---

## 🔧 Chi tiết các bước

### 1. Chuẩn bị ONNX Models

Models được tải tự động khi container khởi động lần đầu. Đảm bảo đã set Kaggle credentials:

```bash
# Test download models locally (optional)
python scripts/download_onnx_files.py
```

**Models bao gồm:**
- `retinaface_best.onnx` - Face detection
- `bisenet_resnet_34.onnx` - Face parsing
- `blendswap_256.onnx` - Face swapping
- `gfpgan_1.4.onnx` - Face restoration
- `wavlm_encoder.onnx` - Voice anonymization
- `hifigan_vocoder.onnx` - Voice synthesis

### 2. Cấu hình Database

**Option A: PostgreSQL trong cùng container (đơn giản)**

Dùng Docker Compose để chạy PostgreSQL cùng backend:

```bash
docker compose up -d postgres
```

**Option B: Managed PostgreSQL (production)**

Sử dụng dịch vụ managed DB:
- Supabase (miễn phí tier)
- Neon (serverless PostgreSQL)
- AWS RDS / Google Cloud SQL

Cập nhật `DATABASE_URL` trong `.env`:
```bash
DATABASE_URL=postgresql+asyncpg://user:password@host.supabase.co:5432/postgres
```

### 3. Cấu hình Cloudflare R2

1. Vào Cloudflare Dashboard > R2
2. Tạo bucket mới (ví dụ: `faceguard-videos`)
3. Tạo API Token với quyền `Object Read & Write`
4. Lấy thông tin:
   - Endpoint URL: `https://<account-id>.r2.cloudflarestorage.com`
   - Access Key ID
   - Secret Access Key

### 4. Build Optimization

**Giảm kích thước image:**
```bash
# Multi-stage build đã được tối ưu trong Dockerfile
# Image size cuối cùng: ~3-4GB

# Scan image cho security vulnerabilities
docker scout cve faceguard:latest
```

**Tăng tốc build với cache:**
```bash
docker build --cache-from your-username/faceguard:latest -t faceguard:latest .
```

---

## 🎯 Cấu hình Vast.ai tối ưu

### Template JSON cho Vast.ai CLI

```json
{
  "image": "your-username/faceguard:latest",
  "gpu_ram": 24,
  "min_ram": 32,
  "disk_space": 100,
  "cuda_max": 12.4,
  "reliability2": 0.5,
  "verified": true,
  "secure_boot": false,
  "direct_port_count": 3,
  "ports": "8000:8000 3000:3000 5432:5432",
  "env": {
    "SECRET_KEY": "your-secret-key",
    "DATABASE_URL": "postgresql+asyncpg://faceguard:faceguard_secret@localhost:5432/faceguard",
    "R2_ENDPOINT_URL": "https://xxx.r2.cloudflarestorage.com",
    "R2_ACCESS_KEY_ID": "xxx",
    "R2_SECRET_ACCESS_KEY": "xxx",
    "R2_BUCKET": "faceguard-videos",
    "KAGGLE_USERNAME": "xxx",
    "KAGGLE_KEY": "xxx",
    "CORS_ORIGINS": "http://localhost:3000,http://0.0.0.0:3000"
  },
  "onstart": "sleep 30 && curl http://localhost:8000/health",
  "runtype": "ssh jupyter lab"
}
```

### Deploy với Vast.ai CLI

```bash
# Cài đặt CLI
pip install vastai

# Login
vastai login

# Deploy với template
vastai create instance \
  --image your-username/faceguard:latest \
  --gpu-ram 24 \
  --ram 32 \
  --disk 100 \
  --ports 8000:8000 3000:3000 5432:5432 \
  --env-file .env \
  --onstart "sleep 30 && curl http://localhost:8000/health"
```

---

## 🔍 Monitoring & Debugging

### Kiểm tra logs

```bash
# SSH vào instance
ssh root@<vast-ip> -p <port>

# Xem logs container
docker logs faceguard-backend
docker logs faceguard-frontend

# Xem logs real-time
docker logs -f faceguard-backend
```

### Health checks

```bash
# Backend health
curl http://<vast-ip>:8000/health

# API docs
curl http://<vast-ip>:8000/docs

# Test video upload (cần auth token)
curl -X POST http://<vast-ip>:8000/api/videos/upload \
  -H "Authorization: Bearer <token>" \
  -F "file=@test.mp4"
```

### Performance monitoring

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Memory usage
docker stats

# Disk usage
df -h
```

---

## 🛡️ Security Best Practices

### 1. Bảo mật biến môi trường

- ✅ Dùng Vast.ai environment variables (encrypted)
- ✅ Không commit `.env` vào Git
- ✅ Rotate SECRET_KEY định kỳ
- ✅ Dùng IAM roles thay vì hard-coded credentials

### 2. Network security

```bash
# Chỉ mở ports cần thiết
# Firewall rules (trong Vast.ai console):
- 8000: Backend API (có authentication)
- 3000: Frontend (public)
- 5432: PostgreSQL (chỉ localhost nếu có thể)
```

### 3. Database security

```bash
# Dùng SSL connection
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db?sslmode=require

# Giới hạn database user permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE videos TO faceguard;
```

### 4. Rate limiting

Thêm rate limiting cho API endpoints (cấu hình trong backend):

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
```

---

## 💰 Cost Optimization

### Ước tính chi phí

| GPU | Giá/giờ | RAM | Storage | Tổng/tháng (24/7) |
|-----|---------|-----|---------|-------------------|
| RTX 3090 | $0.25 | 32GB | 100GB | ~$180 |
| RTX 4090 | $0.35 | 64GB | 200GB | ~$250 |
| A100 40GB | $0.70 | 64GB | 200GB | ~$500 |

### Tips giảm chi phí

1. **Pause instance** khi không dùng: `$0.02/GB/tháng` cho storage
2. **Dùng spot instances**: Giảm 50-70% giá (có thể bị preempt)
3. **Auto-scale**: Tách backend/frontend thành instances riêng
4. **Cache models**: Lưu ONNX models vào volume persistent

---

## 🚨 Troubleshooting

### Lỗi thường gặp

#### 1. "CUDA out of memory"

**Nguyên nhân**: GPU VRAM không đủ
**Giải pháp**:
- Giảm `LIVE_MAX_CONCURRENT_FRAMES=1`
- Upgrade lên GPU nhiều VRAM hơn (A100 80GB)
- Giảm resolution video đầu vào

#### 2. "Model not found"

**Nguyên nhân**: Kaggle download failed
**Giải pháp**:
```bash
# Manual download
docker exec -it <container> python /app/scripts/download_onnx_files.py

# Verify files exist
docker exec -it <container> ls -la /app/ai_core/*/onnx/
```

#### 3. "CORS error" từ frontend

**Giải pháp**: Thêm Vast.ai IP vào CORS_ORIGINS:
```bash
CORS_ORIGINS=http://localhost:3000,http://<vast-ip>:3000
```

#### 4. PostgreSQL connection refused

**Giải pháp**:
```bash
# Check if Postgres is running
docker ps | grep postgres

# Restart if needed
docker restart faceguard-db

# Check logs
docker logs faceguard-db
```

#### 5. Video processing chậm

**Tối ưu**:
- Dùng GPU mạnh hơn (A100 > RTX 4090 > RTX 3090)
- Giảm resolution video đầu vào
- Tăng `LIVE_DETECT_INTERVAL` (detect ít frame hơn)
- Dùng tensor cores: `ONNXRUNTIME_EP=CUDAExecutionProvider`

---

## 📈 Scaling Strategies

### Horizontal scaling

```
Load Balancer
     │
     ├── Instance 1 (GPU 1)
     ├── Instance 2 (GPU 2)
     └── Instance N (GPU N)
```

**Cách triển khai:**
1. Deploy nhiều instances với cùng Docker image
2. Dùng managed PostgreSQL shared
3. Shared R2 bucket cho storage
4. Load balancer (Cloudflare, NGINX, HAProxy)

### Vertical scaling

Upgrade instance specs:
- GPU: RTX 3090 → RTX 4090 → A100
- RAM: 32GB → 64GB → 128GB
- Storage: 100GB → 500GB NVMe

---

## 📝 Checklist trước khi deploy production

- [ ] Đã set SECRET_KEY mạnh (32+ ký tự random)
- [ ] Đã config Cloudflare R2 credentials
- [ ] Đã set Kaggle username/key
- [ ] DATABASE_URL trỏ đến production DB (không phải localhost)
- [ ] CORS_ORIGINS chỉ include domains production
- [ ] Đã test video upload & processing
- [ ] Đã test live camera streaming
- [ ] Đã enable HTTPS (qua Cloudflare Tunnel hoặc reverse proxy)
- [ ] Đã setup monitoring & alerting
- [ ] Đã backup plan cho database
- [ ] Đã estimate chi phí và set budget alert

---

## 🔗 Tài liệu tham khảo

- [Vast.ai Docs](https://docs.vast.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Deployment](https://nextjs.org/docs/deployment)
- [ONNX Runtime GPU](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [Cloudflare R2 Docs](https://developers.cloudflare.com/r2/)

---

## 🆘 Support

Có vấn đề? Mở issue tại GitHub repo hoặc liên hệ:
- Email: support@faceguard.io
- Discord: [link]
- Telegram: [link]

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Maintained by**: FaceGuard Team

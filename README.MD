git clone https://github.com/qhaihcblol/face-anonymization.git
cd face-anonymization

# Tạo env conda
conda create -p ./.conda python=3.12 -y
conda activate ./.conda

# Tải các framework cần thiết
conda install -c conda-forge ffmpeg -y
conda install -c conda-forge "nodejs=24.*" -y
corepack enable
corepack prepare pnpm@latest --activate
pnpm -v

# Tải các onnx
mkdir -p ai_core/face_detection/onnx
mkdir -p ai_core/face_parsing/onnx
mkdir -p ai_core/face_swapping/onnx
mkdir -p ai_core/face_restoration/onnx
mkdir -p ai_core/voice_anonymization/onnx

pip install kagglehub
python scripts/download_onnx_files.py

# Tải các thư viện cần thiết
<!-- pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
pip install onnxruntime-gpu -->

pip install scipy
pip install librosa
pip install opencv-python

pip install "fastapi[standard]"

pip install sqlalchemy alembic asyncpg
pip install bcrypt "python-jose[cryptography]"
pip install pytest pytest-asyncio
pip install boto3

# tạo .env ở backend và frontend
.env tại webapp/backend

# ấy pnpm build
cd webapp/frontend
pnpm 
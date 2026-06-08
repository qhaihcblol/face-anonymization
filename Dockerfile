# FaceGuard Production Dockerfile for Vast.ai
# Multi-stage build for GPU-accelerated AI inference + Next.js frontend

ARG PYTHON_VERSION=3.12
ARG NODE_VERSION=24

# ==============================================================================
# Stage 1: Python Backend with AI Core
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS python-backend

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Set Python symlinks
RUN ln -sf python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf python${PYTHON_VERSION} /usr/bin/python3 \
    && python --version

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code
WORKDIR /app
COPY ai_core/ /app/ai_core/
COPY webapp/backend/ /app/webapp/backend/
COPY scripts/ /app/scripts/

# Set PYTHONPATH to include backend app
ENV PYTHONPATH="/app/webapp/backend:/app"
ENV ONNXRUNTIME_EP="CUDAExecutionProvider,CPUExecutionProvider"

# ==============================================================================
# Stage 2: Node.js Frontend Builder
# ==============================================================================
FROM node:${NODE_VERSION}-alpine AS frontend-builder

WORKDIR /frontend

# Copy package files
COPY webapp/frontend/package.json webapp/frontend/pnpm-lock.yaml ./
COPY webapp/frontend/pnpm-workspace.yaml ./

# Install pnpm and dependencies
RUN npm install -g pnpm@latest \
    && pnpm install --frozen-lockfile

# Copy frontend source
COPY webapp/frontend/ ./

# Build Next.js application
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production
RUN pnpm build

# ==============================================================================
# Stage 3: Production Runtime
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS production

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Create non-root user for security
RUN groupadd -r faceguard && useradd -r -g faceguard faceguard

# Copy Python virtual environment from builder
COPY --from=python-backend /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY --from=python-backend /app/ai_core /app/ai_core
COPY --from=python-backend /app/webapp/backend /app/webapp/backend
COPY --from=python-backend /app/scripts /app/scripts

# Copy built frontend from builder
COPY --from=frontend-builder /frontend/.next /app/webapp/frontend/.next
COPY --from=frontend-builder /frontend/public /app/webapp/frontend/public
COPY --from=frontend-builder /frontend/node_modules /app/webapp/frontend/node_modules
COPY --from=frontend-builder /frontend/package.json /app/webapp/frontend/package.json
COPY --from=frontend-builder /frontend/next.config.mjs /app/webapp/frontend/next.config.mjs

# Set environment variables
ENV PYTHONPATH="/app/webapp/backend:/app"
ENV ONNXRUNTIME_EP="CUDAExecutionProvider,CPUExecutionProvider"
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create directories for uploads and temp files
RUN mkdir -p /app/outputs /app/temp /app/logs \
    && chown -R faceguard:faceguard /app \
    && chmod -R 755 /app

# Switch to non-root user
USER faceguard

# Expose ports
# 8000: FastAPI backend
# 3000: Next.js frontend
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint script
COPY --chown=faceguard:faceguard docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["all"]

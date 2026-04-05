FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_PATH=YatharthS/LuxTTS

WORKDIR /app

# 系统依赖：合并为单层，清理缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get clean

# 先复制 requirements，利用 Docker 缓存层
COPY requirements.txt .

# 先装 torch（指定 CUDA 版本），再装其余依赖
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt

# 复制项目代码（按变更频率排序：不常变的先复制）
COPY zipvoice/ ./zipvoice/
COPY api/ ./api/

# 创建运行时目录，添加非 root 用户
RUN mkdir -p models && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# PRESET_VOICES 格式: "id:label:/path/to/audio.wav,..."
# MODEL_PATH 可覆盖为本地路径或其他 HuggingFace repo
CMD ["python3", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]

# LuxTTS API 服务

<p align="center">
  <b>中文</b> | <a href="README.md">English</a>
</p>

<p align="center">
  <a href="https://github.com/ysharma3501/LuxTTS">
    <img src="https://img.shields.io/badge/上游项目-ysharma3501%2FLuxTTS-blue?logo=github" alt="上游项目">
  </a>
  &nbsp;
  <a href="https://huggingface.co/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-模型-FFD21E" alt="Hugging Face 模型">
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space">
  </a>
</p>

本项目是 [ysharma3501/LuxTTS](https://github.com/ysharma3501/LuxTTS) 的 fork 版本，专注于将 LuxTTS 的能力封装为基于 FastAPI 的 REST HTTP API。核心模型与推理代码保持不变，本 fork 在其基础上新增了 `api/` 服务层。

## 与上游的区别

上游项目提供 Python 库接口。本 fork 将其封装为可生产部署的 HTTP API 服务，支持通过网络从任意语言或工具调用 LuxTTS。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/voices` | 列出所有已注册音色 |
| `POST` | `/api/voices/clone` | 上传参考音频克隆音色 |
| `POST` | `/api/tts` | 语音合成，返回完整 WAV 文件 |
| `POST` | `/api/tts/stream` | 语音合成，返回流式 WAV 响应 |

完整 API 文档：[`docs/api.zh.md`](docs/api.zh.md)

## 快速开始

### 前置要求

- Python 3.10-3.12
- 支持 CUDA 的 GPU（推荐）
- RTX 50 系列（sm_120）：需要 PyTorch nightly 版本

### 使用 uv 安装（推荐）

```bash
git clone https://github.com/Atnagnohil/LuxTTS.git
cd LuxTTS

# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv
uv sync

# RTX 50 系列（5060、5070、5080、5090）或更新的 GPU
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 启动服务
./start_server.sh
```

### 使用 pip 安装

```bash
git clone https://github.com/Atnagnohil/LuxTTS.git
cd LuxTTS
pip install -r requirements.txt

# RTX 50 系列或更新的 GPU
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 启动服务
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Docker（备选方案）

需要安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

```bash
git clone https://github.com/Atnagnohil/LuxTTS.git
cd LuxTTS

# 构建镜像
docker build -t luxtts-api .

# 启动服务（模型从 HuggingFace 自动下载）
docker run --gpus all -p 8000:8000 luxtts-api
```

挂载本地模型目录，避免重复下载：

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_PATH=/models/LuxTTS \
  -v /your/local/models:/models \
  luxtts-api
```

启动时预加载预设音色：

```bash
docker run --gpus all -p 8000:8000 \
  -e PRESET_VOICES="alice:爱丽丝:/voices/alice.wav,bob:鲍勃:/voices/bob.wav" \
  -v /your/local/voices:/voices \
  luxtts-api
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | 自动从缓存检测或下载 | HuggingFace repo ID（`YatharthS/LuxTTS`）或本地模型路径 |
| `PRESET_VOICES` | _(空)_ | 预设音色，格式：`id:显示名:/path/to/audio.wav,...` |
| `TRANSFORMERS_OFFLINE` | `0` | 设为 `1` 禁用 HuggingFace 下载 |
| `DEVICE` | `cuda` | 使用的设备：`cuda`、`cpu` 或 `mps` |

### 配置文件

在项目根目录创建 `.env` 文件：

```bash
# 可选：禁用 HuggingFace 下载（仅使用缓存模型）
TRANSFORMERS_OFFLINE=1

# 可选：指定模型路径（不设置则自动检测）
# MODEL_PATH=/path/to/your/model

# 可选：强制使用 CPU 模式
# DEVICE=cpu
```

## GPU 兼容性

| GPU 系列 | PyTorch 版本 | 安装命令 |
|----------|-------------|---------|
| RTX 50（5060、5070、5080、5090） | Nightly（sm_120） | `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128` |
| RTX 40（4090、4080 等） | 稳定版或 Nightly | `pip install torch` 或 nightly |
| RTX 30（3090、3080 等） | 稳定版 | `pip install torch` |
| RTX 20、GTX 16 系列 | 稳定版 | `pip install torch` |

项目会自动检测您的 GPU 并使用相应的计算能力。RTX 50 系列需要 PyTorch nightly 版本，因为稳定版尚不支持 sm_120 架构。

## 使用示例

```python
import httpx

# 克隆音色
with open("reference.wav", "rb") as f:
    httpx.post("http://localhost:8000/api/voices/clone",
        data={"voice_id": "alice", "label": "爱丽丝", "duration": "5", "rms": "0.01"},
        files={"audio": f})

# 语音合成
resp = httpx.post("http://localhost:8000/api/tts", json={
    "text": "你好，世界！",
    "voice_id": "alice"
})
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

服务启动后，可访问 `http://localhost:8000/docs` 查看交互式 API 文档（Swagger UI）。

## 关于上游项目

LuxTTS 是一个基于 ZipVoice 架构的轻量级 TTS 模型，主要特性：
- 媲美 10 倍体量模型的音色克隆效果
- 48 kHz 高清语音输出
- GPU 上达到 150 倍实时速度
- 显存占用不超过 1 GB

模型详情与原始 Python 接口请参见[上游仓库](https://github.com/ysharma3501/LuxTTS)。

## 许可证

Apache-2.0，详见 [LICENSE](LICENSE)。

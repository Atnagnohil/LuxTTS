# LuxTTS API Server

<p align="center">
  <a href="README.zh.md">中文</a> | <b>English</b>
</p>

<p align="center">
  <a href="https://github.com/ysharma3501/LuxTTS">
    <img src="https://img.shields.io/badge/Upstream-ysharma3501%2FLuxTTS-blue?logo=github" alt="Upstream">
  </a>
  &nbsp;
  <a href="https://huggingface.co/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model">
  </a>
  &nbsp;
  <a href="https://huggingface.co/spaces/YatharthS/LuxTTS">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space">
  </a>
</p>

This is a fork of [ysharma3501/LuxTTS](https://github.com/ysharma3501/LuxTTS), focused on exposing LuxTTS capabilities as a REST HTTP API built with FastAPI. The core model and inference code are unchanged — this fork adds an `api/` layer on top.

## What's different from upstream

The upstream project provides a Python library interface. This fork wraps it in a production-ready HTTP API server, so you can call LuxTTS over the network from any language or tool.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/voices` | List all registered voices |
| `POST` | `/api/voices/clone` | Clone a voice from a reference audio file |
| `POST` | `/api/tts` | Synthesize speech, returns a complete WAV file |
| `POST` | `/api/tts/stream` | Synthesize speech, returns a streaming WAV response |

Full API documentation: [`docs/api.md`](docs/api.md)

## Quick Start

### Prerequisites

- Python 3.10-3.12
- CUDA-capable GPU (recommended)
- For RTX 50 series (sm_120): PyTorch nightly build required

### Installation with uv (Recommended)

```bash
git clone https://github.com/Atnagnohil/LuxTTS.git
cd LuxTTS

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv sync

# For RTX 50 series (5060, 5070, 5080, 5090) or newer GPUs
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Start the server
./start_server.sh
```

### Installation with pip

```bash
git clone https://github.com/Atnagnohil/LuxTTS.git
cd LuxTTS
pip install -r requirements.txt

# For RTX 50 series or newer GPUs
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Start the server
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Docker (Alternative)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
git clone https://github.com/Atnagnohil/LuxTTS.git
cd LuxTTS

# Build the image
docker build -t luxtts-api .

# Start the server (model downloaded from HuggingFace automatically)
docker run --gpus all -p 8000:8000 luxtts-api
```

Mount a local model directory to avoid re-downloading:

```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_PATH=/models/LuxTTS \
  -v /your/local/models:/models \
  luxtts-api
```

Pre-load preset voices at startup:

```bash
docker run --gpus all -p 8000:8000 \
  -e PRESET_VOICES="alice:Alice:/voices/alice.wav,bob:Bob:/voices/bob.wav" \
  -v /your/local/voices:/voices \
  luxtts-api
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | Auto-detect from cache or download | HuggingFace repo ID (`YatharthS/LuxTTS`) or local model path |
| `PRESET_VOICES` | _(empty)_ | Preset voices: `id:label:/path/to/audio.wav,...` |
| `TRANSFORMERS_OFFLINE` | `0` | Set to `1` to disable HuggingFace downloads |
| `DEVICE` | `cuda` | Device to use: `cuda`, `cpu`, or `mps` |

### Configuration File

Create a `.env` file in the project root:

```bash
# Optional: Disable HuggingFace downloads (use cached models only)
TRANSFORMERS_OFFLINE=1

# Optional: Specify model path (auto-detected if not set)
# MODEL_PATH=/path/to/your/model

# Optional: Force CPU mode
# DEVICE=cpu
```

## GPU Compatibility

| GPU Series | PyTorch Version | Installation |
|------------|----------------|--------------|
| RTX 50 (5060, 5070, 5080, 5090) | Nightly (sm_120) | `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128` |
| RTX 40 (4090, 4080, etc.) | Stable or Nightly | `pip install torch` or nightly |
| RTX 30 (3090, 3080, etc.) | Stable | `pip install torch` |
| RTX 20, GTX 16 series | Stable | `pip install torch` |

The project automatically detects your GPU and uses the appropriate compute capability. For RTX 50 series, PyTorch nightly is required as stable releases don't yet support sm_120 architecture.

## Example Usage

```python
import httpx

# Clone a voice
with open("reference.wav", "rb") as f:
    httpx.post("http://localhost:8000/api/voices/clone",
        data={"voice_id": "alice", "label": "Alice", "duration": "5", "rms": "0.01"},
        files={"audio": f})

# Synthesize speech
resp = httpx.post("http://localhost:8000/api/tts", json={
    "text": "Hello, world!",
    "voice_id": "alice"
})
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs` once the server is running.

## Original Project

LuxTTS is a lightweight ZipVoice-based TTS model with:
- SOTA voice cloning quality
- 48 kHz audio output
- 150x realtime speed on GPU
- Under 1 GB VRAM

For model details and the original Python interface, see the [upstream repository](https://github.com/ysharma3501/LuxTTS).

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.

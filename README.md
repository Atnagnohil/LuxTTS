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

### Docker (Recommended)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
git clone https://github.com/ysharma3501/LuxTTS.git
cd LuxTTS

# Build the image
docker build -t luxtts-api .

# Start the server (model downloaded from HuggingFace automatically)
docker run --gpus all -p 8000:8000 \
  -e MODEL_PATH=YatharthS/LuxTTS \
  luxtts-api
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
  -e MODEL_PATH=YatharthS/LuxTTS \
  -e PRESET_VOICES="alice:Alice:/voices/alice.wav,bob:Bob:/voices/bob.wav" \
  -v /your/local/voices:/voices \
  luxtts-api
```

### Local

```bash
git clone https://github.com/ysharma3501/LuxTTS.git
cd LuxTTS
pip install -r requirements.txt

MODEL_PATH=YatharthS/LuxTTS uvicorn api.server:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/luxtts.pt` | HuggingFace repo ID or local model path |
| `PRESET_VOICES` | _(empty)_ | Preset voices: `id:label:/path/to/audio.wav,...` |

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

# LuxTTS API Reference

<p align="center">
  <a href="api.zh.md">中文</a> | <b>English</b>
</p>

Base URL: `http://localhost:8000`

All endpoints are prefixed with `/api`. Error responses use the format `{"detail": "<message>"}`.

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## GET /api/voices

List all registered voices (both preset and cloned).

### Response `200`

```json
{
  "voices": [
    {
      "voice_id": "alice",
      "label": "Alice",
      "type": "preset"
    }
  ],
  "total": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `voices` | array | List of voice objects |
| `voices[].voice_id` | string | Unique voice identifier |
| `voices[].label` | string | Display name |
| `voices[].type` | string | `"preset"` or `"clone"` |
| `total` | integer | Always equals `len(voices)` |

### Example

```bash
curl http://localhost:8000/api/voices
```

---

## POST /api/voices/clone

Clone a voice from a reference audio file. The audio is encoded and cached in memory for subsequent TTS requests.

### Request

`multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `voice_id` | string | yes | — | Unique ID. Only `[a-zA-Z0-9_-]` allowed. |
| `audio` | file | yes | — | Reference audio file (WAV, MP3, FLAC, etc.) |
| `label` | string | no | `""` | Display name |
| `duration` | float | no | `5.0` | Reference audio duration in seconds (1.0–30.0) |
| `rms` | float | no | `0.001` | Target RMS loudness (0.0001–0.1) |

### Response `200`

```json
{
  "voice_id": "alice",
  "message": "Voice cloned successfully"
}
```

### Error Responses

| Status | Condition |
|--------|-----------|
| `422` | `voice_id` contains invalid characters, or the audio file cannot be decoded |

### Example

```bash
curl -X POST http://localhost:8000/api/voices/clone \
  -F "voice_id=alice" \
  -F "label=Alice" \
  -F "duration=5" \
  -F "rms=0.01" \
  -F "audio=@reference.wav"
```

```python
import httpx

with open("reference.wav", "rb") as f:
    resp = httpx.post(
        "http://localhost:8000/api/voices/clone",
        data={"voice_id": "alice", "label": "Alice", "duration": "5", "rms": "0.01"},
        files={"audio": ("reference.wav", f, "audio/wav")},
    )
print(resp.json())
# {"voice_id": "alice", "message": "Voice cloned successfully"}
```

---

## POST /api/tts

Synthesize speech from text. Returns a complete 48 kHz WAV file.

### Request Body (JSON)

| Field | Type | Required | Default | Constraints | Description |
|-------|------|----------|---------|-------------|-------------|
| `text` | string | yes | — | non-empty | Text to synthesize |
| `voice_id` | string | yes | — | `[a-zA-Z0-9_-]` | Voice to use |
| `num_steps` | integer | no | `4` | 1–20 | Inference steps (more = better quality, slower) |
| `t_shift` | float | no | `0.9` | 0.1–1.0 | Sampling temperature (higher = better quality, more errors) |
| `speed` | float | no | `1.0` | 0.5–2.0 | Speech speed multiplier |
| `guidance_scale` | float | no | `3.0` | > 0 | Classifier-free guidance scale |

### Response `200`

Binary WAV audio (`Content-Type: audio/wav`), 48 kHz PCM.

### Error Responses

| Status | Condition |
|--------|-----------|
| `404` | `voice_id` not found — clone it first via `POST /api/voices/clone` |
| `422` | Invalid request parameters |
| `500` | Inference error (e.g. CUDA out of memory) |

### Example

```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_id": "alice"}' \
  --output output.wav
```

```python
import httpx

resp = httpx.post("http://localhost:8000/api/tts", json={
    "text": "Hello, world!",
    "voice_id": "alice",
    "num_steps": 4,
    "t_shift": 0.9,
    "speed": 1.0,
})
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

---

## POST /api/tts/stream

Synthesize speech and stream the audio back as a WAV response. Uses 24 kHz smooth mode. Useful for low-latency playback — you can start playing before the full response is received.

### Request Body (JSON)

Same fields as `POST /api/tts`.

### Response `200`

Streaming WAV audio (`Content-Type: audio/wav`), 24 kHz PCM. Concatenating all chunks produces a complete valid WAV file.

### Error Responses

| Status | Condition |
|--------|-----------|
| `404` | `voice_id` not found |
| `422` | Invalid request parameters |
| `500` | Inference error |

### Example

```bash
curl -X POST http://localhost:8000/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_id": "alice"}' \
  --output stream_output.wav
```

```python
import httpx

with httpx.stream("POST", "http://localhost:8000/api/tts/stream", json={
    "text": "Hello, world!",
    "voice_id": "alice",
}) as r:
    with open("stream_output.wav", "wb") as f:
        for chunk in r.iter_bytes():
            f.write(chunk)
```

---

## Notes

- GPU inference is serialized with a global lock to prevent OOM errors on concurrent requests.
- Preset voices (configured via `PRESET_VOICES` env var) are lazy-loaded on first use and cached.
- Cloned voices are encoded immediately on upload and cached in memory for the server lifetime.

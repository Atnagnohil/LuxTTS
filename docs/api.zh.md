# LuxTTS API 接口文档

<p align="center">
  <b>中文</b> | <a href="api.md">English</a>
</p>

Base URL：`http://localhost:8000`

所有接口均以 `/api` 为前缀。错误响应格式统一为 `{"detail": "<错误信息>"}`。

交互式文档（Swagger UI）：`http://localhost:8000/docs`

---

## GET /api/voices

列出所有已注册的音色（包括预设音色和克隆音色）。

### 响应 `200`

```json
{
  "voices": [
    {
      "voice_id": "alice",
      "label": "爱丽丝",
      "type": "preset"
    }
  ],
  "total": 1
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `voices` | array | 音色对象列表 |
| `voices[].voice_id` | string | 音色唯一标识符 |
| `voices[].label` | string | 显示名称 |
| `voices[].type` | string | `"preset"`（预设）或 `"clone"`（克隆） |
| `total` | integer | 始终等于 `len(voices)` |

### 示例

```bash
curl http://localhost:8000/api/voices
```

---

## POST /api/voices/clone

上传参考音频文件克隆音色。音频编码后缓存于内存，供后续 TTS 请求使用。

### 请求

`multipart/form-data`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `voice_id` | string | 是 | — | 音色唯一 ID，仅允许 `[a-zA-Z0-9_-]` |
| `audio` | file | 是 | — | 参考音频文件（WAV、MP3、FLAC 等） |
| `label` | string | 否 | `""` | 显示名称 |
| `duration` | float | 否 | `5.0` | 参考音频使用时长（秒，1.0–30.0） |
| `rms` | float | 否 | `0.001` | 目标 RMS 响度（0.0001–0.1） |

### 响应 `200`

```json
{
  "voice_id": "alice",
  "message": "Voice cloned successfully"
}
```

### 错误响应

| 状态码 | 触发条件 |
|--------|----------|
| `422` | `voice_id` 包含非法字符，或音频文件无法解码 |

### 示例

```bash
curl -X POST http://localhost:8000/api/voices/clone \
  -F "voice_id=alice" \
  -F "label=爱丽丝" \
  -F "duration=5" \
  -F "rms=0.01" \
  -F "audio=@reference.wav"
```

```python
import httpx

with open("reference.wav", "rb") as f:
    resp = httpx.post(
        "http://localhost:8000/api/voices/clone",
        data={"voice_id": "alice", "label": "爱丽丝", "duration": "5", "rms": "0.01"},
        files={"audio": ("reference.wav", f, "audio/wav")},
    )
print(resp.json())
# {"voice_id": "alice", "message": "Voice cloned successfully"}
```

---

## POST /api/tts

文本转语音，返回完整的 48 kHz WAV 文件。

### 请求体（JSON）

| 字段 | 类型 | 必填 | 默认值 | 约束 | 说明 |
|------|------|------|--------|------|------|
| `text` | string | 是 | — | 非空 | 要合成的文本 |
| `voice_id` | string | 是 | — | `[a-zA-Z0-9_-]` | 使用的音色 ID |
| `num_steps` | integer | 否 | `4` | 1–20 | 推理步数（越大质量越好，速度越慢） |
| `t_shift` | float | 否 | `0.9` | 0.1–1.0 | 采样温度（越高质量越好，但发音错误率可能上升） |
| `speed` | float | 否 | `1.0` | 0.5–2.0 | 语速倍率 |
| `guidance_scale` | float | 否 | `3.0` | > 0 | 无分类器引导强度 |

### 响应 `200`

二进制 WAV 音频（`Content-Type: audio/wav`），48 kHz PCM。

### 错误响应

| 状态码 | 触发条件 |
|--------|----------|
| `404` | `voice_id` 不存在，请先通过 `POST /api/voices/clone` 注册 |
| `422` | 请求参数不合法 |
| `500` | 推理错误（如 CUDA 显存不足） |

### 示例

```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，世界！", "voice_id": "alice"}' \
  --output output.wav
```

```python
import httpx

resp = httpx.post("http://localhost:8000/api/tts", json={
    "text": "你好，世界！",
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

文本转语音，以流式方式返回 WAV 响应。使用 24 kHz 平滑模式，适合低延迟播放场景——无需等待完整音频生成即可开始播放。

### 请求体（JSON）

字段与 `POST /api/tts` 相同。

### 响应 `200`

流式 WAV 音频（`Content-Type: audio/wav`），24 kHz PCM。将所有分块拼接后可得到完整合法的 WAV 文件。

### 错误响应

| 状态码 | 触发条件 |
|--------|----------|
| `404` | `voice_id` 不存在 |
| `422` | 请求参数不合法 |
| `500` | 推理错误 |

### 示例

```bash
curl -X POST http://localhost:8000/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "你好，世界！", "voice_id": "alice"}' \
  --output stream_output.wav
```

```python
import httpx

with httpx.stream("POST", "http://localhost:8000/api/tts/stream", json={
    "text": "你好，世界！",
    "voice_id": "alice",
}) as r:
    with open("stream_output.wav", "wb") as f:
        for chunk in r.iter_bytes():
            f.write(chunk)
```

---

## 注意事项

- GPU 推理使用全局锁串行化，防止并发请求导致显存溢出（OOM）。
- 预设音色（通过 `PRESET_VOICES` 环境变量配置）在首次使用时懒加载并缓存。
- 克隆音色在上传时立即编码，结果缓存于内存，服务重启后失效。

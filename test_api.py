#!/usr/bin/env python3
"""
LuxTTS API 完整测试脚本 — 直接调用四个接口
用法: python3 test_api.py --audio 爱丽丝.mp3 --text "你好，这是语音合成测试"
"""
import argparse
import sys
import os
import httpx

BASE_URL = "http://localhost:8000"

def ok(msg):  print(f"  ✓ {msg}")
def err(msg): print(f"  ✗ {msg}"); sys.exit(1)


def test_list_voices(base: str):
    print("\n[1] GET /api/voices")
    r = httpx.get(f"{base}/api/voices", timeout=10)
    if r.status_code != 200:
        err(f"status={r.status_code} body={r.text}")
    data = r.json()
    ok(f"total={data['total']}  voices={[v['voice_id'] for v in data['voices']]}")
    return data


def test_clone_voice(base: str, audio_path: str, voice_id: str):
    print(f"\n[2] POST /api/voices/clone  (audio={audio_path}, voice_id={voice_id})")
    with open(audio_path, "rb") as f:
        suffix = os.path.splitext(audio_path)[1].lstrip(".")
        mime = "audio/mpeg" if suffix == "mp3" else f"audio/{suffix}"
        r = httpx.post(
            f"{base}/api/voices/clone",
            data={"voice_id": voice_id, "label": "测试音色", "duration": "5", "rms": "0.001"},
            files={"audio": (os.path.basename(audio_path), f, mime)},
            timeout=60,
        )
    if r.status_code != 200:
        err(f"status={r.status_code} body={r.text}")
    ok(f"克隆成功: {r.json()}")


def test_tts(base: str, voice_id: str, text: str, output: str = "output_tts.wav"):
    print(f"\n[3] POST /api/tts  (voice_id={voice_id})")
    r = httpx.post(
        f"{base}/api/tts",
        json={"text": text, "voice_id": voice_id, "num_steps": 4, "t_shift": 0.9, "speed": 1.0},
        timeout=120,
    )
    if r.status_code != 200:
        err(f"status={r.status_code} body={r.text}")
    if r.content[:4] != b"RIFF":
        err("响应不是合法 WAV")
    with open(output, "wb") as f:
        f.write(r.content)
    ok(f"WAV 已保存: {output}  ({len(r.content)} bytes)")


def test_tts_stream(base: str, voice_id: str, text: str, output: str = "output_stream.wav"):
    print(f"\n[4] POST /api/tts/stream  (voice_id={voice_id})")
    chunks = []
    with httpx.stream(
        "POST",
        f"{base}/api/tts/stream",
        json={"text": text, "voice_id": voice_id, "num_steps": 4, "t_shift": 0.9, "speed": 1.0},
        timeout=120,
    ) as r:
        if r.status_code != 200:
            err(f"status={r.status_code}")
        for chunk in r.iter_bytes():
            chunks.append(chunk)
    data = b"".join(chunks)
    if data[:4] != b"RIFF":
        err("流式响应拼接后不是合法 WAV")
    with open(output, "wb") as f:
        f.write(data)
    ok(f"流式 WAV 已保存: {output}  ({len(data)} bytes, {len(chunks)} chunks)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--audio", default="爱丽丝.mp3", help="参考音频路径")
    parser.add_argument("--voice-id", default="test_voice")
    parser.add_argument("--text", default="你好，这是一段语音合成测试。")
    args = parser.parse_args()

    print(f"目标服务: {args.base_url}")

    test_list_voices(args.base_url)
    test_clone_voice(args.base_url, args.audio, args.voice_id)
    test_tts(args.base_url, args.voice_id, args.text)
    test_tts_stream(args.base_url, args.voice_id, args.text)

    print("\n全部测试通过 ✓")


if __name__ == "__main__":
    main()

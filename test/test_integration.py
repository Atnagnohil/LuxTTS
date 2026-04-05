"""
Integration tests for LuxTTS API Server (Task 7).

Covers:
  7.1 - Full chain integration tests: clone → list → TTS → stream TTS
  7.2 - Property 2: clone voice queryable via HTTP layer (Property-Based)

Uses starlette.testclient.TestClient (synchronous) with a test app that
bypasses lifespan to avoid real model loading.
"""
import io
import struct
from unittest.mock import MagicMock

import pytest
import soundfile as sf
import torch
from fastapi import FastAPI
from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.testclient import TestClient

from api.routes import router
from api.voice_store import VoiceStore

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ENCODE_DICT = {
    "prompt_tokens": torch.zeros(10),
    "prompt_features_lens": torch.tensor([10]),
    "prompt_features": torch.zeros(1, 10, 80),
    "prompt_rms": 0.001,
}


def make_mock_tts() -> MagicMock:
    """Return a MagicMock that mimics LuxTTS interface."""
    mock_tts = MagicMock()
    mock_tts.encode_prompt.return_value = SAMPLE_ENCODE_DICT
    # generate_speech returns 24000 samples of silence (1 second at 24kHz)
    mock_tts.generate_speech.return_value = torch.zeros(24000)
    return mock_tts


def make_test_app(mock_tts=None, voices: VoiceStore | None = None) -> FastAPI:
    """Build a minimal FastAPI app with mocked LuxTTS and pre-configured VoiceStore.

    Bypasses lifespan to avoid real model loading.
    """
    app = FastAPI()
    app.include_router(router, prefix="/api")

    if mock_tts is None:
        mock_tts = make_mock_tts()

    if voices is None:
        voices = VoiceStore(tts=mock_tts)

    app.state.tts = mock_tts
    app.state.voices = voices
    return app


def make_wav_bytes(num_samples: int = 24000, sample_rate: int = 24000) -> bytes:
    """Generate a valid WAV file in memory using soundfile."""
    buf = io.BytesIO()
    audio = torch.zeros(num_samples).numpy()
    sf.write(buf, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 7.1 — Integration tests: full chain
# ---------------------------------------------------------------------------

class TestFullChainIntegration:
    """Integration tests covering the complete API chain."""

    def test_clone_then_list_voices(self):
        """Clone a voice, then verify it appears in GET /api/voices."""
        app = make_test_app()
        wav_bytes = make_wav_bytes()

        with TestClient(app) as client:
            # Step 1: Clone a voice
            resp = client.post(
                "/api/voices/clone",
                data={"voice_id": "integration_voice", "label": "Integration Test"},
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 200
            clone_data = resp.json()
            assert clone_data["voice_id"] == "integration_voice"

            # Step 2: List voices — cloned voice must appear
            resp = client.get("/api/voices")
            assert resp.status_code == 200
            data = resp.json()
            voice_ids = [v["voice_id"] for v in data["voices"]]
            assert "integration_voice" in voice_ids
            assert data["total"] == len(data["voices"])

    def test_clone_then_tts(self):
        """Clone a voice, then synthesize speech with POST /api/tts."""
        app = make_test_app()
        wav_bytes = make_wav_bytes()

        with TestClient(app) as client:
            # Clone
            resp = client.post(
                "/api/voices/clone",
                data={"voice_id": "tts_voice", "label": "TTS Test"},
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 200

            # TTS
            resp = client.post(
                "/api/tts",
                json={
                    "text": "你好，世界！",
                    "voice_id": "tts_voice",
                    "num_steps": 4,
                    "t_shift": 0.9,
                    "speed": 1.0,
                    "guidance_scale": 3.0,
                },
            )
            assert resp.status_code == 200
            assert "audio/wav" in resp.headers["content-type"]
            assert len(resp.content) > 44
            assert resp.content[:4] == b"RIFF"

    def test_clone_then_stream_tts(self):
        """Clone a voice, then synthesize speech with POST /api/tts/stream."""
        app = make_test_app()
        wav_bytes = make_wav_bytes()

        with TestClient(app) as client:
            # Clone
            resp = client.post(
                "/api/voices/clone",
                data={"voice_id": "stream_voice", "label": "Stream Test"},
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 200

            # Stream TTS
            resp = client.post(
                "/api/tts/stream",
                json={
                    "text": "这是流式合成测试。",
                    "voice_id": "stream_voice",
                    "num_steps": 4,
                    "t_shift": 0.9,
                    "speed": 1.0,
                    "guidance_scale": 3.0,
                },
            )
            assert resp.status_code == 200
            assert "audio/wav" in resp.headers["content-type"]
            body = resp.content
            assert len(body) > 44
            assert body[:4] == b"RIFF"
            # Verify 24kHz sample rate in WAV header
            sample_rate = struct.unpack_from("<I", body, 24)[0]
            assert sample_rate == 24000

    def test_full_chain_clone_list_tts_stream(self):
        """Full chain: clone → list → TTS → stream TTS."""
        app = make_test_app()
        wav_bytes = make_wav_bytes()

        with TestClient(app) as client:
            # 1. Clone voice
            resp = client.post(
                "/api/voices/clone",
                data={"voice_id": "full_chain_voice", "label": "Full Chain"},
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 200

            # 2. List voices
            resp = client.get("/api/voices")
            assert resp.status_code == 200
            data = resp.json()
            assert any(v["voice_id"] == "full_chain_voice" for v in data["voices"])
            assert data["total"] == len(data["voices"])

            # 3. Normal TTS
            resp = client.post(
                "/api/tts",
                json={"text": "Hello world", "voice_id": "full_chain_voice"},
            )
            assert resp.status_code == 200
            assert resp.content[:4] == b"RIFF"

            # 4. Streaming TTS
            resp = client.post(
                "/api/tts/stream",
                json={"text": "Hello stream", "voice_id": "full_chain_voice"},
            )
            assert resp.status_code == 200
            assert resp.content[:4] == b"RIFF"
            sample_rate = struct.unpack_from("<I", resp.content, 24)[0]
            assert sample_rate == 24000

    def test_tts_unknown_voice_returns_404(self):
        """POST /api/tts with unregistered voice_id returns 404."""
        app = make_test_app()
        with TestClient(app) as client:
            resp = client.post(
                "/api/tts",
                json={"text": "hello", "voice_id": "nonexistent"},
            )
        assert resp.status_code == 404
        assert "nonexistent" in resp.json()["detail"]

    def test_stream_unknown_voice_returns_404(self):
        """POST /api/tts/stream with unregistered voice_id returns 404."""
        app = make_test_app()
        with TestClient(app) as client:
            resp = client.post(
                "/api/tts/stream",
                json={"text": "hello", "voice_id": "nonexistent"},
            )
        assert resp.status_code == 404
        assert "nonexistent" in resp.json()["detail"]

    def test_clone_invalid_audio_returns_422(self):
        """POST /api/voices/clone with invalid audio returns 422."""
        mock_tts = make_mock_tts()
        mock_tts.encode_prompt.side_effect = Exception("Cannot decode audio")
        app = make_test_app(mock_tts=mock_tts)

        with TestClient(app) as client:
            resp = client.post(
                "/api/voices/clone",
                data={"voice_id": "bad_voice"},
                files={"audio": ("bad.bin", b"not audio", "application/octet-stream")},
            )
        assert resp.status_code == 422
        assert "Invalid audio file" in resp.json()["detail"]

    def test_encode_prompt_called_with_correct_params(self):
        """encode_prompt is called with duration and rms from form data."""
        mock_tts = make_mock_tts()
        app = make_test_app(mock_tts=mock_tts)
        wav_bytes = make_wav_bytes()

        with TestClient(app) as client:
            client.post(
                "/api/voices/clone",
                data={"voice_id": "param_voice", "duration": "7.5", "rms": "0.005"},
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )

        mock_tts.encode_prompt.assert_called_once()
        call_kwargs = mock_tts.encode_prompt.call_args
        assert call_kwargs.kwargs.get("duration") == 7.5 or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] == 7.5
        )

    def test_generate_speech_called_with_return_smooth_false_for_tts(self):
        """POST /api/tts calls generate_speech with return_smooth=False."""
        mock_tts = make_mock_tts()
        app = make_test_app(mock_tts=mock_tts)
        app.state.voices.register_clone("v1", SAMPLE_ENCODE_DICT, "V1")

        with TestClient(app) as client:
            client.post("/api/tts", json={"text": "test", "voice_id": "v1"})

        call_kwargs = mock_tts.generate_speech.call_args.kwargs
        assert call_kwargs.get("return_smooth") is False

    def test_generate_speech_called_with_return_smooth_true_for_stream(self):
        """POST /api/tts/stream calls generate_speech with return_smooth=True."""
        mock_tts = make_mock_tts()
        app = make_test_app(mock_tts=mock_tts)
        app.state.voices.register_clone("v2", SAMPLE_ENCODE_DICT, "V2")

        with TestClient(app) as client:
            client.post("/api/tts/stream", json={"text": "test", "voice_id": "v2"})

        call_kwargs = mock_tts.generate_speech.call_args.kwargs
        assert call_kwargs.get("return_smooth") is True


# ---------------------------------------------------------------------------
# 7.2 — Property 2: clone voice queryable via HTTP layer
# ---------------------------------------------------------------------------

valid_voice_id_strategy = st.text(
    alphabet=st.sampled_from(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    ),
    min_size=1,
    max_size=32,
)


@given(voice_id=valid_voice_id_strategy)
@settings(max_examples=50)
def test_property2_clone_voice_queryable_via_http(voice_id):
    """**Validates: Requirements 2.1, 2.3**

    Property 2: For any valid voice_id, after a successful clone operation,
    GET /api/voices returns a list containing exactly one entry with that voice_id.
    """
    app = make_test_app()
    wav_bytes = make_wav_bytes()

    with TestClient(app) as client:
        # Clone the voice
        resp = client.post(
            "/api/voices/clone",
            data={"voice_id": voice_id, "label": "test"},
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert resp.status_code == 200, f"Clone failed: {resp.json()}"

        # Query voices
        resp = client.get("/api/voices")
        assert resp.status_code == 200

        data = resp.json()
        matching = [v for v in data["voices"] if v["voice_id"] == voice_id]
        assert len(matching) == 1, (
            f"Expected exactly 1 entry for voice_id={voice_id!r}, "
            f"got {len(matching)}: {data['voices']}"
        )


@given(voice_id=valid_voice_id_strategy)
@settings(max_examples=30)
def test_property2_clone_twice_appears_once(voice_id):
    """**Validates: Requirements 2.1, 2.3**

    Property 2 (overwrite variant): Cloning the same voice_id twice results in
    exactly one entry in GET /api/voices (idempotent overwrite).
    """
    app = make_test_app()
    wav_bytes = make_wav_bytes()

    with TestClient(app) as client:
        # Clone twice
        for _ in range(2):
            resp = client.post(
                "/api/voices/clone",
                data={"voice_id": voice_id, "label": "test"},
                files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            )
            assert resp.status_code == 200

        # Query voices
        resp = client.get("/api/voices")
        assert resp.status_code == 200

        data = resp.json()
        matching = [v for v in data["voices"] if v["voice_id"] == voice_id]
        assert len(matching) == 1, (
            f"Expected exactly 1 entry after double clone for voice_id={voice_id!r}, "
            f"got {len(matching)}"
        )
        assert data["total"] == len(data["voices"])

"""
Tests for API routes (api/routes.py).

Covers:
  5.1 - Unit tests for GET /voices
  5.2 - Unit tests for POST /voices/clone
  5.3 - Property tests for POST /tts  (Property 4 + Property 5)
  5.4 - Property tests for POST /tts/stream  (Property 9 + Property 5)
"""
import struct
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi import FastAPI
from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.testclient import TestClient

from api.routes import router
from api.voice_store import VoiceStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_ENCODE_DICT = {
    "prompt_tokens": b"tok",
    "prompt_features_lens": 10,
    "prompt_features": b"feat",
    "prompt_rms": 0.001,
}


def make_wav_tensor(num_samples: int = 24000) -> torch.Tensor:
    """Return a 1-D float tensor representing a silent audio clip."""
    return torch.zeros(num_samples)


def make_app(mock_tts=None, voices: VoiceStore | None = None) -> FastAPI:
    """Create a minimal FastAPI app with the router mounted at /api."""
    app = FastAPI()
    app.include_router(router, prefix="/api")

    if mock_tts is None:
        mock_tts = MagicMock()
        mock_tts.encode_prompt.return_value = SAMPLE_ENCODE_DICT
        mock_tts.generate_speech.return_value = make_wav_tensor()

    if voices is None:
        voices = VoiceStore(tts=mock_tts)

    app.state.tts = mock_tts
    app.state.voices = voices
    return app


# ---------------------------------------------------------------------------
# 5.1 — Unit tests for GET /voices
# ---------------------------------------------------------------------------

class TestListVoices:
    def test_empty_voices(self):
        """GET /voices returns empty list and total=0 when no voices registered."""
        app = make_app()
        with TestClient(app) as client:
            resp = client.get("/api/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert data["voices"] == []
        assert data["total"] == 0

    def test_single_voice(self):
        """GET /voices returns one voice with correct fields."""
        app = make_app()
        app.state.voices.register_clone("alice", SAMPLE_ENCODE_DICT, "爱丽丝")
        with TestClient(app) as client:
            resp = client.get("/api/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["voices"]) == 1
        assert data["voices"][0]["voice_id"] == "alice"
        assert data["voices"][0]["label"] == "爱丽丝"
        assert data["voices"][0]["type"] == "clone"

    def test_multiple_voices_total_equals_len(self):
        """total field always equals len(voices) for multiple voices."""
        app = make_app()
        for i in range(5):
            app.state.voices.register_clone(f"voice_{i}", SAMPLE_ENCODE_DICT, f"Voice {i}")
        with TestClient(app) as client:
            resp = client.get("/api/voices")
        data = resp.json()
        assert data["total"] == len(data["voices"])
        assert data["total"] == 5

    def test_total_equals_len_after_overwrite(self):
        """Overwriting a voice_id keeps total == len(voices) (no duplicates)."""
        app = make_app()
        app.state.voices.register_clone("dup", SAMPLE_ENCODE_DICT, "First")
        app.state.voices.register_clone("dup", SAMPLE_ENCODE_DICT, "Second")
        with TestClient(app) as client:
            resp = client.get("/api/voices")
        data = resp.json()
        assert data["total"] == len(data["voices"])
        assert data["total"] == 1


# ---------------------------------------------------------------------------
# 5.2 — Unit tests for POST /voices/clone
# ---------------------------------------------------------------------------

class TestCloneVoice:
    def _make_audio_bytes(self) -> bytes:
        """Return minimal valid WAV bytes for upload."""
        import io
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, torch.zeros(24000).numpy(), samplerate=24000, format="WAV")
        return buf.getvalue()

    def test_successful_clone(self):
        """POST /voices/clone with valid audio returns 200 and registers voice."""
        mock_tts = MagicMock()
        mock_tts.encode_prompt.return_value = SAMPLE_ENCODE_DICT
        app = make_app(mock_tts=mock_tts)

        audio_bytes = self._make_audio_bytes()
        with TestClient(app) as client:
            resp = client.post(
                "/api/voices/clone",
                data={"voice_id": "test_voice", "label": "Test", "duration": "5.0", "rms": "0.001"},
                files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["voice_id"] == "test_voice"
        assert "message" in data
        # Voice should now be registered
        assert app.state.voices.has_voice("test_voice")

    def test_duplicate_voice_id_overwrite(self):
        """Cloning with an existing voice_id overwrites it (no duplicates)."""
        mock_tts = MagicMock()
        mock_tts.encode_prompt.return_value = SAMPLE_ENCODE_DICT
        app = make_app(mock_tts=mock_tts)

        audio_bytes = self._make_audio_bytes()
        with TestClient(app) as client:
            # Clone twice with same voice_id
            for label in ("First", "Second"):
                resp = client.post(
                    "/api/voices/clone",
                    data={"voice_id": "dup_voice", "label": label},
                    files={"audio": ("test.wav", audio_bytes, "audio/wav")},
                )
                assert resp.status_code == 200

        voices = app.state.voices.list_voices()
        matching = [v for v in voices if v.voice_id == "dup_voice"]
        assert len(matching) == 1

    def test_invalid_audio_returns_422(self):
        """POST /voices/clone with audio that fails encode_prompt returns 422."""
        mock_tts = MagicMock()
        mock_tts.encode_prompt.side_effect = Exception("Cannot decode audio")
        app = make_app(mock_tts=mock_tts)

        with TestClient(app) as client:
            resp = client.post(
                "/api/voices/clone",
                data={"voice_id": "bad_voice", "label": "Bad"},
                files={"audio": ("bad.bin", b"not audio data", "application/octet-stream")},
            )
        assert resp.status_code == 422
        assert "Invalid audio file" in resp.json()["detail"]

    def test_temp_file_deleted_after_clone(self):
        """Temporary file is deleted after successful clone."""
        import os
        captured_paths = []

        def capture_encode(path, duration=None, rms=None):
            captured_paths.append(path)
            return SAMPLE_ENCODE_DICT

        mock_tts = MagicMock()
        mock_tts.encode_prompt.side_effect = capture_encode
        app = make_app(mock_tts=mock_tts)

        audio_bytes = self._make_audio_bytes()
        with TestClient(app) as client:
            resp = client.post(
                "/api/voices/clone",
                data={"voice_id": "cleanup_voice"},
                files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            )
        assert resp.status_code == 200
        assert len(captured_paths) == 1
        # Temp file should be deleted
        assert not os.path.exists(captured_paths[0])


# ---------------------------------------------------------------------------
# Strategies for property tests
# ---------------------------------------------------------------------------

voice_id_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"),
    min_size=1,
    max_size=32,
)

text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=200,
)

tts_request_strategy = st.fixed_dictionaries({
    "text": text_strategy,
    "voice_id": voice_id_strategy,
    "num_steps": st.integers(min_value=1, max_value=20),
    "t_shift": st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
    "speed": st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
    "guidance_scale": st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
})

nonexistent_voice_id_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"),
    min_size=1,
    max_size=32,
).filter(lambda v: v != "registered_voice")


# ---------------------------------------------------------------------------
# 5.3 — Property tests for POST /tts
# ---------------------------------------------------------------------------

@given(req=tts_request_strategy)
@settings(max_examples=30)
def test_property4_tts_returns_valid_wav(req):
    """**Validates: Requirements 3.1, 3.3, 3.4**

    Property 4: For any valid TTSRequest with an existing voice_id,
    POST /tts returns Content-Type audio/wav and body length > 44.
    """
    mock_tts = MagicMock()
    mock_tts.generate_speech.return_value = make_wav_tensor(24000)
    app = make_app(mock_tts=mock_tts)
    # Register the voice so it exists
    app.state.voices.register_clone(req["voice_id"], SAMPLE_ENCODE_DICT, "test")

    with TestClient(app) as client:
        resp = client.post("/api/tts", json=req)

    assert resp.status_code == 200
    assert "audio/wav" in resp.headers["content-type"]
    assert len(resp.content) > 44


@given(voice_id=nonexistent_voice_id_strategy)
@settings(max_examples=30)
def test_property5_tts_nonexistent_voice_returns_404(voice_id):
    """**Validates: Requirements 3.4**

    Property 5: For any voice_id not registered in VoiceStore,
    POST /tts returns HTTP 404.
    """
    app = make_app()
    # Do NOT register the voice_id

    req = {
        "text": "hello",
        "voice_id": voice_id,
        "num_steps": 4,
        "t_shift": 0.9,
        "speed": 1.0,
        "guidance_scale": 3.0,
    }
    with TestClient(app) as client:
        resp = client.post("/api/tts", json=req)

    assert resp.status_code == 404
    assert voice_id in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 5.4 — Property tests for POST /tts/stream
# ---------------------------------------------------------------------------

@given(req=tts_request_strategy)
@settings(max_examples=30)
def test_property9_stream_returns_valid_wav(req):
    """**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

    Property 9: For any valid TTSRequest, the bytes from POST /tts/stream
    when concatenated form a valid WAV file (starts with RIFF, sample rate = 24000).
    """
    mock_tts = MagicMock()
    mock_tts.generate_speech.return_value = make_wav_tensor(24000)
    app = make_app(mock_tts=mock_tts)
    app.state.voices.register_clone(req["voice_id"], SAMPLE_ENCODE_DICT, "test")

    with TestClient(app) as client:
        resp = client.post("/api/tts/stream", json=req)

    assert resp.status_code == 200
    assert "audio/wav" in resp.headers["content-type"]

    body = resp.content
    # Must start with RIFF header
    assert body[:4] == b"RIFF", f"Expected RIFF header, got {body[:4]!r}"
    assert len(body) > 44

    # Check sample rate field at bytes 24-28 (little-endian uint32)
    sample_rate = struct.unpack_from("<I", body, 24)[0]
    assert sample_rate == 24000, f"Expected sample rate 24000, got {sample_rate}"


@given(voice_id=nonexistent_voice_id_strategy)
@settings(max_examples=30)
def test_property5_stream_nonexistent_voice_returns_404(voice_id):
    """**Validates: Requirements 4.4**

    Property 5: For any voice_id not registered in VoiceStore,
    POST /tts/stream returns HTTP 404.
    """
    app = make_app()
    # Do NOT register the voice_id

    req = {
        "text": "hello",
        "voice_id": voice_id,
        "num_steps": 4,
        "t_shift": 0.9,
        "speed": 1.0,
        "guidance_scale": 3.0,
    }
    with TestClient(app) as client:
        resp = client.post("/api/tts/stream", json=req)

    assert resp.status_code == 404
    assert voice_id in resp.json()["detail"]

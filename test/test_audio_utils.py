"""
Property-based tests for api/audio_utils.py

Property 4: TTS 返回有效 WAV（WAV header 部分）— Validates: Requirements 3.3
Property 9: 流式响应构成合法 WAV — Validates: Requirements 4.3
"""
import struct

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from api.audio_utils import tensor_to_wav_bytes, tensor_to_wav_chunks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COMMON_SAMPLE_RATES = [8000, 16000, 22050, 24000, 44100, 48000]


def _parse_wav_sample_rate(wav_bytes: bytes) -> int:
    """Extract sample rate from WAV header (bytes 24-27, little-endian uint32)."""
    return struct.unpack_from("<I", wav_bytes, 24)[0]


# ---------------------------------------------------------------------------
# Property 4: TTS 返回有效 WAV
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

@given(
    num_samples=st.integers(min_value=1, max_value=48000),
    sample_rate=st.sampled_from(COMMON_SAMPLE_RATES),
)
@settings(max_examples=50)
def test_property4_tensor_to_wav_bytes_valid_header(num_samples, sample_rate):
    """**Validates: Requirements 3.3**
    For any random-length tensor, tensor_to_wav_bytes must return bytes that:
    - Start with b'RIFF'
    - Are longer than 44 bytes (header + at least some audio data)
    - Contain the correct sample rate in the WAV header
    """
    wav = torch.randn(num_samples)
    result = tensor_to_wav_bytes(wav, sample_rate)

    assert result[:4] == b"RIFF", "WAV must start with RIFF marker"
    assert len(result) > 44, f"WAV must be longer than 44 bytes, got {len(result)}"
    assert _parse_wav_sample_rate(result) == sample_rate, (
        f"Sample rate in header must be {sample_rate}, "
        f"got {_parse_wav_sample_rate(result)}"
    )


@given(
    num_samples=st.integers(min_value=1, max_value=48000),
)
@settings(max_examples=30)
def test_property4_wav_bytes_contains_fmt_chunk(num_samples):
    """**Validates: Requirements 3.3** — WAV bytes must contain 'WAVE' and 'fmt ' markers."""
    wav = torch.randn(num_samples)
    result = tensor_to_wav_bytes(wav, 24000)

    assert result[8:12] == b"WAVE", "Bytes 8-12 must be 'WAVE'"
    assert b"fmt " in result[:44], "WAV must contain 'fmt ' chunk marker"


# ---------------------------------------------------------------------------
# Property 9: 流式响应构成合法 WAV
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------

@given(
    num_samples=st.integers(min_value=1, max_value=48000),
    chunk_size=st.integers(min_value=64, max_value=8192),
)
@settings(max_examples=50)
def test_property9_stream_chunks_form_valid_wav(num_samples, chunk_size):
    """**Validates: Requirements 4.3**
    Concatenated bytes from tensor_to_wav_chunks must form a valid WAV file:
    - Starts with b'RIFF'
    - Length > 44
    - Sample rate field equals 24000 Hz
    """
    wav = torch.randn(num_samples)
    sample_rate = 24000

    chunks = list(tensor_to_wav_chunks(wav, sample_rate, chunk_size=chunk_size))
    concatenated = b"".join(chunks)

    assert concatenated[:4] == b"RIFF", "Concatenated stream must start with RIFF"
    assert len(concatenated) > 44, (
        f"Concatenated stream must be > 44 bytes, got {len(concatenated)}"
    )
    assert _parse_wav_sample_rate(concatenated) == sample_rate, (
        f"Sample rate in streamed WAV header must be {sample_rate}"
    )


@given(
    num_samples=st.integers(min_value=1, max_value=24000),
)
@settings(max_examples=30)
def test_property9_stream_equals_single_wav(num_samples):
    """**Validates: Requirements 4.3**
    Concatenated stream chunks must equal the output of tensor_to_wav_bytes.
    """
    wav = torch.randn(num_samples)
    sample_rate = 24000

    expected = tensor_to_wav_bytes(wav, sample_rate)
    chunks = list(tensor_to_wav_chunks(wav, sample_rate))
    concatenated = b"".join(chunks)

    assert concatenated == expected, "Stream chunks must reconstruct the full WAV bytes"

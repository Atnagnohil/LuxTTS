"""
Audio utility functions for converting PyTorch tensors to WAV bytes.
"""
import io
from typing import Iterator

import soundfile as sf
import torch


def tensor_to_wav_bytes(wav: torch.Tensor, sample_rate: int) -> bytes:
    """Convert a torch.Tensor audio waveform to WAV-encoded bytes.

    Args:
        wav: 1-D or 2-D float tensor (channels x samples or samples).
        sample_rate: Sample rate in Hz.

    Returns:
        Raw WAV file bytes including RIFF header.
    """
    # Ensure numpy array on CPU
    audio_np = wav.detach().cpu().numpy()

    # soundfile expects (samples,) for mono or (samples, channels) for multi-channel
    if audio_np.ndim == 2:
        # (channels, samples) -> (samples, channels)
        audio_np = audio_np.T

    buf = io.BytesIO()
    sf.write(buf, audio_np, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def tensor_to_wav_chunks(
    wav: torch.Tensor, sample_rate: int, chunk_size: int = 4096
) -> Iterator[bytes]:
    """Yield WAV bytes in chunks for streaming responses.

    The first (and only) chunk is the complete WAV file so that the
    concatenated bytes always form a valid RIFF/WAV file.

    Args:
        wav: Audio tensor.
        sample_rate: Sample rate in Hz.
        chunk_size: Maximum bytes per chunk (unused in current single-chunk
                    implementation, kept for API compatibility).

    Yields:
        Bytes chunks of the encoded WAV file.
    """
    wav_bytes = tensor_to_wav_bytes(wav, sample_rate)
    for offset in range(0, len(wav_bytes), chunk_size):
        yield wav_bytes[offset : offset + chunk_size]

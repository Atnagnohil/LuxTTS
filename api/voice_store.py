"""
VoiceStore: in-memory registry for preset and clone voices.
"""
import threading
from typing import Optional

from api.models import VoiceInfo


class VoiceStore:
    """Thread-safe in-memory voice registry.

    Supports two voice types:
    - preset: stored as an audio file path, encode_dict is lazily computed on first access.
    - clone:  stored with a pre-computed encode_dict.
    """

    def __init__(self, tts=None):
        """
        Args:
            tts: LuxTTS instance used for lazy-encoding preset voices.
        """
        self._tts = tts
        self._store: dict = {}  # voice_id -> entry dict
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_preset(self, voice_id: str, audio_path: str, label: str) -> None:
        """Register a preset voice (lazy-loaded from file path)."""
        with self._lock:
            self._store[voice_id] = {
                "type": "preset",
                "audio_path": audio_path,
                "label": label,
            }

    def register_clone(self, voice_id: str, encode_dict: dict, label: str) -> None:
        """Register a clone voice with a pre-computed encode_dict."""
        with self._lock:
            self._store[voice_id] = {
                "type": "clone",
                "encode_dict": encode_dict,
                "label": label,
            }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_voices(self) -> list[VoiceInfo]:
        """Return all registered voices."""
        with self._lock:
            return [
                VoiceInfo(voice_id=vid, label=entry["label"], type=entry["type"])
                for vid, entry in self._store.items()
            ]

    def has_voice(self, voice_id: str) -> bool:
        """Return True if voice_id is registered."""
        with self._lock:
            return voice_id in self._store

    def get_encode_dict(self, voice_id: str) -> Optional[dict]:
        """Return the encode_dict for a voice, lazily encoding preset voices.

        Returns None if voice_id is not registered.
        """
        with self._lock:
            entry = self._store.get(voice_id)
            if entry is None:
                return None
            if entry["type"] == "clone":
                return entry["encode_dict"]
            # preset: lazy-load on first access
            if "encode_dict" not in entry:
                encode_dict = self._tts.encode_prompt(entry["audio_path"])
                entry["encode_dict"] = encode_dict
            return entry["encode_dict"]

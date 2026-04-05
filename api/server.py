"""
FastAPI application entry point for LuxTTS API Server.

Manages model lifecycle via lifespan context manager and registers API routes.
"""
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from api.routes import router
from api.voice_store import VoiceStore

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    tts: Any  # LuxTTS instance
    voices: VoiceStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # --- Startup ---
    model_path = os.environ.get("MODEL_PATH", "models/luxtts.pt")

    try:
        from zipvoice.luxvoice import LuxTTS
        tts = LuxTTS(model_path)
    except Exception as exc:
        logger.error("Failed to load LuxTTS model from '%s': %s", model_path, exc)
        sys.exit(1)

    voices = VoiceStore(tts=tts)

    # Register preset voices from PRESET_VOICES env var
    # Format: "voice_id:label:path,voice_id2:label2:path2"
    preset_voices_env = os.environ.get("PRESET_VOICES", "")
    if preset_voices_env.strip():
        for entry in preset_voices_env.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":", 2)
            if len(parts) != 3:
                logger.warning("Skipping malformed PRESET_VOICES entry: %r", entry)
                continue
            voice_id, label, path = parts
            voices.register_preset(voice_id.strip(), path.strip(), label.strip())
            logger.info("Registered preset voice '%s' from '%s'", voice_id.strip(), path.strip())

    app.state.app_state = AppState(tts=tts, voices=voices)
    # Expose tts and voices directly on app.state for route convenience
    app.state.tts = tts
    app.state.voices = voices

    yield

    # --- Shutdown ---
    app.state.app_state = None
    app.state.tts = None
    app.state.voices = None


app = FastAPI(title="LuxTTS API Server", lifespan=lifespan)
app.include_router(router, prefix="/api")

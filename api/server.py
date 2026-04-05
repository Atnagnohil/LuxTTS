"""
FastAPI application entry point for LuxTTS API Server.

Manages model lifecycle via lifespan context manager and registers API routes.
"""
import logging
import os
import sys

# 使用本地缓存，禁止联网下载模型
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
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
    model_path = os.environ.get("MODEL_PATH", "YatharthS/LuxTTS")
    
    # If MODEL_PATH is not set or is the default HF repo, try to find in cache
    if model_path == "YatharthS/LuxTTS":
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_pattern = os.path.join(cache_dir, "models--YatharthS--LuxTTS/snapshots/*")
        import glob
        cached_models = glob.glob(model_cache_pattern)
        if cached_models:
            # Use the most recent snapshot
            model_path = max(cached_models, key=os.path.getmtime)
            logger.info(f"Using cached model from: {model_path}")
        else:
            logger.info("Model not found in cache, will download from HuggingFace")
            model_path = None

    try:
        from zipvoice.luxvoice import LuxTTS
        device = os.environ.get("DEVICE", "cuda")
        tts = LuxTTS(model_path, device=device)
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

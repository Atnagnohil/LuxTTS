"""
API routes for LuxTTS server.

Implements four endpoints:
  GET  /voices
  POST /voices/clone
  POST /tts
  POST /tts/stream
"""
import asyncio
import os
import tempfile
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from api.audio_utils import tensor_to_wav_chunks, tensor_to_wav_bytes
from api.models import CloneVoiceResponse, TTSRequest, VoicesResponse

router = APIRouter()

# Global lock to serialize GPU inference and prevent OOM errors
_inference_lock = asyncio.Lock()


@router.get("/voices", response_model=VoicesResponse)
async def list_voices(request: Request) -> VoicesResponse:
    """Return all registered voices."""
    voices = request.app.state.voices.list_voices()
    return VoicesResponse(voices=voices, total=len(voices))


@router.post("/voices/clone", response_model=CloneVoiceResponse)
async def clone_voice(
    request: Request,
    voice_id: Annotated[str, Form()],
    label: Annotated[str, Form()] = "",
    duration: Annotated[float, Form()] = 5.0,
    rms: Annotated[float, Form()] = 0.001,
    audio: UploadFile = File(...),
) -> CloneVoiceResponse:
    """Clone a voice from an uploaded audio file."""
    # Save uploaded file to a temporary path
    suffix = os.path.splitext(audio.filename or "audio")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        contents = await audio.read()
        tmp.write(contents)
        tmp.close()

        # Encode the audio prompt
        try:
            encode_dict = request.app.state.tts.encode_prompt(
                tmp.name, duration=duration, rms=rms
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid audio file: {exc}") from exc

        # Register the clone in VoiceStore
        request.app.state.voices.register_clone(voice_id, encode_dict, label)
    finally:
        # Always clean up the temporary file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return CloneVoiceResponse(voice_id=voice_id, message="Voice cloned successfully")


@router.post("/tts")
async def synthesize(req: TTSRequest, request: Request) -> FileResponse:
    """Synthesize speech and return a complete WAV file."""
    encode_dict = request.app.state.voices.get_encode_dict(req.voice_id)
    if encode_dict is None:
        raise HTTPException(status_code=404, detail=f"Voice '{req.voice_id}' not found")

    loop = asyncio.get_event_loop()

    loop = asyncio.get_event_loop()

    async with _inference_lock:
        try:
            wav = await loop.run_in_executor(
                None,
                lambda: request.app.state.tts.generate_speech(
                    req.text,
                    encode_dict,
                    num_steps=req.num_steps,
                    t_shift=req.t_shift,
                    speed=req.speed,
                    return_smooth=False,
                ),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    # Write WAV bytes to a temporary file and return as FileResponse
    wav_bytes = tensor_to_wav_bytes(wav, sample_rate=48000)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_bytes)
    tmp.close()

    return FileResponse(tmp.name, media_type="audio/wav", filename="output.wav")


@router.post("/tts/stream")
async def synthesize_stream(req: TTSRequest, request: Request) -> StreamingResponse:
    """Synthesize speech and return a streaming WAV response."""
    encode_dict = request.app.state.voices.get_encode_dict(req.voice_id)
    if encode_dict is None:
        raise HTTPException(status_code=404, detail=f"Voice '{req.voice_id}' not found")

    loop = asyncio.get_event_loop()

    loop = asyncio.get_event_loop()

    async with _inference_lock:
        try:
            wav = await loop.run_in_executor(
                None,
                lambda: request.app.state.tts.generate_speech(
                    req.text,
                    encode_dict,
                    num_steps=req.num_steps,
                    t_shift=req.t_shift,
                    speed=req.speed,
                    return_smooth=False,
                ),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    def audio_generator():
        yield from tensor_to_wav_chunks(wav, sample_rate=48000)

    return StreamingResponse(audio_generator(), media_type="audio/wav")

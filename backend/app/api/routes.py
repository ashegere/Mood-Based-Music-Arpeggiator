"""
FastAPI router for arpeggio generation.

This module is intentionally thin — it only translates between HTTP types
(request/response schemas) and domain types (GenerationRequest / GenerationResult).
All generation logic lives in the active ``BaseGenerator`` backend.

Non-blocking inference
----------------------
``BaseGenerator.generate()`` is a synchronous, CPU/GPU-bound call that may
take hundreds of milliseconds.  Calling it directly inside an ``async def``
handler would block the entire asyncio event loop for every request.

Instead, we offload it to Starlette's default thread-pool executor via
``run_in_threadpool``.  The event loop stays free to handle other requests
while the model runs in a worker thread.  Thread safety is guaranteed by the
``threading.RLock`` inside ``PretrainedMusicTransformerGenerator``.
"""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.api.dependencies import get_generator
from app.api.schemas import GenerateArpeggioRequest, GenerateArpeggioResponse, NoteEvent, SamplingParams
from app.config import settings
from app.generators.base import BaseGenerator, GenerationRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/generate-arpeggio",
    response_model=GenerateArpeggioResponse,
    summary="Generate a mood-conditioned arpeggio",
    tags=["Generation"],
)
async def generate_arpeggio(
    request: GenerateArpeggioRequest,
    generator: BaseGenerator = Depends(get_generator),
) -> GenerateArpeggioResponse:
    """
    Generate a MIDI arpeggio conditioned on a mood description.

    **Pipeline** (delegated to the active ``BaseGenerator`` backend):
    1. Resolve `mood` text → nearest model mood label.
    2. Autoregressive token generation with temperature + top-k sampling.
    3. Decode token stream → Note objects.
    4. Render to a MIDI file and return as base64.

    **Returns** a JSON body containing:
    - `midi_base64`: The MIDI file, base64-encoded.
    - `notes`: Note events for direct client-side playback.
    - Metadata: key, scale, tempo, mood, note count, duration.

    Inference runs in a thread-pool worker so the event loop is never blocked.
    """
    try:
        gen_request = GenerationRequest(
            key=request.key,
            scale=request.scale,
            tempo=request.tempo,
            note_count=request.note_count,
            mood=request.mood,
            octave=request.octave,
            seed=request.seed,
            pattern=request.pattern,
            bars=request.bars,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_length=request.max_length,
        )
        # run_in_threadpool offloads the blocking, CPU/GPU-bound generate() call
        # to Starlette's thread-pool so the asyncio event loop stays responsive.
        result = await run_in_threadpool(generator.generate, gen_request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # sampling_params may be empty when CustomTransformerGenerator is used;
    # fill in server defaults so the response schema is always satisfied.
    sp = result.sampling_params or {}
    sampling = SamplingParams(
        temperature=sp.get("temperature", settings.GENERATION_TEMPERATURE),
        top_k=sp.get("top_k", settings.GENERATION_TOP_K),
        top_p=sp.get("top_p", 1.0),
        repetition_penalty=sp.get("repetition_penalty", 1.0),
        max_length=sp.get("max_length", settings.GENERATION_MAX_GEN_TOKENS),
    )

    return GenerateArpeggioResponse(
        midi_base64=base64.b64encode(result.midi_bytes).decode("utf-8"),
        notes=[
            NoteEvent(
                pitch=n.pitch,
                velocity=n.velocity,
                position=n.position,
                duration=n.duration,
            )
            for n in result.notes
        ],
        key=result.key,
        scale=result.scale,
        tempo=result.tempo,
        mood=result.mood,
        note_count=result.note_count,
        duration_seconds=result.duration_seconds,
        sampling=sampling,
        alignment_score=result.alignment_score,
    )

"""
FastAPI router for arpeggio generation.

This module is intentionally thin — it only translates between HTTP types
(request/response schemas) and domain types (GenerationRequest / GenerationResult).
All generation logic lives in the active ``BaseGenerator`` backend.
"""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_generator
from app.api.schemas import GenerateArpeggioRequest, GenerateArpeggioResponse, NoteEvent
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
    2. Generate a base arpeggio (rule-based, deterministic).
    3. Transform via mood-conditioned model inference.
    4. Render to a MIDI file and return as base64.

    **Returns** a JSON body containing:
    - `midi_base64`: The MIDI file, base64-encoded.
    - `notes`: Note events for direct client-side playback.
    - Metadata: key, scale, tempo, mood, note count, duration.
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
        )
        result = generator.generate(gen_request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

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
    )

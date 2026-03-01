"""
FastAPI dependency providers for the generation layer.

Usage in route handlers:

    from app.api.dependencies import get_generator

    @router.post("/generate-arpeggio")
    async def generate(
        request: GenerateArpeggioRequest,
        generator: BaseGenerator = Depends(get_generator),
    ):
        ...

Startup wiring (call once from ``app.main`` lifespan):

    from app.api.dependencies import set_generator
    from app.generators.pretrained_transformer import PretrainedMusicTransformerGenerator

    generator = PretrainedMusicTransformerGenerator(checkpoint_path=...)
    generator.load()
    set_generator(generator)
"""

from __future__ import annotations

import threading
from typing import Optional

from fastapi import HTTPException

from app.generators.base import BaseGenerator

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_generator: Optional[BaseGenerator] = None
_lock = threading.Lock()


def set_generator(generator: BaseGenerator) -> None:
    """
    Register the active generation backend.

    Must be called before the first request arrives — typically inside the
    FastAPI ``lifespan`` startup block in ``app.main``.

    Args:
        generator: A fully loaded ``BaseGenerator`` subclass instance.
    """
    global _generator
    with _lock:
        _generator = generator


def get_generator() -> BaseGenerator:
    """
    FastAPI ``Depends``-compatible provider for the active generator.

    Raises HTTP 503 when the generator has not been registered or has not
    finished loading, so clients receive a structured error rather than an
    unhandled exception.

    Returns:
        The registered ``BaseGenerator`` instance.

    Raises:
        HTTPException(503): Generator not registered or not ready.
    """
    gen = _generator
    if gen is None:
        raise HTTPException(
            status_code=503,
            detail="Generator not initialised — the service is still starting up.",
        )
    if not gen.is_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Generator '{gen.name}' is not ready — the model may still be loading.",
        )
    return gen

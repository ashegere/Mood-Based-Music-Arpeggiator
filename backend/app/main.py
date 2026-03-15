"""
FastAPI application — mood-conditioned arpeggio generator.

Generator
---------
The active backend is always ``PretrainedMusicTransformerGenerator``: a
GPT-style decoder-only transformer that generates symbolic music in the
REMI token format with optional mood conditioning via a lightweight
``MoodConditioningModule`` adapter.

Startup sequence
----------------
1. Database tables are created (skipped gracefully if the DB is unreachable).
2. ``_build_generator()`` reads checkpoint paths and sampling hyperparameters
   from ``app.config.settings`` (overridable via environment variables /
   ``.env`` file), then calls ``gen.load()`` in the main startup coroutine.
3. The loaded generator is registered via ``set_generator()`` so that the
   ``get_generator`` FastAPI dependency can inject it into route handlers.

If the backbone checkpoint is missing, the app starts in **degraded mode**:
all generation endpoints return HTTP 503 until the checkpoint is supplied and
the process is restarted.  This avoids a hard crash during deployment while
a checkpoint is being provisioned.

Configurable generation params (via .env or environment)
---------------------------------------------------------
``PRETRAINED_CHECKPOINT``      Path to backbone .pt file.
``MOOD_ADAPTER_CHECKPOINT``    Path to mood adapter .pt file (optional).
``GENERATION_TEMPERATURE``     Sampling temperature (default 0.95).
``GENERATION_TOP_K``           Top-k filter width (default 50; 0 = off).
``GENERATION_MAX_GEN_TOKENS``  Per-request token budget (default 1024).

Non-blocking inference
----------------------
``generator.generate()`` is CPU/GPU-bound synchronous work.  It is always
called via ``fastapi.concurrency.run_in_threadpool`` in the route layer, so
the asyncio event loop is never blocked during inference.  Thread safety
inside the generator is guaranteed by an internal ``threading.RLock``.

Checkpoint paths (defaults)
---------------------------
Backbone : backend/checkpoints/pretrained_music_transformer.pt
Adapter  : backend/checkpoints/mood_adapter.pt
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.api.dependencies import get_generator, set_generator
from app.config import settings
from app.database import Base, engine
from app.generators.base import BaseGenerator
from app.generators.transformer import CustomTransformerGenerator
from app.routers import auth, favorites
# Import models so Base.metadata.create_all picks them up
import app.models.saved_midi  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------

def _build_generator() -> BaseGenerator:
    """
    Instantiate and load the CustomTransformerGenerator.

    The checkpoint path is resolved relative to ``backend/`` so the same
    default works whether the process is launched from ``backend/`` or from
    the repository root.

    Raises:
        FileNotFoundError: Checkpoint not found.
    """
    base_dir = Path(__file__).parent.parent  # …/backend/
    custom_path = base_dir / settings.CUSTOM_CHECKPOINT

    if not custom_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {custom_path}\n"
            "Override via the CUSTOM_CHECKPOINT environment variable."
        )

    logger.info("Loading CustomTransformerGenerator | checkpoint=%s", custom_path)
    gen = CustomTransformerGenerator(checkpoint_path=custom_path)
    gen.load()
    return gen


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----------------------------------------------------------------- startup
    logger.info("Starting %s v%s", settings.APP_NAME, settings.VERSION)

    # ---- Database --------------------------------------------------------
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ready")
    except Exception as exc:
        logger.warning("Database init skipped: %s", exc)

    # ---- Generator -------------------------------------------------------
    # Load runs synchronously here at startup (before any requests arrive),
    # so blocking the event loop is acceptable.  Inference is non-blocking
    # at request time via run_in_threadpool in the route layer.
    try:
        gen = _build_generator()
        set_generator(gen)
        logger.info(
            "Generator ready | backend='%s' mood_adapter=%s",
            gen.name,
            getattr(gen, "_mood_adapter", None) is not None,
        )
    except FileNotFoundError as exc:
        logger.error(
            "Backbone checkpoint missing — starting in degraded mode.\n%s\n"
            "Generation endpoints will return HTTP 503 until resolved.",
            exc,
        )
    except Exception as exc:
        logger.error(
            "Unexpected error loading generator — starting in degraded mode: %s",
            exc,
            exc_info=True,
        )

    yield
    # ---------------------------------------------------------------- shutdown
    logger.info("Shutting down %s", settings.APP_NAME)


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "Mood-conditioned symbolic music arpeggio generator.\n\n"
        "Uses a pretrained GPT-style transformer with an optional "
        "fine-tuned mood conditioning adapter."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(routes.router, prefix="/api")
app.include_router(favorites.router)


# ---------------------------------------------------------------------------
# Built-in endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.VERSION,
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Report service health and the active generator's runtime configuration.

    Returns ``status: "healthy"`` only when a generator is registered and
    reports ``is_ready = True``.  The response includes the sampling
    hyperparameters so clients can confirm the expected configuration
    without inspecting server logs.
    """
    try:
        gen = get_generator()
        # Expose sampling config so operators can verify deployed settings.
        gen_config: Dict[str, Any] = {
            "temperature": getattr(gen, "temperature", None),
            "top_k":       getattr(gen, "top_k", None),
            "max_gen_tokens": getattr(gen, "max_gen_tokens", None),
            "mood_adapter_loaded": getattr(gen, "_mood_adapter", None) is not None,
        }
        return {
            "status":          "healthy",
            "model_loaded":    True,
            "backend":         gen.name,
            "version":         settings.VERSION,
            "generation":      gen_config,
        }
    except Exception:
        return {
            "status":          "degraded",
            "model_loaded":    False,
            "backend":         None,
            "version":         settings.VERSION,
            "generation":      None,
        }


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8006,
        reload=settings.DEBUG,
    )

"""
FastAPI application entry point.

Generator startup strategy
--------------------------
The application tries to load the **PretrainedMusicTransformerGenerator**
first.  If its checkpoint does not exist it falls back to the
**CustomTransformerGenerator** (the in-house mood-conditioned model).

To swap the active backend, change only the ``_load_generator`` block in
the lifespan below — routes and schemas are never touched because they
depend only on the ``BaseGenerator`` interface.

Checkpoint paths
----------------
Pretrained  : backend/checkpoints/pretrained_music_transformer.pt
Custom      : backend/checkpoints/best_model.pt
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.api.dependencies import get_generator, set_generator
from app.config import settings
from app.database import Base, engine
from app.generators.base import BaseGenerator
from app.generators.pretrained_transformer import PretrainedMusicTransformerGenerator
from app.generators.transformer import CustomTransformerGenerator
from app.routers import auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_generator(base_dir: Path) -> BaseGenerator:
    """
    Attempt to load the pretrained generator; fall back to the custom one.

    Args:
        base_dir: ``backend/`` directory (parent of ``app/``).

    Returns:
        A fully loaded ``BaseGenerator`` instance.

    Raises:
        RuntimeError: If neither checkpoint is loadable.
    """
    pretrained_ckpt = base_dir / "checkpoints" / "pretrained_music_transformer.pt"
    custom_ckpt     = base_dir / "checkpoints" / "best_model.pt"
    dataset_path    = base_dir / "data" / "training" / "train_dataset.pt"

    # ── Option 1: Pretrained symbolic music transformer ──────────────
    if pretrained_ckpt.exists():
        logger.info("Found pretrained checkpoint: %s", pretrained_ckpt)
        gen = PretrainedMusicTransformerGenerator(
            checkpoint_path=pretrained_ckpt,
            temperature=0.95,
            top_k=50,
        )
        gen.load()
        return gen

    logger.info(
        "Pretrained checkpoint not found (%s) — "
        "falling back to CustomTransformerGenerator.",
        pretrained_ckpt,
    )

    # ── Option 2: In-house mood-conditioned transformer ───────────────
    if not custom_ckpt.exists():
        raise RuntimeError(
            f"No usable checkpoint found.\n"
            f"  Pretrained path: {pretrained_ckpt}\n"
            f"  Custom path:     {custom_ckpt}\n"
            "Train a model first (see backend/scripts/) or place a "
            "pretrained checkpoint at the paths above."
        )

    gen = CustomTransformerGenerator(
        checkpoint_path=custom_ckpt,
        dataset_path=dataset_path if dataset_path.exists() else None,
    )
    gen.load()
    return gen


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------ startup
    logger.info("Starting %s v%s", settings.APP_NAME, settings.VERSION)

    # ---- Database ----
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ready")
    except Exception as exc:
        logger.warning("Database init skipped: %s", exc)

    # ---- Generator ----
    try:
        _base = Path(__file__).parent.parent   # backend/
        gen   = _load_generator(_base)
        set_generator(gen)
        logger.info(
            "Active generator: '%s'",
            gen.name,
        )
    except Exception as exc:
        logger.error(
            "Failed to load any generator: %s — "
            "generation endpoints will return 503 until resolved.",
            exc,
            exc_info=True,
        )

    yield
    # ----------------------------------------------------------------- shutdown
    logger.info("Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Mood-conditioned arpeggio generation API",
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


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.VERSION,
        "docs":    "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Report service health and the active generator backend.

    Returns ``status: "healthy"`` only when a generator is registered
    and has finished loading.
    """
    try:
        gen = get_generator()
        return {
            "status":       "healthy",
            "model_loaded": True,
            "backend":      gen.name,
            "version":      settings.VERSION,
        }
    except Exception:
        return {
            "status":       "initializing",
            "model_loaded": False,
            "backend":      None,
            "version":      settings.VERSION,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8006,
        reload=settings.DEBUG,
    )

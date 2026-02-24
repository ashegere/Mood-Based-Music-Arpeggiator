"""
FastAPI application entry point.

Startup wires the active generation backend via dependency injection:
  1. Instantiate ``CustomTransformerGenerator`` with the checkpoint path.
  2. Call ``generator.load()`` to allocate model weights.
  3. Call ``set_generator(generator)`` to make it available to route handlers.

Swapping backends requires only changing steps 1-3 here — routes and schemas
are untouched because they depend only on the ``BaseGenerator`` interface.
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
from app.generators.transformer import CustomTransformerGenerator
from app.routers import auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------ startup
    logger.info("Starting %s v%s", settings.APP_NAME, settings.VERSION)

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ready")
    except Exception as exc:
        logger.warning("Database init skipped: %s", exc)

    try:
        _base      = Path(__file__).parent.parent          # backend/
        checkpoint = _base / "checkpoints" / "best_model.pt"
        dataset    = _base / "data" / "training" / "train_dataset.pt"

        generator = CustomTransformerGenerator(
            checkpoint_path=checkpoint,
            dataset_path=dataset if dataset.exists() else None,
        )
        generator.load()
        set_generator(generator)
        logger.info("Generator '%s' ready (device=%s)", generator.name, generator._engine.device)
    except Exception as exc:
        logger.error("Failed to load generator: %s", exc, exc_info=True)
        # Service starts without a generator; /health will report not-ready.

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
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Report service health and the active generator backend."""
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

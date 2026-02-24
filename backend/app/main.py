"""
FastAPI application entry point.

Startup loads the inference engine once; all requests share the singleton.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.config import settings
from app.database import Base, engine
from app.model.inference import load_model
from app.routers import auth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --------------- startup ---------------
    logger.info("Starting %s v%s", settings.APP_NAME, settings.VERSION)

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ready")
    except Exception as exc:
        logger.warning("Database init skipped: %s", exc)

    try:
        _base = Path(__file__).parent.parent   # backend/
        checkpoint = _base / "checkpoints" / "best_model.pt"
        dataset    = _base / "data" / "training" / "train_dataset.pt"
        load_model(checkpoint, dataset_path=dataset if dataset.exists() else None)
        logger.info("Inference engine loaded from %s", checkpoint)
    except Exception as exc:
        logger.error("Failed to load inference engine: %s", exc, exc_info=True)

    yield
    # --------------- shutdown ---------------
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
    from app.model.inference import get_engine
    engine = get_engine()
    return {
        "status": "healthy" if engine.is_loaded else "initializing",
        "model_loaded": engine.is_loaded,
        "version": settings.VERSION,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8006,
        reload=settings.DEBUG,
    )

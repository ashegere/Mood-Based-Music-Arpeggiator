
from pydantic_settings import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):
    APP_NAME: str = "AI Arpeggiator"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database settings
    DATABASE_URL: str = "postgresql://postgres:postgres@127.0.0.1:5432/arpeggiator"

    # JWT settings
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Google OAuth settings
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8006/api/auth/google/callback"

    # CORS settings
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # Music settings
    MIN_BPM: int = 40
    MAX_BPM: int = 240
    MIN_BARS: int = 1
    MAX_BARS: int = 8

    # Generator — pretrained music transformer
    # Override any of these in your .env file without touching source code.
    PRETRAINED_CHECKPOINT: str = "checkpoints/pretrained_music_transformer.pt"
    MOOD_ADAPTER_CHECKPOINT: str = "checkpoints/mood_adapter.pt"
    # Fallback generator checkpoint (CustomTransformerGenerator / best_model.pt)
    CUSTOM_CHECKPOINT: str = "checkpoints/best_model.pt"
    # Sampling temperature: lower = more conservative, higher = more creative.
    GENERATION_TEMPERATURE: float = 0.95
    # Top-k filtering: 0 disables (pure temperature sampling).
    GENERATION_TOP_K: int = 50
    # Hard upper bound on generated tokens per request (auto-scaled by note_count).
    GENERATION_MAX_GEN_TOKENS: int = 1024

    # Mood alignment classifier
    # Path to mood_classifier.pt (written by scripts/train_mood_classifier.py).
    MOOD_CLASSIFIER_CHECKPOINT: str = "checkpoints/mood_classifier.pt"
    # Minimum alignment score [0, 1] to accept a generation without retrying.
    # Set to 0.0 (default) to score but never regenerate.
    # Set > 0.0 (e.g. 0.5) to enable automatic re-generation on low-scoring outputs.
    ALIGNMENT_SCORE_THRESHOLD: float = 0.0
    # Maximum generation attempts when re-generation is enabled (threshold > 0).
    ALIGNMENT_MAX_ATTEMPTS: int = 3

    class Config:
        env_file = ".env"

settings = Settings()

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

    # Model settings
    USE_GPU: bool = False
    MODEL_NAME: str = "gpt2"
    MAX_PATTERN_LENGTH: int = 32

    # Music settings
    MIN_BPM: int = 40
    MAX_BPM: int = 240
    MIN_BARS: int = 1
    MAX_BARS: int = 8

    class Config:
        env_file = ".env"

settings = Settings()
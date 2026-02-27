"""Configuration settings for the urban crime risk perception project."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Container for all configurable settings used across the project."""

    openai_api_key: str
    model_name: str = "gpt-4o"
    request_timeout_seconds: int = 30
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    requests_per_minute: int = 30


SETTINGS = Settings(
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
    request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30")),
    max_retries=int(os.getenv("MAX_RETRIES", "5")),
    retry_backoff_seconds=float(os.getenv("RETRY_BACKOFF_SECONDS", "2.0")),
    requests_per_minute=int(os.getenv("REQUESTS_PER_MINUTE", "30")),
)

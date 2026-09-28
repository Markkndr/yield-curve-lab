"""Project paths and environment settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


def get_fred_api_key() -> str:
    """Return the FRED API key from the environment or `.env`."""
    key = os.getenv("FRED_API_KEY")
    if not key or key == "your_key_here":
        raise RuntimeError("FRED_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key

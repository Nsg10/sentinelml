"""
Resolves the correct models directory path
whether running locally or on Render.
"""
from pathlib import Path

def get_models_dir() -> Path:
    # Always resolve relative to this file's location
    return Path(__file__).parent.parent / "models"

"""
Model utilities for loading, saving, and managing prediction models.

Models are stored under ai/lina/models/ only.
"""

import pickle
from pathlib import Path
from typing import Optional, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_model_dir() -> Optional[Path]:
    """
    Locate the ai/lina/models/ directory by walking up from this file.

    Returns:
        Path to models directory, or None if not found.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "models"
        if candidate.exists():
            return candidate
    return None


def save_model(obj: Any, filename: str) -> Optional[Path]:
    """
    Serialize and save a model (or scaler) as a pickle file.

    Args:
        obj: Object to serialize (XGBoost model, sklearn scaler, etc.).
        filename: Filename (e.g., "crowd_predictor.pkl").

    Returns:
        Path to the saved file, or None on failure.
    """
    model_dir = get_model_dir()
    if model_dir is None:
        logger.error("Models directory not found — cannot save model")
        return None

    filepath = model_dir / filename
    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Model saved → %s", filepath)
        return filepath
    except (OSError, pickle.PicklingError) as exc:
        logger.error("Failed to save model '%s': %s", filename, exc)
        return None


def load_model(filename: str) -> Optional[Any]:
    """
    Load a serialized model from the models directory.

    Args:
        filename: Filename (e.g., "crowd_predictor.pkl").

    Returns:
        Deserialized object, or None if not found or failed to load.
    """
    model_dir = get_model_dir()
    if model_dir is None:
        logger.error("Models directory not found — cannot load model")
        return None

    filepath = model_dir / filename
    if not filepath.exists():
        logger.warning("Model file not found: %s", filepath)
        return None

    try:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        logger.info("Model loaded from %s", filepath)
        return obj
    except (OSError, pickle.UnpicklingError) as exc:
        logger.error("Failed to load model '%s': %s", filename, exc)
        return None


def model_exists(filename: str) -> bool:
    """
    Check whether a model file exists.

    Args:
        filename: Model filename.

    Returns:
        True if the model file exists in the models directory.
    """
    model_dir = get_model_dir()
    if model_dir is None:
        return False
    return (model_dir / filename).exists()

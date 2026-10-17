"""Runtime prompt-injection guard for the AI Firewall.

Loads the model produced by ``train_model.py`` once and scores incoming prompts
with it. The positive class (1) means "prompt injection".
"""

from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).with_name("injection_model.pkl")


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the trained pipeline, failing loudly if it is missing."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def check_injection(text: str) -> dict[str, bool | float]:
    """Classify ``text`` as a prompt-injection attempt (1) or benign (0).

    Returns ``is_injection`` and ``confidence``, where confidence is the
    probability of the label that was predicted (1.0 = certain).
    """
    model = _load_model()

    probabilities = model.predict_proba([text])[0]
    class_index = list(model.classes_).index(1)
    injection_probability = float(probabilities[class_index])

    is_injection = injection_probability >= 0.5
    confidence = injection_probability if is_injection else 1 - injection_probability

    return {"is_injection": is_injection, "confidence": round(confidence, 4)}

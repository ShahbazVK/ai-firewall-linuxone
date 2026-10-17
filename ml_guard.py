"""Runtime prompt-injection guard for the AI Firewall.

Loads the model produced by ``train_model.py`` once, scores incoming prompts
with it, and explains blocked prompts using the classifier's coefficients. The
positive class (1) means "prompt injection".
"""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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


def explain_injection(text: str, top_k: int = 3) -> list[str]:
    """Return the words that contributed most to an injection prediction.

    A linear classifier has one coefficient per hashed feature. For every
    unigram and bigram the vectorizer recognizes in ``text``, this function
    multiplies that coefficient by the feature's value and splits the result
    across the words in the n-gram (so ``"system prompt"`` credits both words).
    Only positive contributions are kept, because those are the signals that
    pushed the model toward the injection class.

    Common English stop words are excluded so the explanation stays readable.
    Returns at most ``top_k`` words, ordered by contribution.
    """
    if top_k <= 0:
        return []

    model = _load_model()
    try:
        vectorizer = model.named_steps["hash"]
    except KeyError as exc:
        raise RuntimeError(
            "Model is missing the HashingVectorizer step. "
            "Retrain with `python train_model.py`."
        ) from exc
    classifier = model.named_steps["clf"]

    coefficients = classifier.coef_
    positive_class = list(model.classes_).index(1)
    weights = (
        coefficients[0]
        if coefficients.shape[0] == 1
        else coefficients[positive_class]
    )

    features = vectorizer.build_analyzer()(text)
    if not features:
        return []

    # HashingVectorizer is stateless, so a matching FeatureHasher maps any
    # token or n-gram back to the bucket the vectorizer used.
    feature_hasher = FeatureHasher(
        n_features=vectorizer.n_features,
        input_type="string",
        alternate_sign=vectorizer.alternate_sign,
    )
    vectorized = vectorizer.transform([text])

    contributions: dict[str, float] = defaultdict(float)
    for feature in features:
        index = int(feature_hasher.transform([[feature]]).indices[0])
        score = float(weights[index]) * float(vectorized[0, index])
        words = feature.split()
        share = score / len(words)
        for word in words:
            contributions[word] += share

    ranked = sorted(
        (
            (word, score)
            for word, score in contributions.items()
            if score > 0 and word not in ENGLISH_STOP_WORDS
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [word for word, _ in ranked[:top_k]]

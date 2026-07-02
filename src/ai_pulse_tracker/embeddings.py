from __future__ import annotations

import hashlib
import math
import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+'-]*")
_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "for",
    "from",
    "how",
    "the",
    "this",
    "what",
    "which",
    "with",
}
_SHORT_TERMS = {"ai", "ia", "ml"}
DEFAULT_DIMENSIONS = 128
EMBEDDING_MODEL_NAME = f"local-hashed-token-ngrams-{DEFAULT_DIMENSIONS}"


def embed_text(text: str, *, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokenize(text):
        _add_feature(vector, token, 1.0)
        for ngram in _char_ngrams(token):
            _add_feature(vector, ngram, 0.35)
    return _normalize(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def article_embedding_text(
    *,
    title: str,
    summary: str,
    description: str | None,
    source: str,
    keywords: list[str],
) -> str:
    return " ".join(
        [
            title or "",
            summary or "",
            description or "",
            source or "",
            " ".join(keywords),
        ]
    )


def _tokenize(text: str) -> list[str]:
    terms = [term.lower().strip("'-") for term in _TOKEN_RE.findall(text)]
    return [
        term
        for term in terms
        if (len(term) >= 3 or term in _SHORT_TERMS)
        and term not in _STOPWORDS
        and not term.isnumeric()
    ]


def _char_ngrams(token: str, size: int = 3) -> list[str]:
    if len(token) <= size:
        return [token]
    padded = f"_{token}_"
    return [padded[index : index + size] for index in range(len(padded) - size + 1)]


def _add_feature(vector: list[float], feature: str, weight: float) -> None:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % len(vector)
    vector[bucket] += weight


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]

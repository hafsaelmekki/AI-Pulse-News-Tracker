from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass(slots=True)
class Article:
    source: str
    title: str
    description: str | None
    url: str
    published_at: datetime


@dataclass(slots=True)
class AnalyzedArticle(Article):
    sentiment: str
    confidence_pos: float
    confidence_neu: float
    confidence_neg: float

    def to_cosmos_document(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "sentiment": self.sentiment,
            "confidence": {
                "pos": self.confidence_pos,
                "neu": self.confidence_neu,
                "neg": self.confidence_neg,
            },
            "url": self.url,
            "date": self.published_at.isoformat(),
        }

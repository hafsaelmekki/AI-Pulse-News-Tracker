"""Core package for the AI Pulse News Tracker."""

from .config import Settings, load_settings

__all__ = [
    "Settings",
    "load_settings",
    "NewsAnalyzerPipeline",
]


def __getattr__(name: str):
    if name == "NewsAnalyzerPipeline":
        from .pipeline import NewsAnalyzerPipeline

        return NewsAnalyzerPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Retrieval system exports."""
from .base import (
    MemoryRetriever,
    RetrievalResult,
    RetrievalConfig,
)
from .spreading_activation import (
    SpreadingActivationRetriever,
    create_retriever,
)

# Production default: Hybrid retriever (FAISS + CA evolution + ranking)
_HYBRID_AVAILABLE = False
try:
    from .hybrid_retriever import HybridRetriever  # noqa: F401
    _HYBRID_AVAILABLE = True
except ImportError:
    pass  # sentence-transformers/FAISS not installed

__all__ = [
    "MemoryRetriever",
    "RetrievalResult", 
    "RetrievalConfig",
    "SpreadingActivationRetriever",
    "create_retriever",
]

# Conditionally add HybridRetriever if available
if _HYBRID_AVAILABLE:
    __all__.append("HybridRetriever")

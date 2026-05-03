"""
Retrieval System Base Classes
==============================

Defines the interface for memory retrieval systems compatible with the 
Nouse memory framework. Supports multiple backends:

  - Spreading activation (CA-based associative recall)
  - Vector similarity (FAISS/semantic search)  
  - Keyword matching (fallback)
  - Hybrid (combines multiple methods)

Each retriever implements a common interface that works with any CA engine
or memory store, making the retrieval system decoupled from storage.

Usage:
    from ..retrieval.base import MemoryRetriever
    
    class MyCustomRetriever(MemoryRetriever):
        def retrieve(self, query: str, **kwargs) -> RetrievalResult:
            # Custom retrieval logic
            return RetrievalResult(...)
    
    retriever = MyCustomRetriever()
    results = retriever.retrieve("domain_failure mechanisms")
"""

from __future__ import annotations

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Retrieval Data Model
# ============================================================================

@dataclass
class RetrievedMemory:
    """A single memory retrieved from the store."""
    id: str                              # Memory node ID
    content: str                         # Human-readable representation
    score: float                         # Relevance/activation score (0-1)
    tier: str = "unknown"                # Which tier it came from
    semantic_type: str = "observation"   # Type of memory
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "tier": self.tier,
            "semantic_type": self.semantic_type,
        }


@dataclass
class RetrievalResult:
    """Complete result from a memory retrieval query."""
    query: str                           # The original query
    results: List[RetrievedMemory] = field(default_factory=list)
    top_k: int = 5                       # Requested number of results
    confidence: float = 0.0              # Overall confidence in results
    retrieval_method: str = "unknown"    # Which method was used
    elapsed_ms: float = 0.0              # Time taken to retrieve
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "top_k": self.top_k,
            "confidence": self.confidence,
            "retrieval_method": self.retrieval_method,
            "elapsed_ms": self.elapsed_ms,
        }


# ============================================================================
# Retrieval Configuration
# ============================================================================

@dataclass
class RetrievalConfig:
    """Configuration for memory retrieval."""
    
    # General settings
    default_top_k: int = 5               # Default number of results
    min_score_threshold: float = 0.01    # Minimum score to include in results
    
    # Tier search scope
    search_all_tiers: bool = True        # Search all tiers or just short-term?
    
    # Method-specific settings
    spreading_activation_steps: int = 20 # Max steps for activation spread
    faiss_top_k: int = 50               # FAISS candidate pool size (before ranking)
    
    # Hybrid retrieval weights (when combining methods)
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        "spreading_activation": 0.4,
        "vector_similarity": 0.4,
        "keyword_match": 0.2,
    })


# ============================================================================
# Retriever Abstract Base Class
# ============================================================================

class MemoryRetriever(ABC):
    """
    Abstract base class for memory retrieval systems.
    
    Subclasses implement the `retrieve` method to search memories using 
    their specific algorithm (spreading activation, vector similarity, etc.).
    
    The retriever works with any CA engine or memory store — it doesn't 
    depend on a specific storage backend. It only needs access to:
      - Node states and content
      - Edge weights (for graph-based retrieval)
      - Metadata (semantic type, tier membership)
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
    
    @abstractmethod
    def retrieve(
        self, 
        query: str,
        memories_by_tier: Optional[Dict[str, Dict[str, Any]]] = None,
        edges: Optional[Dict[Tuple[str, str], float]] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve memories matching a query.
        
        Args:
            query: The search query string
            memories_by_tier: {tier_name: {nid: memory_data}} — memories to search
                Each memory_data should have at least: {"content": str, "state": float}
            edges: {(src, tgt): weight} — associative graph edges (for graph-based retrieval)
            top_k: Number of results (overrides config default)
            
        Returns:
            RetrievalResult with ranked memories and metadata
        """
        ...
    
    def retrieve_from_store(
        self, 
        query: str,
        store: Any,  # CAEngine or LaceMemory instance
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Convenience method to retrieve from a specific store instance."""
        # Extract memories and edges from the store
        if hasattr(store, 'nodes'):  # CAEngine
            nodes_by_tier = {"short_term": {nid: {"content": n.id, "state": n.state} 
                                              for nid, n in store.nodes.items()}}
            edges = {(k[0], k[1]): e.weight for k, e in store.edges.items()}
        elif hasattr(store, 'short_term'):  # LaceMemory
            nodes_by_tier = {
                "short_term": {nid: {"content": n.content if hasattr(n, 'content') else nid, 
                                      "state": n.state} 
                               for nid, n in store.short_term.items()},
                "mid_term": {nid: {"content": n.content if hasattr(n, 'content') else nid,
                                    "state": n.state}
                             for nid, n in store.mid_term.items()},
                "long_term": {nid: {"content": n.content if hasattr(n, 'content') else nid,
                                     "state": n.state}
                              for nid, n in store.long_term.items()},
            }
            edges = {(k[0], k[1]): v.weight for k, v in store.edges.items()}
        else:
            raise ValueError(f"Unsupported store type: {type(store)}")
        
        return self.retrieve(
            query=query,
            memories_by_tier=nodes_by_tier,
            edges=edges,
            top_k=top_k,
        )


# ============================================================================
# Retriever Registry (for dynamic loading)
# ============================================================================

_RETRIEVER_REGISTRY: Dict[str, type] = {}

def register_retriever(name: str):
    """Decorator to register a retriever class."""
    def decorator(cls: type) -> type:
        _RETRIEVER_REGISTRY[name] = cls
        return cls
    return decorator


def get_retriever(name: str, **kwargs) -> MemoryRetriever:
    """Get a registered retriever by name."""
    if name not in _RETRIEVER_REGISTRY:
        raise KeyError(f"Unknown retriever '{name}'. Registered: {list(_RETRIEVER_REGISTRY.keys())}")
    
    return _RETRIEVER_REGISTRY[name](**kwargs)


def list_retrievers() -> List[str]:
    """List all registered retriever names."""
    return list(_RETRIEVER_REGISTRY.keys())

"""
Nouse Hermes — LACE-Powered Memory System (General Framework)
==============================================================

A multi-tiered memory architecture using cellular automata as the substrate 
for storing, retrieving, and consolidating memories. Designed to be general-
purpose enough for contribution back to both LACE and MemGPT/mem0 projects.

Architecture:
  - Core CA engine (decoupled from LACE) — standalone rule engine
  - Modular encoders — string diagrams, embeddings, knowledge graphs, raw text
  - Flexible retrieval — spreading activation, FAISS vector search, hybrid
  - Multi-agent support — per-agent stores + shared pool

Usage:
    # General-purpose (text input → CA memory):
    from nouse_hermes.memory import LaceMemory, create_embedding_encoder
    
    mem = LaceMemory()
    encoder = create_embedding_encoder(grid_size=(100, 100))
    
    items, _ = encoder.encode("domain_failure causes system_failure")
    mem.encode_batch(items)
    mem.evolve(steps=5)
    results = mem.retrieve("surface phenomena", top_k=3)

    # Multi-agent:
    from nouse_hermes.memory.agents import create_multi_agent_system
    
    system = create_multi_agent_system()
    system.register_agent("physics_analyst")
    system.register_agent("knowledge_engineer")
    
    # Each agent has its own memory + shared pool access

Key Design Principles:
  1. Organic forgetting — decay built into CA rules, no manual cleanup
  2. Modular encoders — accept any structured data (text, graphs, embeddings)
  3. Pluggable retrieval — spreading activation, FAISS, or hybrid
  4. Multi-agent ready — per-agent stores + shared knowledge pool
  5. Decoupled from LACE — standalone CA engine can be used independently

Files:
    core/ca_engine.py        — Standalone CA rule engine (decoupled)
    encoders/base.py          — Encoder ABC interface
    encoders/embedding.py     — Embedding-based text encoder
    encoders/string_diagram.py — Causal string diagram encoder  
    retrieval/base.py         — Retriever ABC interface
    retrieval/spreading_activation.py — Spreading activation engine
    retrieval/faiss_retriever.py   — FAISS vector similarity search
    agents/agent_memory.py    — Multi-agent memory management
"""

# Core exports — use relative imports for local execution
from .core.ca_engine import (
    CAEngine,
    NodeState,
    Edge,
    CARule,
    MemoryDecayRule,
    ConsolidationRule,
    SpreadingActivationRule,
    AssociativeStrengtheningRule,
    create_memory_engine,
)

# Encoder exports
from .encoders.base import (
    MemoryEncoder,
    EncodingResult,
    EncodedNode,
    EncodedEdge,
    SemanticType,
    CausalRole,
    register_encoder,
    get_encoder,
    list_encoders,
)
from .encoders.embedding import (
    EmbeddingEncoder,
    EmbeddingConfig,
    create_embedding_encoder,
)
from .encoders.string_diagram import (
    StringDiagramEncoder,
    causal_triplets_to_diagram,
    create_string_diagram_encoder,
)

# Retrieval exports
from .retrieval.base import (
    MemoryRetriever,
    RetrievalResult,
    RetrievedMemory,
    RetrievalConfig,
    register_retriever,
    get_retriever,
    list_retrievers,
)
from .retrieval.spreading_activation import (
    SpreadingActivationRetriever,
    create_retriever,
)
from .retrieval.faiss_retriever import FAISSRetriever

# Production default: Hybrid retriever (FAISS + CA evolution + ranking)
try:
    from .retrieval.hybrid_retriever import HybridRetriever
except ImportError:
    HybridRetriever = None  # sentence-transformers/FAISS not installed

# Agent exports
from .agents.agent_memory import (
    AgentMemory,
    SharedMemoryPool,
    MultiAgentMemorySystem,
    create_multi_agent_system,
)

# Legacy LaceMemory wrapper (for backward compatibility)
# Note: This is kept for existing code but new code should use CAEngine directly
try:
    from .lace_memory import LaceMemory
except ImportError:
    LaceMemory = None  # type: ignore

__all__ = [
    # Core engine
    "CAEngine",
    "NodeState",
    "Edge", 
    "CARule",
    "MemoryDecayRule",
    "ConsolidationRule",
    "SpreadingActivationRule",
    "AssociativeStrengtheningRule",
    "create_memory_engine",
    
    # Encoders
    "MemoryEncoder",
    "EncodingResult",
    "EncodedNode",
    "EncodedEdge",
    "SemanticType",
    "CausalRole",
    "register_encoder",
    "get_encoder",
    "list_encoders",
    "EmbeddingEncoder",
    "EmbeddingConfig",
    "create_embedding_encoder",
    "StringDiagramEncoder",
    "causal_triplets_to_diagram",
    "create_string_diagram_encoder",
    
    # Retrieval
    "MemoryRetriever",
    "RetrievalResult", 
    "RetrievedMemory",
    "RetrievalConfig",
    "register_retriever",
    "get_retriever",
    "list_retrievers",
    "SpreadingActivationRetriever",
    "create_retriever",
    "FAISSRetriever",
    "HybridRetriever",
    
    # Multi-agent
    "AgentMemory",
    "SharedMemoryPool",
    "MultiAgentMemorySystem",
    "create_multi_agent_system",
    
    # Legacy
    "LaceMemory",
]

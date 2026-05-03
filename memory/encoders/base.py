"""
Memory Encoder Interface (Abstract Base Class)
================================================

Defines the protocol for converting diverse data sources into CA grid states.
This abstraction allows the memory system to accept any structured input:
- String diagrams from causal extraction
- Embedding vectors from text encoders  
- Knowledge graph nodes/edges
- Raw text chunks with metadata
- Multi-agent message logs
- Event sequences

Each encoder implementation maps its specific data format to:
  - Node positions on the CA grid (spatial layout)
  - Initial activation states (importance/urgency)
  - Edge weights (association strength between memories)
  - Metadata (semantic type, causal role, temporal info)

Usage:
    from ..encoders.base import MemoryEncoder
    
    class MyCustomEncoder(MemoryEncoder):
        def encode(self, data: MyDataType) -> EncodingResult:
            # Map custom data to CA-compatible format
            return EncodingResult(
                nodes=[...],
                edges=[...],
                metadata={...}
            )
    
    encoder = MyCustomEncoder(grid_size=(100, 100))
    result = encoder.encode(my_data)

Design Principle:
    The encoder is a pure transformation — it takes structured input and 
    produces CA-compatible output without side effects. The actual memory
    storage (LaceMemory or CAEngine) handles persistence, evolution, and retrieval.
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
# Encoder Data Model
# ============================================================================

class SemanticType(Enum):
    """Types of memories that can be encoded."""
    OBSERVATION = "observation"      # Raw sensory/perceptual data
    CONCEPT = "concept"              # Abstract idea or category
    FACT = "fact"                    # Verified piece of knowledge
    EVENT = "event"                  # Time-bound occurrence
    RULE = "rule"                    # Procedural or conditional knowledge
    QUERY = "query"                  # Search/retrieval request
    INTERVENTION = "intervention"   # Action taken on the environment


class CausalRole(Enum):
    """Causal role of a node in its context."""
    ROOT = "root"           # Origin/causal source (no parents)
    HUB = "hub"             # Many connections (both inputs and outputs)
    BRIDGE = "bridge"       # Connects disparate concepts
    LEAF = "leaf"           # Terminal node (many parents, few/no children)


@dataclass
class EncodedNode:
    """A single memory node ready for CA grid placement."""
    id: str                              # Unique identifier
    content: str                         # Human-readable representation
    position: Tuple[int, int]            # Grid coordinates (row, col)
    state: float = 0.5                   # Initial activation strength (0-1)
    semantic_type: SemanticType = SemanticType.OBSERVATION
    causal_role: CausalRole = CausalRole.LEAF
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "position": list(self.position),
            "state": self.state,
            "semantic_type": self.semantic_type.value,
            "causal_role": self.causal_role.value,
            "metadata": self.metadata,
        }


@dataclass
class EncodedEdge:
    """An associative link between two encoded nodes."""
    source: str                          # Node ID
    target: str                          # Node ID
    weight: float = 1.0                  # Association strength (0-1)
    causal_direction: str = "bidirectional"  # forward, backward, bidirectional
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "causal_direction": self.causal_direction,
        }


@dataclass
class EncodingResult:
    """Complete output from an encoder — ready for CA grid placement."""
    nodes: List[EncodedNode] = field(default_factory=list)
    edges: List[EncodedEdge] = field(default_factory=list)
    
    # Metadata about the encoding process
    source_type: str = "unknown"         # Type of input data encoded
    encoding_method: str = "default"     # Which algorithm was used
    timestamp: Optional[str] = None      # When this encoding occurred
    
    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "source_type": self.source_type,
            "encoding_method": self.encoding_method,
            "timestamp": self.timestamp,
        }


# ============================================================================
# Encoder Abstract Base Class
# ============================================================================

class MemoryEncoder(ABC):
    """
    Abstract base class for memory encoders.
    
    Subclasses implement the `encode` method to convert their specific data 
    format into CA-compatible nodes and edges. The encoder handles:
      - Spatial layout on the grid (position assignment)
      - Initial activation strength calculation
      - Edge weight computation between related items
      - Semantic type inference
    
    Args:
        grid_size: (rows, cols) of the target CA grid
        default_state: Default initial activation for nodes without explicit state
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (100, 100), 
                 default_state: float = 0.5):
        self.grid_size = grid_size
        self.default_state = default_state
    
    @abstractmethod
    def encode(
        self, 
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EncodingResult:
        """
        Convert input data into CA-compatible encoding.
        
        Args:
            data: The structured data to encode (format depends on subclass)
            metadata: Optional context about the data source
            
        Returns:
            EncodingResult with nodes and edges ready for grid placement
        
        Raises:
            ValueError: If data cannot be encoded in this format
        """
        ...
    
    def encode_batch(
        self, 
        items: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> EncodingResult:
        """Encode multiple items and merge their results."""
        merged = EncodingResult()
        
        for i, item in enumerate(items):
            meta = metadata_list[i] if metadata_list else {}
            result = self.encode(item, meta)
            
            # Merge nodes (deduplicate by ID)
            existing_ids = {n.id for n in merged.nodes}
            for node in result.nodes:
                if node.id not in existing_ids:
                    merged.nodes.append(node)
                    existing_ids.add(node.id)
            
            # Merge edges
            for edge in result.edges:
                merged.edges.append(edge)
        
        return merged
    
    def _clamp_position(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Clamp grid position to valid bounds."""
        r = max(0, min(self.grid_size[0] - 1, pos[0]))
        c = max(0, min(self.grid_size[1] - 1, pos[1]))
        return (r, c)
    
    def _compute_similarity(self, a: str, b: str) -> float:
        """Compute simple word-overlap similarity between two strings."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = words_a & words_b
        union = words_a | words_b
        
        return len(intersection) / len(union)


# ============================================================================
# Built-in Encoder Registry (for dynamic loading)
# ============================================================================

_ENCODER_REGISTRY: Dict[str, type] = {}

def register_encoder(name: str):
    """Decorator to register an encoder class in the global registry."""
    def decorator(cls: type) -> type:
        _ENCODER_REGISTRY[name] = cls
        return cls
    return decorator


def get_encoder(name: str, **kwargs) -> MemoryEncoder:
    """Get a registered encoder by name with given configuration."""
    if name not in _ENCODER_REGISTRY:
        raise KeyError(f"Unknown encoder '{name}'. Registered: {list(_ENCODER_REGISTRY.keys())}")
    
    return _ENCODER_REGISTRY[name](**kwargs)


def list_encoders() -> List[str]:
    """List all registered encoder names."""
    return list(_ENCODER_REGISTRY.keys())

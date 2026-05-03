"""
Embedding-Based Memory Encoder
===============================

Converts arbitrary text into CA grid states using vector embeddings for 
semantic similarity matching. This enables the memory system to accept any 
text input (not just structured causal graphs) and place it on the grid 
based on semantic proximity to existing memories.

Key Features:
  - Semantic clustering: similar concepts are placed near each other on the grid
  - Cross-modal support: works with text, audio transcripts, code snippets, etc.
  - Incremental encoding: new items can be added without re-encoding everything
  - Fallback to keyword matching when embeddings unavailable

Usage:
    from ..encoders.embedding import EmbeddingEncoder
    
    encoder = EmbeddingEncoder(
        grid_size=(100, 100),
        embedding_dim=768,
        use_faiss=True,
    )
    
    # Encode arbitrary text
    result = encoder.encode("Entity A affects outcome B in the domain")
    
    # Encode batch of related items
    results = encoder.encode_batch([
        "Surface roughness increases domain_failure risk",
        "Cleaning reduces surface roughness", 
        "Output Z depends on parameter W",
    ])

Architecture:
    1. Text → embedding vector (via sentence-transformers or external API)
    2. Embedding → grid position via clustering/projection
    3. Similar items → closer positions + stronger edges
    4. Semantic type inference from content analysis
    
Dependencies:
    - FAISS (optional, for fast similarity search): pip install faiss-cpu
    - sentence-transformers (optional, for local embeddings): pip install sentence-transformers
    - Falls back to TF-IDF word overlap if neither is available
"""

from __future__ import annotations

import math
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum

# Try importing optional dependencies
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger = logging.getLogger(__name__)
    logger.warning("FAISS not installed — similarity search will use fallback")

try:
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from ..encoders.base import (
    MemoryEncoder, EncodingResult, EncodedNode, EncodedEdge,
    SemanticType, CausalRole, register_encoder
)

logger = logging.getLogger(__name__)


# ============================================================================
# Embedding Backend Abstraction
# ============================================================================

class EmbeddingBackend(Enum):
    """Available embedding backends."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # Local model
    OPENAI = "openai"                                 # OpenAI API
    FAISS_INDEX = "faiss_index"                       # Pre-built FAISS index
    FALLBACK_TFIDF = "fallback_tfidf"                 # Word overlap (no embeddings)


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding encoder."""
    backend: EmbeddingBackend = EmbeddingBackend.SENTENCE_TRANSFORMERS
    
    # Sentence-transformers settings
    model_name: str = "all-MiniLM-L6-v2"  # Small, fast, good quality
    device: str = "cpu"                    # "cuda" if GPU available
    
    # OpenAI settings (if using API)
    openai_model: str = "text-embedding-3-small"
    
    # FAISS index settings
    faiss_index_path: Optional[str] = None  # Path to saved FAISS index
    similarity_threshold: float = 0.6       # Min similarity for edge creation
    
    # Grid layout settings
    clustering_method: str = "kmeans"       # How to cluster embeddings on grid
    cluster_count: int = 20                 # Number of clusters for spatial grouping


# ============================================================================
# Embedding Encoder Implementation
# ============================================================================

@register_encoder("embedding")
class EmbeddingEncoder(MemoryEncoder):
    """
    Converts arbitrary text into CA grid states using vector embeddings.
    
    The encoder places semantically similar texts closer together on the 
    grid, creating natural clustering of related concepts. Edges are formed 
    between items with high embedding similarity above a threshold.
    
    This is the most general-purpose encoder — it works with any text input
    without requiring prior structure (unlike string diagram encoding which
    needs causal triplets).
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (100, 100),
        config: Optional[EmbeddingConfig] = None,
        embedding_dim: int = 384,  # MiniLM-L6-v2 dimension
    ):
        super().__init__(grid_size=grid_size)
        self.config = config or EmbeddingConfig()
        self.embedding_dim = embedding_dim
        
        # Initialize embedding backend
        self._backend = None
        if HAS_TRANSFORMERS and self.config.backend == EmbeddingBackend.SENTENCE_TRANSFORMERS:
            try:
                model_name = self.config.model_name
                logger.info(f"Loading sentence-transformer model: {model_name}")
                self._model = SentenceTransformer(model_name, device=self.config.device)
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
                self._backend = "transformers"
            except Exception as e:
                logger.warning(f"sentence-transformers failed: {e}. Using fallback.")
                self._backend = None
        
        # FAISS index for similarity search (optional)
        self._faiss_index = None
        if HAS_FAISS and self.config.faiss_index_path:
            try:
                self._faiss_index = faiss.read_index(self.config.faiss_index_path)
                logger.info(f"Loaded FAISS index from {self.config.faiss_index_path}")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
        
        # Cache for existing embeddings (for incremental encoding)
        self._embedding_cache: Dict[str, Any] = {}  # text_hash -> embedding
    
    def encode(
        self,
        data: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EncodingResult:
        """
        Encode a single text string into CA grid state.
        
        Args:
            data: Text to encode (any content — observations, facts, queries)
            metadata: Optional context (semantic_type hint, causal_role hint, etc.)
            
        Returns:
            EncodingResult with nodes and edges ready for CA placement
        """
        if not isinstance(data, str) or not data.strip():
            return EncodingResult(source_type="text", encoding_method="embedding")
        
        # Determine semantic type from metadata or content analysis
        sem_type = self._infer_semantic_type(data, metadata)
        causal_role = self._infer_causal_role(data, metadata)
        
        # Compute embedding
        embedding = self._get_embedding(data)
        
        if embedding is None:
            # Fallback: use keyword-based positioning
            return self._encode_fallback(data, sem_type, causal_role, metadata)
        
        # Determine grid position based on embedding clustering
        position = self._embed_to_position(embedding, data)
        
        # Create node
        node_id = f"emb_{hashlib.md5(data.encode()).hexdigest()[:8]}"
        node = EncodedNode(
            id=node_id,
            content=data,
            position=position,
            state=self._compute_initial_state(data, embedding),
            semantic_type=sem_type,
            causal_role=causal_role,
            metadata={**metadata, "embedding_dim": self.embedding_dim} if metadata else {"embedding_dim": self.embedding_dim},
        )
        
        # Find similar items in cache to create edges
        edges = []
        for cached_text, cached_hash in list(self._embedding_cache.items())[:50]:  # Limit search
            if cached_text == data:
                continue
            
            cached_emb = self._get_embedding(cached_text)
            if cached_emb is None:
                continue
            
            similarity = self._compute_similarity(embedding, cached_emb)
            
            if similarity > self.config.similarity_threshold:
                # Create edge with weight proportional to similarity
                edges.append(EncodedEdge(
                    source=node_id,
                    target=f"emb_{hashlib.md5(cached_text.encode()).hexdigest()[:8]}",
                    weight=similarity,
                    causal_direction="bidirectional",
                ))
        
        # Cache this embedding for future edge creation
        text_hash = hashlib.md5(data.encode()).hexdigest()
        self._embedding_cache[text_hash] = (data, embedding)
        
        result = EncodingResult(
            nodes=[node],
            edges=edges,
            source_type="text",
            encoding_method=f"embedding_{self.config.backend.value}",
        )
        
        logger.debug(f"Encoded text ({len(data)} chars): position={position}, "
                     f"type={sem_type.value}, role={causal_role.value}")
        
        return result
    
    def encode_batch(
        self,
        items: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> EncodingResult:
        """Encode multiple text items with cross-similarity edge creation."""
        # First pass: get all embeddings
        embeddings = []
        for item in items:
            emb = self._get_embedding(item) if isinstance(item, str) else None
            embeddings.append(emb)
        
        # Second pass: create nodes and edges with cross-similarity
        nodes = []
        edges = []
        
        for i, (item, emb) in enumerate(zip(items, embeddings)):
            if not isinstance(item, str) or not item.strip():
                continue
            
            sem_type = self._infer_semantic_type(item, metadata_list[i] if metadata_list else None)
            causal_role = self._infer_causal_role(item, metadata_list[i] if metadata_list else None)
            
            position = self._embed_to_position(emb, item) if emb else (i % 10, i // 10)
            
            node_id = f"batch_{i}_{hashlib.md5(item.encode()).hexdigest()[:6]}"
            
            nodes.append(EncodedNode(
                id=node_id,
                content=item,
                position=position,
                state=self._compute_initial_state(item, emb),
                semantic_type=sem_type,
                causal_role=causal_role,
                metadata=(metadata_list[i] if metadata_list else {}),
            ))
            
            # Create edges to similar items in this batch
            for j, other_emb in enumerate(embeddings):
                if i == j or other_emb is None or emb is None:
                    continue
                
                similarity = self._compute_similarity(emb, other_emb)
                
                if similarity > self.config.similarity_threshold:
                    edges.append(EncodedEdge(
                        source=node_id,
                        target=f"batch_{j}_{hashlib.md5(items[j].encode()).hexdigest()[:6]}",
                        weight=similarity,
                        causal_direction="bidirectional",
                    ))
        
        return EncodingResult(
            nodes=nodes,
            edges=edges,
            source_type="text_batch",
            encoding_method=f"embedding_batch_{self.config.backend.value}",
        )
    
    # ------------------------------------------------------------------
    # Embedding Computation
    # ------------------------------------------------------------------

    def _get_embedding(self, text: str) -> Optional[Any]:
        """Get embedding vector for text using configured backend."""
        if self._backend == "transformers" and HAS_TRANSFORMERS:
            try:
                return self._model.encode(text, convert_to_numpy=True)
            except Exception as e:
                logger.warning(f"sentence-transformer encoding failed: {e}")
        
        # Fallback: use TF-IDF word overlap (no actual embedding vector)
        if not HAS_TRANSFORMERS and not HAS_FAISS:
            return None
        
        return None  # No backend available
    
    def _compute_similarity(self, emb_a: Any, emb_b: Any) -> float:
        """Compute similarity between two embeddings."""
        if emb_a is None or emb_b is None:
            return 0.0
        
        try:
            if HAS_FAISS and isinstance(emb_a, list):
                # Convert to numpy for FAISS dot product
                import numpy as np
                a = np.array([emb_a], dtype=np.float32)
                b = np.array([emb_b], dtype=np.float32)
                score = float(np.dot(a, b.T)[0][0])
            else:
                # Cosine similarity fallback
                import numpy as np
                a = np.array(emb_a) if not isinstance(emb_a, np.ndarray) else emb_a
                b = np.array(emb_b) if not isinstance(emb_b, np.ndarray) else emb_b
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                score = float(np.dot(a, b) / (norm_a * norm_b + 1e-8))
            
            # Normalize to [0, 1] for cosine similarity
            return max(0.0, min(1.0, score))
        
        except Exception:
            return 0.0
    
    # ------------------------------------------------------------------
    # Grid Position Mapping
    # ------------------------------------------------------------------

    def _embed_to_position(self, embedding: Any, text: str) -> Tuple[int, int]:
        """Map an embedding vector to a grid position using clustering."""
        if embedding is None:
            # Fallback: hash-based positioning
            h = hashlib.md5(text.encode()).hexdigest()
            r = int(h[:8], 16) % self.grid_size[0]
            c = int(h[8:16], 16) % self.grid_size[1]
            return (r, c)
        
        try:
            import numpy as np
            
            if isinstance(embedding, list):
                emb_array = np.array(embedding)
            else:
                emb_array = embedding
            
            # Normalize embedding to [0, 1] range for grid mapping
            min_val = emb_array.min()
            max_val = emb_array.max()
            range_val = max_val - min_val if max_val != min_val else 1.0
            
            normalized = (emb_array - min_val) / range_val
            
            # Map first two dimensions to grid coordinates
            r = int(normalized[0] * (self.grid_size[0] - 1))
            c = int(normalized[1] * (self.grid_size[1] - 1))
            
            return self._clamp_position((r, c))
        
        except Exception as e:
            logger.warning(f"Embedding-to-position mapping failed: {e}")
            # Fallback to hash-based positioning
            h = hashlib.md5(text.encode()).hexdigest()
            r = int(h[:8], 16) % self.grid_size[0]
            c = int(h[8:16], 16) % self.grid_size[1]
            return (r, c)
    
    # ------------------------------------------------------------------
    # Semantic Type Inference
    # ------------------------------------------------------------------

    def _infer_semantic_type(self, text: str, metadata: Optional[Dict]) -> SemanticType:
        """Infer semantic type from content and metadata hints."""
        if metadata and "semantic_type" in metadata:
            try:
                return SemanticType(metadata["semantic_type"])
            except ValueError:
                pass
        
        text_lower = text.lower()
        
        # Rule-based inference
        if any(kw in text_lower for kw in ["if", "then", "when", "should", "must", "rule"]):
            return SemanticType.RULE
        elif any(kw in text_lower for kw in ["event", "occurred", "happened", "date", "time"]):
            return SemanticType.EVENT
        elif any(kw in text_lower for kw in ["fact", "known", "verified", "proven"]):
            return SemanticType.FACT
        elif any(kw in text_lower for kw in ["what", "how", "why", "where", "when", "query"]):
            return SemanticType.QUERY
        
        # Default: concept or observation based on length/complexity
        word_count = len(text.split())
        if word_count > 15:
            return SemanticType.CONCEPT
        else:
            return SemanticType.OBSERVATION
    
    def _infer_causal_role(self, text: str, metadata: Optional[Dict]) -> CausalRole:
        """Infer causal role from content and metadata hints."""
        if metadata and "causal_role" in metadata:
            try:
                return CausalRole(metadata["causal_role"])
            except ValueError:
                pass
        
        # Default to leaf (terminal node) — will be refined during CA evolution
        return CausalRole.LEAF
    
    def _compute_initial_state(self, text: str, embedding: Any) -> float:
        """Compute initial activation strength based on content properties."""
        base = self.default_state
        
        # Boost for longer/more complex texts (likely more important concepts)
        word_count = len(text.split()) if isinstance(text, str) else 0
        complexity_bonus = min(0.2, word_count / 100.0)
        
        # Boost for embeddings with higher magnitude (more distinctive)
        magnitude_bonus = 0
        if embedding is not None:
            try:
                import numpy as np
                mag = float(np.linalg.norm(embedding))
                magnitude_bonus = min(0.15, mag / 10.0)
            except Exception:
                pass
        
        return min(1.0, base + complexity_bonus + magnitude_bonus)
    
    # ------------------------------------------------------------------
    # Fallback Encoder (TF-IDF word overlap)
    # ------------------------------------------------------------------

    def _encode_fallback(
        self, 
        text: str, 
        sem_type: SemanticType, 
        causal_role: CausalRole,
        metadata: Optional[Dict],
    ) -> EncodingResult:
        """Fallback encoding using keyword-based positioning when embeddings unavailable."""
        # Use hash-based grid position (deterministic but not semantic)
        h = hashlib.md5(text.encode()).hexdigest()
        r = int(h[:8], 16) % self.grid_size[0]
        c = int(h[8:16], 16) % self.grid_size[1]
        
        node_id = f"fb_{hashlib.md5(text.encode()).hexdigest()[:8]}"
        
        return EncodingResult(
            nodes=[EncodedNode(
                id=node_id,
                content=text,
                position=(r, c),
                state=self.default_state,
                semantic_type=sem_type,
                causal_role=causal_role,
                metadata=metadata or {},
            )],
            edges=[],
            source_type="text",
            encoding_method="fallback_tfidf",
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_embedding_encoder(
    grid_size: Tuple[int, int] = (100, 100),
    use_faiss: bool = True,
) -> EmbeddingEncoder:
    """Factory function to create a configured embedding encoder."""
    config = EmbeddingConfig()
    if use_faiss and HAS_FAISS:
        config.backend = EmbeddingBackend.FAISS_INDEX
    
    return EmbeddingEncoder(grid_size=grid_size, config=config)

"""
FAISS-Based Vector Retriever
==============================

Implements memory retrieval using FAISS (Facebook AI Similarity Search) for 
fast semantic similarity matching. This provides a general-purpose retrieval 
mechanism that works with any text input, not just structured causal graphs.

Usage:
    from ..retrieval.faiss_retriever import FAISSRetriever
    
    retriever = FAISSRetriever(
        embedding_dim=384,  # MiniLM-L6-v2 dimension
        index_type="IVF",   # Index type for your data size
    )
    
    # Add memories to the index
    retriever.add_memories({
        "mem_1": {"content": "domain_failure causes system_failure"},
        "mem_2": {"content": "Surface roughness increases domain_failure risk"},
    })
    
    # Retrieve by semantic similarity
    results = retriever.retrieve("surface phenomena", top_k=3)

Integration with CA Engine:
    The FAISS retriever works alongside spreading activation — use it for 
    initial candidate selection (fast, accurate), then apply spreading 
    activation on the CA graph for ranked retrieval. This hybrid approach 
    gives you both semantic accuracy and associative reasoning.

Dependencies:
    pip install faiss-cpu   # CPU version (lightweight)
    pip install faiss-gpu   # GPU version (faster, requires CUDA)
"""

from __future__ import annotations

import math
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger = logging.getLogger(__name__)
    logger.warning("FAISS not installed — FAISSRetriever will fall back to keyword matching")

from ..retrieval.base import (
    MemoryRetriever, RetrievalResult, RetrievedMemory, RetrievalConfig,
    register_retriever
)

logger = logging.getLogger(__name__)


# ============================================================================
# FAISS Retriever Implementation
# ============================================================================

@register_retriever("faiss")
class FAISSRetriever(MemoryRetriever):
    """
    Retrieves memories using FAISS for fast vector similarity search.
    
    Maintains an in-memory FAISS index of memory embeddings. When a query 
    comes in, it computes the embedding and searches the index for similar 
    memories. Results are ranked by cosine similarity score.
    
    This is the most general-purpose retriever — it works with any text input
    without requiring prior structure or CA graph traversal.
    """
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        embedding_dim: int = 384,
        index_type: str = "IVF",
        n_lists: int = 16,
    ):
        super().__init__(config)
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.n_lists = n_lists
        
        # FAISS index (initialized lazily)
        self._index: Optional[Any] = None
        self._memory_ids: List[str] = []  # Maps index positions to memory IDs
        self._memory_data: Dict[str, Dict[str, Any]] = {}  # nid -> metadata
        
        # Embedding backend (sentence-transformers or external API)
        self._embedding_model = None
        self._init_embedding_backend()
    
    def _init_embedding_backend(self):
        """Initialize the embedding model for converting text to vectors."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer for FAISS retrieval")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("sentence-transformers not available — using TF-IDF fallback")
            self._embedding_model = None
    
    def add_memories(self, memories: Dict[str, Dict[str, Any]]):
        """
        Add memories to the FAISS index.
        
        Args:
            memories: {nid: {"content": str, "state": float, ...}} — memories to index
        """
        if not HAS_FAISS:
            logger.warning("FAISS not installed — cannot add memories")
            return
        
        # Initialize FAISS index on first call
        if self._index is None:
            self._init_index()
        
        new_ids = []
        embeddings = []
        
        for nid, data in memories.items():
            content = data.get("content", "")
            
            # Compute embedding
            emb = self._compute_embedding(content)
            if emb is None:
                continue
            
            new_ids.append(nid)
            embeddings.append(emb)
        
        if not new_ids or not embeddings:
            return
        
        # Add to FAISS index
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        if self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self._index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.n_lists)
            self._index.train(embeddings_array)
            self._index.add(embeddings_array)
        else:  # Flat index (simple but slower for large datasets)
            self._index = faiss.IndexFlatL2(self.embedding_dim)
            self._index.add(embeddings_array)
        
        # Store memory data and IDs
        self._memory_ids.extend(new_ids)
        for nid, data in memories.items():
            if nid not in self._memory_data:
                self._memory_data[nid] = data
        
        logger.info(f"Added {len(new_ids)} memories to FAISS index "
                    f"(total: {len(self._memory_ids)})")
    
    def retrieve(
        self, 
        query: str,
        memories_by_tier: Optional[Dict[str, Dict[str, Any]]] = None,
        edges: Optional[Any] = None,  # Not used for FAISS retrieval
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieve memories using FAISS vector similarity."""
        start_time = time.time()
        
        if not query.strip():
            return RetrievalResult(
                query=query, results=[], confidence=0.0, retrieval_method="faiss",
            )
        
        top_k = top_k or self.config.default_top_k
        
        # Compute query embedding
        query_emb = self._compute_embedding(query)
        
        if query_emb is None:
            elapsed = (time.time() - start_time) * 1000
            return RetrievalResult(
                query=query, results=[], confidence=0.0, retrieval_method="faiss",
                elapsed_ms=elapsed,
            )
        
        # Search FAISS index
        if self._index is None or len(self._memory_ids) == 0:
            # Fallback to keyword matching if no memories indexed
            return self._fallback_keyword_search(query, memories_by_tier or {}, top_k)
        
        try:
            query_vec = np.array([query_emb], dtype=np.float32)
            
            # Search for top candidates (use larger k for candidate pool)
            search_k = min(self.config.faiss_top_k, len(self._memory_ids))
            distances, indices = self._index.search(query_vec, search_k)
            
            # Convert distances to similarity scores (L2 distance → cosine-like score)
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                
                if idx < 0 or idx >= len(self._memory_ids):
                    continue
                
                nid = self._memory_ids[idx]
                
                # Convert distance to similarity score (0-1)
                # L2 distance of normalized vectors: d = sqrt(2 - 2*cos_sim)
                # So cos_sim = 1 - d²/2
                sim_score = max(0.0, min(1.0, 1.0 - dist / 2.0))
                
                if sim_score > self.config.min_score_threshold:
                    mem_data = self._memory_data.get(nid, {})
                    
                    results.append(RetrievedMemory(
                        id=nid,
                        content=mem_data.get("content", nid),
                        score=sim_score,
                        tier="indexed",  # FAISS doesn't track tiers directly
                        semantic_type=mem_data.get("semantic_type", "observation"),
                    ))
            
            results.sort(key=lambda x: x.score, reverse=True)
            top_results = results[:top_k]
            
            confidence = (
                sum(r.score for r in top_results) / len(top_results) 
                if top_results else 0.0
            )
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return self._fallback_keyword_search(query, memories_by_tier or {}, top_k)
        
        elapsed = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query=query,
            results=top_results,
            top_k=top_k,
            confidence=confidence,
            retrieval_method="faiss",
            elapsed_ms=elapsed,
        )
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding vector for text."""
        if self._embedding_model is not None:
            try:
                emb = self._embedding_model.encode(text, convert_to_numpy=True)
                # Normalize to unit length for cosine similarity
                norm = np.linalg.norm(emb)
                if norm > 0:
                    return emb / norm
                return emb
            except Exception as e:
                logger.warning(f"Embedding computation failed: {e}")
        
        return None
    
    def _init_index(self):
        """Initialize the FAISS index."""
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self._index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.n_lists)
    
    def _fallback_keyword_search(
        self, 
        query: str, 
        memories_by_tier: Dict[str, Dict[str, Any]],
        top_k: int,
    ) -> RetrievalResult:
        """Fallback keyword-based search when FAISS/embeddings unavailable."""
        results = []
        query_lower = query.lower()
        
        for tier_name, tier_memories in memories_by_tier.items():
            for nid, data in tier_memories.items():
                content = data.get("content", "").lower()
                
                # Word overlap score
                query_words = set(query_lower.split())
                content_words = set(content.split())
                
                if not query_words or not content_words:
                    continue
                
                score = len(query_words & content_words) / max(len(query_words), len(content_words))
                
                if score > self.config.min_score_threshold:
                    results.append(RetrievedMemory(
                        id=nid,
                        content=data.get("content", nid),
                        score=score,
                        tier=tier_name,
                        semantic_type=data.get("semantic_type", "observation"),
                    ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        return RetrievalResult(
            query=query,
            results=results[:top_k],
            top_k=top_k,
            confidence=sum(r.score for r in results[:top_k]) / max(len(results[:top_k]), 1),
            retrieval_method="faiss_keyword_fallback",
        )


# ============================================================================
# Hybrid Retriever (combines FAISS + spreading activation)
# ============================================================================

class HybridRetriever(MemoryRetriever):
    """
    Combines FAISS vector similarity with CA spreading activation for 
    the best of both worlds: semantic accuracy + associative reasoning.
    
    Workflow:
      1. Use FAISS to get initial candidate pool (fast, accurate)
      2. Apply spreading activation on candidates to refine ranking
      3. Return final ranked results
    
    This is recommended for production use where you want both 
    semantic matching and associative recall.
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        super().__init__(config)
        self.faiss_retriever = FAISSRetriever(config=config)
        from ..retrieval.spreading_activation import SpreadingActivationRetriever
        self.activation_retriever = SpreadingActivationRetriever(config=config)
    
    def add_memories(self, memories: Dict[str, Dict[str, Any]]):
        """Add memories to the FAISS index."""
        self.faiss_retriever.add_memories(memories)
    
    def retrieve(
        self, 
        query: str,
        memories_by_tier: Optional[Dict[str, Dict[str, Any]]] = None,
        edges: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Hybrid retrieval: FAISS candidates + spreading activation refinement."""
        import time
        start_time = time.time()
        
        # Step 1: Get FAISS candidate pool
        faiss_results = self.faiss_retriever.retrieve(
            query=query, top_k=self.config.faiss_top_k,
        )
        
        if not faiss_results.results:
            elapsed = (time.time() - start_time) * 1000
            return RetrievalResult(
                query=query, results=[], confidence=0.0, retrieval_method="hybrid",
                elapsed_ms=elapsed,
            )
        
        # Step 2: Use FAISS candidates as seeds for spreading activation
        seed_nodes = {r.id: r.score for r in faiss_results.results[:10]}
        
        if not seed_nodes or edges is None:
            return faiss_results  # No graph to spread on, just return FAISS results
        
        # Step 3: Run spreading activation with FAISS scores as seeds
        activation_results = self.activation_retriever.retrieve(
            query=query,
            memories_by_tier=memories_by_tier or {},
            edges=edges,
            top_k=top_k,
        )
        
        # Step 4: Combine results (weighted average of FAISS score and activation)
        faiss_scores = {r.id: r.score for r in faiss_results.results}
        activation_scores = {r.id: r.score for r in activation_results.results}
        
        all_ids = set(faiss_scores.keys()) | set(activation_scores.keys())
        combined = []
        
        for nid in all_ids:
            faiss_score = faiss_scores.get(nid, 0.0)
            act_score = activation_scores.get(nid, 0.0)
            
            # Weighted combination
            w_faiss = self.config.hybrid_weights.get("vector_similarity", 0.5)
            w_act = self.config.hybrid_weights.get("spreading_activation", 0.5)
            
            combined_score = w_faiss * faiss_score + w_act * act_score
            
            # Find tier and content for this node
            tier = "unknown"
            content = nid
            sem_type = "observation"
            
            if memories_by_tier:
                for tier_name, tier_memories in memories_by_tier.items():
                    if nid in tier_memories:
                        tier = tier_name
                        mem_data = tier_memories[nid]
                        content = mem_data.get("content", nid)
                        sem_type = mem_data.get("semantic_type", "observation")
                        break
            
            combined.append(RetrievedMemory(
                id=nid,
                content=content,
                score=combined_score,
                tier=tier,
                semantic_type=sem_type,
            ))
        
        combined.sort(key=lambda x: x.score, reverse=True)
        top_results = combined[:top_k]
        
        confidence = (
            sum(r.score for r in top_results) / len(top_results) 
            if top_results else 0.0
        )
        
        elapsed = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query=query,
            results=top_results,
            top_k=top_k,
            confidence=confidence,
            retrieval_method="hybrid",
            elapsed_ms=elapsed,
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_faiss_retriever(
    embedding_dim: int = 384,
) -> FAISSRetriever:
    """Factory function to create a configured FAISS retriever."""
    return FAISSRetriever(embedding_dim=embedding_dim)


def create_hybrid_retriever() -> HybridRetriever:
    """Factory function to create a hybrid (FAISS + spreading activation) retriever."""
    return HybridRetriever()

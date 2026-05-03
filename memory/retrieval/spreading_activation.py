"""
Spreading Activation Retriever
===============================

Implements memory retrieval via spreading activation on the associative graph.
This is the core retrieval mechanism for CA-based memories — it models how 
human associative recall works: activating one concept triggers related memories
through weighted connections, with strength decaying over "distance" (hops).

Usage:
    from ..retrieval.spreading_activation import SpreadingActivationRetriever
    
    retriever = SpreadingActivationRetriever(
        spread_rate=0.3,
        decay_per_hop=0.9,
        max_steps=20,
    )
    
    results = retriever.retrieve("surface roughness", memories_by_tier={...})

Algorithm:
  1. Seed nodes activated based on query similarity to existing memories
  2. Activation spreads through weighted edges (proportional to weight)
  3. Each hop applies decay factor (closer = stronger activation)
  4. Nodes with highest final activation = retrieved memories
  5. Early stopping when convergence reached

This is analogous to:
  - Neural network forward propagation
  - PageRank-style importance scoring  
  - Human associative memory recall
"""

from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

from ..retrieval.base import (
    MemoryRetriever, RetrievalResult, RetrievedMemory, RetrievalConfig,
    register_retriever
)

logger = logging.getLogger(__name__)


@register_retriever("spreading_activation")
class SpreadingActivationRetriever(MemoryRetriever):
    """
    Retrieves memories via spreading activation on the associative graph.
    
    Activates seed nodes based on query similarity, then propagates 
    activation through weighted edges. Nodes with highest final activation
    are returned as retrieved memories.
    """
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        spread_rate: float = 0.3,
        decay_per_hop: float = 0.9,
        max_steps: int = 20,
    ):
        super().__init__(config)
        self.spread_rate = spread_rate
        self.decay_per_hop = decay_per_hop
        self.max_steps = max_steps
    
    def retrieve(
        self,
        query: str,
        memories_by_tier: Optional[Dict[str, Dict[str, Any]]] = None,
        edges: Optional[Dict[Tuple[str, str], float]] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieve memories via spreading activation."""
        start_time = time.time()
        
        if not query.strip():
            return RetrievalResult(
                query=query, results=[], confidence=0.0,
                retrieval_method="spreading_activation",
            )
        
        top_k = top_k or self.config.default_top_k
        
        # Step 1: Find seed nodes based on query similarity
        seeds = self._find_seed_nodes(query, memories_by_tier or {})
        
        if not seeds:
            elapsed = (time.time() - start_time) * 1000
            return RetrievalResult(
                query=query, results=[], confidence=0.0,
                retrieval_method="spreading_activation", elapsed_ms=elapsed,
            )
        
        # Step 2: Initialize activations with seeds
        all_ids = set()
        for tier_memories in (memories_by_tier or {}).values():
            all_ids.update(tier_memories.keys())
        
        activations = {nid: 0.0 for nid in all_ids}
        
        for nid, score in seeds.items():
            if nid in activations:
                activations[nid] = max(activations[nid], score)
        
        # Step 3: Run spreading activation
        for step in range(self.max_steps):
            new_activations = dict(activations)
            
            for nid, current_state in activations.items():
                if current_state < self.config.min_score_threshold:
                    continue
                
                # Get neighbors and spread activation
                neighbors = self._get_neighbors(nid, edges or {})
                
                for neighbor_id, weight in neighbors:
                    if neighbor_id not in new_activations:
                        continue
                    
                    spread = current_state * weight * self.spread_rate * self.decay_per_hop
                    new_activations[neighbor_id] = min(1.0, new_activations[neighbor_id] + spread)
            
            activations = new_activations
            
            # Check convergence (early stopping)
            max_change = max(abs(activations[nid] - new_activations.get(nid, 0))
                           for nid in activations if nid in new_activations)
            
            if max_change < 1e-4:
                logger.debug(f"Activation converged after {step + 1} steps")
                break
        
        # Step 4: Collect and rank results
        retrieved = [
            (nid, score) for nid, score in activations.items()
            if score > self.config.min_score_threshold
        ]
        
        retrieved.sort(key=lambda x: x[1], reverse=True)
        top_results = retrieved[:top_k]
        
        # Calculate confidence based on activation spread
        total_activation = sum(s for _, s in retrieved)
        confidence = min(1.0, total_activation / max(len(retrieved), 1)) if retrieved else 0.0
        
        # Build result objects with tier info
        results = []
        for nid, score in top_results:
            tier = "unknown"
            content = nid
            sem_type = "observation"
            
            for tier_name, tier_memories in (memories_by_tier or {}).items():
                if nid in tier_memories:
                    tier = tier_name
                    mem_data = tier_memories[nid]
                    content = mem_data.get("content", nid)
                    sem_type = mem_data.get("semantic_type", "observation")
                    break
            
            results.append(RetrievedMemory(
                id=nid,
                content=content,
                score=score,
                tier=tier,
                semantic_type=sem_type,
            ))
        
        elapsed = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query=query,
            results=results,
            top_k=top_k,
            confidence=confidence,
            retrieval_method="spreading_activation",
            elapsed_ms=elapsed,
        )
    
    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _find_seed_nodes(
        self, 
        query: str, 
        memories_by_tier: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Find best matching existing memories to use as activation seeds."""
        seeds = {}
        query_lower = query.lower()
        
        for tier_name, tier_memories in memories_by_tier.items():
            for nid, memory_data in tier_memories.items():
                score = self._compute_similarity(query_lower, memory_data)
                
                if score > 0.1:  # Minimum relevance threshold
                    seeds[nid] = score
        
        return seeds
    
    def _compute_similarity(self, query: str, memory_data: Dict[str, Any]) -> float:
        """Compute similarity between query and a memory's content."""
        content = memory_data.get("content", "").lower()
        
        if not content:
            return 0.0
        
        # Word overlap score
        query_words = set(query.split())
        content_words = set(content.split())
        
        if not query_words or not content_words:
            return 0.0
        
        overlap = len(query_words & content_words) / max(len(query_words), len(content_words))
        
        # Boost by memory's current activation state
        state = memory_data.get("state", 0.5)
        
        return overlap * state
    
    def _get_neighbors(
        self, 
        nid: str, 
        edges: Dict[Tuple[str, str], float],
    ) -> List[Tuple[str, float]]:
        """Get all neighbors of a node with their edge weights."""
        neighbors = []
        
        for (src, tgt), weight in edges.items():
            if src == nid:
                neighbors.append((tgt, weight))
            elif tgt == nid:
                neighbors.append((src, weight))
        
        return neighbors


# ============================================================================
# Convenience Functions
# ============================================================================

def create_retriever(
    spread_rate: float = 0.3,
    decay_per_hop: float = 0.9,
    max_steps: int = 20,
) -> SpreadingActivationRetriever:
    """Factory function to create a configured spreading activation retriever."""
    return SpreadingActivationRetriever(
        spread_rate=spread_rate,
        decay_per_hop=decay_per_hop,
        max_steps=max_steps,
    )

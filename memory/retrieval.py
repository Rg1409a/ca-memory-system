"""
Memory Retrieval via Spreading Activation
==========================================

Implements retrieval from the LACE-based memory system using a spreading 
activation model inspired by cognitive science and neural network theory.

Retrieval Process:
  1. Seed node(s) activated based on query similarity
  2. Activation spreads through associative edges (weighted propagation)
  3. Nodes with highest final activation = retrieved memories
  4. Tier-aware retrieval (can search all tiers or specific ones)

This models how human memory works: recalling one concept triggers related 
memories through associative links, with strength proportional to the 
connection weight and decay over "distance" (number of hops).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Retrieval Configuration
# ============================================================================

@dataclass
class RetrievalConfig:
    """Configuration for spreading activation retrieval."""
    
    # Propagation parameters
    max_steps: int = 20                    # Maximum propagation steps
    spread_rate: float = 0.3               # Activation spread per hop (0-1)
    decay_per_hop: float = 0.9             # Decay factor per hop (0-1)
    
    # Thresholds
    min_activation_threshold: float = 0.01 # Minimum activation to be considered retrieved
    
    # Tier search scope
    default_tiers: List[str] = field(default_factory=lambda: [
        "short_term", "mid_term", "long_term"
    ])
    
    # Top-K results
    default_top_k: int = 5                 # Number of top results to return


# ============================================================================
# Spreading Activation Engine
# ============================================================================

class SpreadingActivationRetriever:
    """
    Implements memory retrieval via spreading activation on the associative graph.
    
    The algorithm:
      1. Activate seed nodes based on query similarity
      2. Iteratively spread activation through weighted edges
      3. Apply decay at each hop (closer = stronger)
      4. Return top-K nodes by final activation state
    
    This is analogous to:
      - Neural network forward propagation
      - PageRank-style importance scoring  
      - Human associative memory recall
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()

    # ------------------------------------------------------------------
    # Main Retrieval Interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        memories_by_tier: Dict[str, Dict[str, Any]],
        edges: Dict[Tuple[str, str], float],
        top_k: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve memories matching a query via spreading activation.
        
        Args:
            query: The search query string
            memories_by_tier: Dict mapping tier names to {nid: memory_data} dicts
                Each memory_data should have at least: {"state": float, "content": str}
            edges: Associative graph edges as {(src, tgt): weight} dict
            top_k: Number of results (default from config)
            max_steps: Max propagation steps (default from config)
            
        Returns:
            List of (memory_id, activation_score) sorted by score descending
        """
        top_k = top_k or self.config.default_top_k
        max_steps = max_steps or self.config.max_steps
        
        # Step 1: Find seed nodes based on query similarity
        seeds = self._find_seed_nodes(query, memories_by_tier)
        
        if not seeds:
            logger.warning(f"No seed nodes found for query: '{query[:50]}...'")
            return []
        
        # Step 2: Initialize activation with seeds
        activations = {nid: 0.0 for nid in self._all_memory_ids(memories_by_tier)}
        
        for nid, score in seeds.items():
            activations[nid] = score
        
        # Step 3: Run spreading activation
        for step in range(max_steps):
            new_activations = dict(activations)
            
            for nid, current_state in activations.items():
                if current_state < self.config.min_activation_threshold:
                    continue
                
                # Get neighbors and spread activation
                neighbors = self._get_neighbors(nid, edges)
                
                for neighbor_id, weight in neighbors:
                    if neighbor_id not in new_activations:
                        continue
                    
                    # Activation spreads with decay
                    spread = current_state * weight * self.config.spread_rate * self.config.decay_per_hop
                    
                    # Additive spreading with saturation at 1.0
                    new_activations[neighbor_id] = min(
                        1.0, 
                        new_activations[neighbor_id] + spread
                    )
            
            activations = new_activations
            
            # Check convergence (early stopping)
            max_change = max(abs(activations[nid] - new_activations.get(nid, 0))
                           for nid in activations if nid in new_activations)
            
            if max_change < 0.001:
                logger.debug(f"Activation converged after {step + 1} steps")
                break
        
        # Step 4: Collect and rank results
        results = [
            (nid, score) for nid, score in activations.items()
            if score > self.config.min_activation_threshold
        ]
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]

    # ------------------------------------------------------------------
    # Tier-Specific Retrieval
    # ------------------------------------------------------------------

    def retrieve_from_tier(
        self,
        query: str,
        tier_name: str,
        tier_memories: Dict[str, Any],
        edges: Dict[Tuple[str, str], float],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Retrieve memories from a specific tier only."""
        return self.retrieve(
            query=query,
            memories_by_tier={tier_name: tier_memories},
            edges=edges,
            top_k=top_k,
        )

    def retrieve_cross_tier(
        self,
        query: str,
        short_term: Dict[str, Any],
        mid_term: Dict[str, Any],
        long_term: Dict[str, Any],
        edges: Dict[Tuple[str, str], float],
        boost_long_term: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Cross-tier retrieval with optional boosting for long-term memories.
        
        Long-term memories (semantic knowledge) get a slight activation boost
        when they're relevant to the query, reflecting how foundational concepts
        are more readily accessible in human memory.
        """
        # Get base results from all tiers
        results = self.retrieve(
            query=query,
            memories_by_tier={
                "short_term": short_term,
                "mid_term": mid_term, 
                "long_term": long_term,
            },
            edges=edges,
        )
        
        if not boost_long_term:
            return results
        
        # Apply tier-based boosting to results
        boosted_results = []
        for nid, score in results:
            final_score = score
            
            # Boost long-term memories slightly (they're more "stable")
            if nid in long_term:
                final_score *= 1.1
            
            # Penalize very old short-term memories (less reliable)
            if nid in short_term and tier_memories.get(nid, {}).get("age", 0) > 30:
                final_score *= 0.9
            
            boosted_results.append((nid, final_score))
        
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results

    # ------------------------------------------------------------------
    # Seed Node Discovery
    # ------------------------------------------------------------------

    def _find_seed_nodes(
        self,
        query: str,
        memories_by_tier: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Find the best matching existing memories to use as activation seeds."""
        seeds = {}
        query_lower = query.lower()
        
        for tier_name, tier_memories in memories_by_tier.items():
            for nid, memory_data in tier_memories.items():
                # Simple keyword similarity (can be enhanced with embeddings)
                score = self._compute_similarity(query_lower, memory_data)
                
                if score > 0.1:  # Minimum relevance threshold
                    seeds[nid] = score
        
        return seeds

    def _compute_similarity(
        self, 
        query: str, 
        memory_data: Dict[str, Any]
    ) -> float:
        """Compute similarity between query and a memory's content."""
        content = memory_data.get("content", "").lower()
        
        if not content:
            return 0.0
        
        # Keyword overlap score
        query_words = set(query.split())
        content_words = set(content.split())
        
        if not query_words or not content_words:
            return 0.0
        
        overlap = len(query_words & content_words) / max(len(query_words), len(content_words))
        
        # Boost by memory's current activation state (more active memories are more retrievable)
        state = memory_data.get("state", 0.5)
        
        return overlap * state

    # ------------------------------------------------------------------
    # Graph Helpers
    # ------------------------------------------------------------------

    def _get_neighbors(
        self, 
        nid: str, 
        edges: Dict[Tuple[str, str], float]
    ) -> List[Tuple[str, float]]:
        """Get all neighbors of a node with their edge weights."""
        neighbors = []
        
        for (src, tgt), weight in edges.items():
            if src == nid:
                neighbors.append((tgt, weight))
            elif tgt == nid:
                neighbors.append((src, weight))
        
        return neighbors

    def _all_memory_ids(
        self, 
        memories_by_tier: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Get all memory IDs across tiers."""
        ids = []
        for tier_memories in memories_by_tier.values():
            ids.extend(tier_memories.keys())
        return list(set(ids))  # Deduplicate


# ============================================================================
# Convenience Functions
# ============================================================================

def create_retriever(config: Optional[RetrievalConfig] = None) -> SpreadingActivationRetriever:
    """Factory function to create a configured retriever."""
    return SpreadingActivationRetriever(config or RetrievalConfig())


def simple_keyword_search(
    query: str, 
    memories: Dict[str, Any],
    min_score: float = 0.1,
) -> List[Tuple[str, float]]:
    """Simple keyword-based search (fallback when spreading activation isn't available)."""
    results = []
    query_lower = query.lower()
    
    for nid, data in memories.items():
        content = data.get("content", "").lower()
        
        # Word overlap score
        query_words = set(query_lower.split())
        content_words = set(content.split())
        
        if not query_words or not content_words:
            continue
        
        score = len(query_words & content_words) / max(len(query_words), len(content_words))
        
        if score >= min_score:
            results.append((nid, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

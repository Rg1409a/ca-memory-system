"""
Memory Tier Management
======================

Manages the three-tier memory architecture:

  SHORT-TERM (Working Buffer)
    - Current grid state in CA simulation
    - ~10-50 active memories at a time
    - High temporal resolution, transient
    - Maps to: immediate context, recent observations
    
  MID-TERM (Pattern Buffer)  
    - Persistent subgraphs that survived N generations
    - Episodic memories with moderate stability
    - Can be promoted back to short-term via retrieval
    - Maps to: recent experiences, learned patterns
    
  LONG-TERM (Structural Memory)
    - Graph topology anchors with high betweenness centrality
    - Semantic knowledge, core concepts, fundamental rules
    - Highly resistant to decay
    - Maps to: domain expertise, foundational knowledge

Promotion Criteria:
  Short → Mid:   Survived N consecutive steps above threshold
  Mid → Long:    High betweenness centrality in associative graph
  
Demotion/Pruning:
  Memories below activation threshold for extended periods are pruned
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
# Tier Configuration
# ============================================================================

@dataclass
class TierConfig:
    """Configuration for a single memory tier."""
    name: str
    max_capacity: int                    # Maximum memories in this tier
    decay_rate: float                    # Base decay rate for this tier
    consolidation_threshold: float       # State threshold for promotion
    
    # Promotion criteria
    min_persistence_steps: int = 0       # Min steps to survive before promotion eligible
    centrality_threshold: float = 0.0    # Betweenness centrality for long-term promotion
    
    # Retrieval boost (when promoted back from lower tier)
    retrieval_boost: float = 1.0         # Multiplier when retrieved into short-term


@dataclass
class MemoryTierSystemConfig:
    """Configuration for the full three-tier memory system."""
    
    # Short-term config
    short_term: TierConfig = field(default_factory=lambda: TierConfig(
        name="short_term",
        max_capacity=50,
        decay_rate=0.02,
        consolidation_threshold=0.7,
        min_persistence_steps=10,
    ))
    
    # Mid-term config  
    mid_term: TierConfig = field(default_factory=lambda: TierConfig(
        name="mid_term",
        max_capacity=200,
        decay_rate=0.005,
        consolidation_threshold=0.5,
        min_persistence_steps=30,
        centrality_threshold=0.05,
    ))
    
    # Long-term config
    long_term: TierConfig = field(default_factory=lambda: TierConfig(
        name="long_term",
        max_capacity=1000,
        decay_rate=0.001,  # Very slow decay for long-term
        consolidation_threshold=0.3,
        min_persistence_steps=50,
        centrality_threshold=0.02,
    ))


# ============================================================================
# Tier Manager
# ============================================================================

class MemoryTierManager:
    """
    Manages promotion and demotion between memory tiers.
    
    Tracks persistence history for each memory node and determines when
    to promote memories to higher tiers based on survival, centrality,
    and causal importance metrics.
    """
    
    def __init__(self, config: Optional[MemoryTierSystemConfig] = None):
        self.config = config or MemoryTierSystemConfig()
        
        # Persistence tracking per node
        self.persistence_history: Dict[str, List[float]] = defaultdict(list)
        
        # Tier membership tracking
        self.tier_membership: Dict[str, str] = {}  # nid -> tier name
        
        # Promotion history for analytics
        self.promotion_log: List[Dict[str, Any]] = []
        
        # Metrics
        self.metrics = {
            "total_promotions": {"short_to_mid": 0, "mid_to_long": 0},
            "total_demotions": {"mid_to_short": 0, "long_to_mid": 0},
            "total_pruned": 0,
        }

    # ------------------------------------------------------------------
    # Promotion Logic
    # ------------------------------------------------------------------

    def check_promotions(
        self,
        short_term: Dict[str, Any],      # nid -> {state, role, age, ...}
        mid_term: Dict[str, Any],
        long_term: Dict[str, Any],
        associative_graph: Any,           # NetworkX graph with edge weights
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Check all memories for promotion eligibility.
        
        Returns:
            (promotions_to_mid, promotions_to_long) — lists of (nid, new_tier_name)
        """
        to_mid = []
        to_long = []
        
        # --- Short → Mid term check ---
        for nid, data in short_term.items():
            if self._is_eligible_for_promotion(
                nid, data, "short_term", "mid_term"
            ):
                to_mid.append((nid, "mid_term"))
        
        # --- Mid → Long term check (requires centrality computation) ---
        if mid_term and len(mid_term) > 10:
            centrality = self._compute_centrality(
                list(mid_term.keys()), 
                list(long_term.keys()),
                associative_graph
            )
            
            for nid, data in mid_term.items():
                node_centrality = centrality.get(nid, 0.0)
                
                if (node_centrality > self.config.mid_term.centrality_threshold and
                    self._is_eligible_for_promotion(
                        nid, data, "mid_term", "long_term"
                    )):
                    to_long.append((nid, "long_term"))
        
        # Log promotions
        for nid, tier in to_mid:
            self.promotion_log.append({
                "nid": nid,
                "from_tier": "short_term",
                "to_tier": tier,
                "timestamp": self._get_timestamp(),
            })
            self.metrics["total_promotions"]["short_to_mid"] += 1
        
        for nid, tier in to_long:
            self.promotion_log.append({
                "nid": nid,
                "from_tier": "mid_term", 
                "to_tier": tier,
                "timestamp": self._get_timestamp(),
            })
            self.metrics["total_promotions"]["mid_to_long"] += 1
        
        return to_mid, to_long

    def _is_eligible_for_promotion(
        self,
        nid: str,
        data: Dict[str, Any],
        current_tier: str,
        target_tier: str,
    ) -> bool:
        """Check if a memory is eligible for promotion to the next tier."""
        history = self.persistence_history.get(nid, [])
        
        # Must have survived minimum persistence steps
        min_steps = getattr(self.config, current_tier).min_persistence_steps
        if len(history) < min_steps:
            return False
        
        # Must maintain state above threshold for sufficient fraction of time
        recent = history[-min_steps:]
        threshold = getattr(self.config, target_tier).consolidation_threshold
        survival_ratio = sum(1 for s in recent if s > threshold * 0.5) / len(recent)
        
        return survival_ratio > 0.6

    # ------------------------------------------------------------------
    # Demotion Logic  
    # ------------------------------------------------------------------

    def check_demotions(
        self,
        mid_term: Dict[str, Any],
        long_term: Dict[str, Any],
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Check for memories that should be demoted to lower tiers.
        
        Returns:
            (demotions_to_short, demotions_to_mid) — lists of (nid, new_tier_name)
        """
        to_short = []
        to_mid = []
        
        # Mid → Short: if state drops significantly below threshold
        for nid, data in mid_term.items():
            state = data.get("state", 0.0)
            if state < self.config.mid_term.consolidation_threshold * 0.3:
                to_short.append((nid, "short_term"))
        
        # Long → Mid: rarely happens, only for severely degraded anchors
        for nid, data in long_term.items():
            state = data.get("state", 0.0)
            if state < self.config.long_term.consolidation_threshold * 0.2:
                to_mid.append((nid, "mid_term"))
        
        return to_short, to_mid

    # ------------------------------------------------------------------
    # Pruning Logic
    # ------------------------------------------------------------------

    def prune_dead_memories(
        self,
        short_term: Dict[str, Any],
        mid_term: Dict[str, Any],
        threshold: float = 0.05,
    ) -> List[str]:
        """Remove memories that have decayed below the pruning threshold."""
        pruned = []
        
        for nid in list(short_term.keys()):
            if short_term[nid].get("state", 0) < threshold:
                del short_term[nid]
                self.persistence_history.pop(nid, None)
                pruned.append(nid)
        
        # Also prune from mid-term (but less aggressively)
        for nid in list(mid_term.keys()):
            if mid_term[nid].get("state", 0) < threshold * 0.5:
                del mid_term[nid]
                self.persistence_history.pop(nid, None)
                pruned.append(nid)
        
        self.metrics["total_pruned"] += len(pruned)
        return pruned

    # ------------------------------------------------------------------
    # Capacity Management
    # ------------------------------------------------------------------

    def enforce_capacity(
        self,
        tier_name: str,
        memories: Dict[str, Any],
        max_capacity: int,
    ) -> List[str]:
        """
        Enforce maximum capacity for a tier by removing weakest memories.
        
        Returns list of pruned node IDs.
        """
        if len(memories) <= max_capacity:
            return []
        
        # Sort by strength (state * age factor) and remove weakest
        sorted_memories = sorted(
            memories.items(),
            key=lambda x: x[1].get("state", 0) * min(1.0, x[1].get("age", 0) / 50.0),
        )
        
        to_remove = len(memories) - max_capacity
        pruned = []
        
        for i in range(to_remove):
            nid = sorted_memories[i][0]
            del memories[nid]
            self.persistence_history.pop(nid, None)
            pruned.append(nid)
        
        logger.info(f"Enforced capacity on {tier_name}: removed {to_remove} weakest memories")
        return pruned

    # ------------------------------------------------------------------
    # Retrieval Boost (promote retrieved mid/long-term back to short-term)
    # ------------------------------------------------------------------

    def boost_retrieved_memories(
        self,
        retrieved_ids: List[str],
        mid_term: Dict[str, Any],
        long_term: Dict[str, Any],
        short_term: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """
        When memories are retrieved from mid/long-term, boost their activation
        and promote them back to short-term for active processing.
        
        Returns list of (nid, boosted_state) tuples.
        """
        boosted = []
        
        for nid in retrieved_ids:
            source_tier = None
            source_data = None
            
            if nid in mid_term:
                source_tier = "mid_term"
                source_data = mid_term[nid]
            elif nid in long_term:
                source_tier = "long_term" 
                source_data = long_term[nid]
            
            if source_data is None:
                continue
            
            # Calculate boost based on tier and current state
            base_boost = self.config.mid_term.retrieval_boost if source_tier == "mid_term" else 1.5
            current_state = source_data.get("state", 0.0)
            
            boosted_state = min(1.0, current_state * base_boost + 0.2)
            
            # Add to short-term with boost
            short_term[nid] = {**source_data, "state": boosted_state}
            self.tier_membership[nid] = "short_term"
            
            boosted.append((nid, boosted_state))
        
        return boosted

    # ------------------------------------------------------------------
    # Centrality Computation (for long-term promotion)
    # ------------------------------------------------------------------

    def _compute_centrality(
        self,
        mid_ids: List[str],
        long_ids: List[str],
        graph: Any,
    ) -> Dict[str, float]:
        """Compute betweenness centrality for mid-term candidates."""
        all_ids = set(mid_ids + long_ids)
        
        # Build subgraph for efficiency
        try:
            subgraph = graph.subgraph(all_ids)
            
            if len(subgraph.nodes) < 50:
                return nx.betweenness_centrality(subgraph)
            else:
                return nx.betweenness_centrality(subgraph, k=min(20, len(subgraph.nodes)))
        except Exception as e:
            logger.warning(f"Centrality computation failed: {e}")
            return {nid: 0.0 for nid in mid_ids}

    # ------------------------------------------------------------------
    # Analytics & Reporting
    # ------------------------------------------------------------------

    def get_tier_stats(self) -> Dict[str, Any]:
        """Get statistics about tier distribution and health."""
        return {
            "metrics": self.metrics.copy(),
            "promotion_history_count": len(self.promotion_log),
            "persistence_tracking_nodes": len(self.persistence_history),
        }

    def get_promotion_history(
        self, 
        limit: int = 50,
        nid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get promotion history, optionally filtered by node ID."""
        if nid:
            return [p for p in self.promotion_log if p["nid"] == nid][-limit:]
        return self.promotion_log[-limit:]

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_tier_system(config: Optional[MemoryTierSystemConfig] = None) -> MemoryTierManager:
    """Factory function to create a configured tier management system."""
    return MemoryTierManager(config or MemoryTierSystemConfig())

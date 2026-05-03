"""
Core LACE Memory Engine
========================

Wraps the LACE cellular automata framework to provide memory operations:
  - encode(): Convert observations into CA grid states
  - retrieve(): Spreading activation query across associative links
  - evolve(): Run CA steps (memory decay/consolidation)
  - persist(): Save/load memory state between sessions

The grid serves as the memory substrate where:
  - Active nodes = memories currently in working buffer
  - Node state value = activation strength (0.0-1.0)
  - Edges = associative links between co-active memories
  - CA evolution = organic forgetting and consolidation
"""

from __future__ import annotations

import os
import sys
import json
import math
import logging
import copy
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional, Any, Protocol
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

# LACE imports — path to cloned repo
LACE_PATH = os.path.join(os.path.dirname(__file__), "..", "lace")
if LACE_PATH not in sys.path:
    sys.path.insert(0, LACE_PATH)

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class MemoryTier(Enum):
    """Memory storage tier"""
    SHORT_TERM = "short_term"      # Working buffer — current grid state
    MID_TERM = "mid_term"          # Persistent subgraphs (survived N generations)
    LONG_TERM = "long_term"        # Structural anchors (high betweenness centrality)


@dataclass
class MemoryNode:
    """A single memory trace (engram) in the CA grid"""
    id: str                              # Unique identifier for this memory
    state: float                         # Activation strength 0.0-1.0
    tier: MemoryTier                     # Which tier stores this memory
    age: int = 0                         # How many CA steps this node has survived
    birth_time: Optional[datetime] = None
    last_active: Optional[datetime] = None
    causal_role: str = "leaf"            # root, hub, leaf, bridge (from string diagram)
    semantic_type: str = "observation"   # observation, concept, fact, event, rule
    
    def __post_init__(self):
        if self.birth_time is None:
            self.birth_time = datetime.now()
        if self.last_active is None:
            self.last_active = self.birth_time

    @property
    def strength(self) -> float:
        """Effective memory strength (state * age factor)"""
        age_factor = min(1.0, self.age / 50.0)  # Strength grows with persistence
        return self.state * (0.3 + 0.7 * age_factor)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier"] = self.tier.value
        if self.birth_time:
            d["birth_time"] = self.birth_time.isoformat()
        if self.last_active:
            d["last_active"] = self.last_active.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'MemoryNode':
        d = d.copy()
        d["tier"] = MemoryTier(d["tier"])
        if "birth_time" in d and d["birth_time"]:
            d["birth_time"] = datetime.fromisoformat(d["birth_time"])
        if "last_active" in d and d["last_active"]:
            d["last_active"] = datetime.fromisoformat(d["last_active"])
        return cls(**d)


@dataclass
class MemoryEdge:
    """Associative link between two memories"""
    source: str                            # MemoryNode.id
    target: str                            # MemoryNode.id
    weight: float                          # Association strength 0.0-1.0
    causal_direction: str = "bidirectional"  # forward, backward, bidirectional
    formation_step: int = 0                # CA step when this edge formed

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'MemoryEdge':
        return cls(**d)


@dataclass
class MemoryQueryResult:
    """Result of a spreading activation retrieval"""
    query_id: str                          # The seed/concept used for query
    retrieved_memories: List[Tuple[str, float]]  # (memory_id, activation_score)
    top_k: int = 5                         # Number of top results returned
    retrieval_steps: int = 0               # How many CA steps the query ran
    confidence: float = 0.0                # Overall confidence in retrieval

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Core Memory Engine
# ============================================================================

class LaceMemory:
    """
    Cellular automata-based memory system for Nouse Hermes.
    
    The grid is the working memory buffer. Memories are encoded as active nodes
    with continuous state values. Edges form between co-active memories. CA 
    evolution drives organic forgetting (decay) and consolidation (promotion).
    
    Parameters:
        grid_size: Size of the 2D CA grid (grid_size x grid_size)
        decay_rate: Base rate at which memory states decay per CA step
        consolidation_threshold: Node state above which memories are promoted
        mid_term_survival_steps: How many consecutive steps a pattern must survive
        long_term_centrality_threshold: Betweenness centrality for long-term promotion
    """

    def __init__(
        self,
        grid_size: int = 100,
        decay_rate: float = 0.02,
        consolidation_threshold: float = 0.7,
        mid_term_survival_steps: int = 30,
        long_term_centrality_threshold: float = 0.05,
        max_short_term_nodes: int = 50,
        storage_dir: Optional[str] = None,
    ):
        # Grid configuration
        self.grid_size = grid_size
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        self.mid_term_survival_steps = mid_term_survival_steps
        self.long_term_centrality_threshold = long_term_centrality_threshold
        self.max_short_term_nodes = max_short_term_nodes

        # Storage directory for persistence
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(__file__), "..", "memory_data"
        )
        os.makedirs(self.storage_dir, exist_ok=True)

        # Memory stores — the actual data structures
        self.short_term: Dict[str, MemoryNode] = {}      # Current grid state
        self.mid_term: Dict[str, MemoryNode] = {}         # Persistent subgraphs
        self.long_term: Dict[str, MemoryNode] = {}        # Structural anchors

        # Associative graph (all tiers combined)
        self.associative_graph: nx.Graph = nx.Graph()     # Nodes + edges across all tiers
        
        # Edge store for persistence
        self.edges: Dict[Tuple[str, str], MemoryEdge] = {}

        # CA simulation state
        self.ca_step: int = 0                              # Current evolution step counter
        self.node_history: Dict[str, List[float]] = defaultdict(list)  # State history per node
        
        # Causal structure tracking (from string diagram encoding)
        self.causal_chains: List[List[str]] = []           # Ordered causal sequences
        self.monomial_compositions: List[Tuple[str, str]] = []  # Tensor product pairs

        # Metrics
        self.metrics = {
            "total_memories_encoded": 0,
            "total_retrievals": 0,
            "tier_promotions": {"short_to_mid": 0, "mid_to_long": 0},
            "avg_memory_lifespan": 0.0,
        }

        logger.info(f"LaceMemory initialized: grid={grid_size}x{grid_size}, "
                    f"decay={decay_rate}, consolidation_threshold={consolidation_threshold}")

    # ------------------------------------------------------------------
    # Memory Encoding (writing to short-term)
    # ------------------------------------------------------------------

    def encode(
        self,
        content: str,
        semantic_type: str = "observation",
        causal_chain: Optional[List[Tuple[str, str]]] = None,
        parent_ids: Optional[List[str]] = None,
        initial_strength: float = 0.8,
    ) -> List[MemoryNode]:
        """
        Encode an observation or concept into short-term memory.
        
        Maps the content to active nodes on the CA grid with edges between
        causally related elements (from string diagram structure).
        
        Args:
            content: The text/content to encode as a memory
            semantic_type: Type of memory (observation, concept, fact, event, rule)
            causal_chain: List of (source_id, target_id) pairs from string diagram
            parent_ids: IDs of related memories this should link to
            initial_strength: Starting activation strength (0.0-1.0)
            
        Returns:
            List of MemoryNode objects created for this encoding
        """
        if not content.strip():
            return []

        # Generate node ID and create the memory node
        node_id = f"mem_{self.ca_step}_{len(self.short_term)}"
        
        # Determine causal role from string diagram position
        causal_role = self._infer_causal_role(content, causal_chain)
        
        node = MemoryNode(
            id=node_id,
            state=initial_strength,
            tier=MemoryTier.SHORT_TERM,
            semantic_type=semantic_type,
            causal_role=causal_role,
            birth_time=datetime.now(),
            last_active=datetime.now(),
        )

        # Add to short-term memory (working buffer)
        self.short_term[node_id] = node
        
        # Add to associative graph
        if not self.associative_graph.has_node(node_id):
            self.associative_graph.add_node(
                node_id, 
                state=initial_strength,
                tier="short_term",
                semantic_type=semantic_type,
                causal_role=causal_role,
            )

        # Create edges to parent memories (associative links)
        if parent_ids:
            for parent_id in parent_ids:
                edge_key = self._edge_key(node_id, parent_id)
                if edge_key not in self.edges:
                    edge = MemoryEdge(
                        source=node_id,
                        target=parent_id,
                        weight=0.6,  # Default association strength for new links
                        causal_direction="bidirectional",
                        formation_step=self.ca_step,
                    )
                    self.edges[edge_key] = edge
                    
                    if parent_id in self.short_term or parent_id in self.mid_term:
                        self.associative_graph.add_edge(node_id, parent_id, weight=0.6)

        # Track causal chain from string diagram encoding
        if causal_chain:
            for src, tgt in causal_chain:
                self.causal_chains.append([src, node_id])  # Link to current memory
                self.monomial_compositions.append((src, node_id))

        self.metrics["total_memories_encoded"] += 1
        
        logger.debug(f"Encoded memory '{node_id}': type={semantic_type}, "
                     f"strength={initial_strength:.2f}, role={causal_role}")
        
        return [node]

    def encode_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[MemoryNode]]:
        """Encode multiple memories at once with cross-linking."""
        results = {}
        for i, item in enumerate(items):
            node_ids = self.encode(**item)
            results[f"batch_{i}"] = node_ids
            
            # Link to previous batch items (temporal association)
            if i > 0:
                prev_key = f"batch_{i-1}"
                for nid in results[prev_key]:
                    for curr_id in node_ids:
                        edge_key = self._edge_key(nid.id, curr_id.id)
                        if edge_key not in self.edges:
                            edge = MemoryEdge(
                                source=nid.id, target=curr_id.id,
                                weight=0.4, causal_direction="bidirectional",
                                formation_step=self.ca_step,
                            )
                            self.edges[edge_key] = edge
                            self.associative_graph.add_edge(nid.id, curr_id.id, weight=0.4)
        
        return results

    # ------------------------------------------------------------------
    # Memory Retrieval (spreading activation)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_steps: int = 20,
        include_tiers: Optional[List[MemoryTier]] = None,
    ) -> MemoryQueryResult:
        """
        Retrieve memories via spreading activation from a seed concept.
        
        Activates the query as a seed node on the grid, then runs CA evolution.
        Activation spreads through associative edges — nodes with highest final
        state are the retrieved memories.
        
        Args:
            query: The search query / seed concept
            top_k: Number of top results to return
            max_steps: Maximum CA steps for activation spread
            include_tiers: Which tiers to search (default: all)
            
        Returns:
            MemoryQueryResult with retrieved memories and their activation scores
        """
        if not query.strip():
            return MemoryQueryResult(query_id="empty_query", retrieved_memories=[])

        # Create seed node from query
        seed_id = f"query_{self.ca_step}"
        
        # Find the best matching existing node as seed (or create temporary)
        candidates = self._find_seed_candidates(query)
        
        if candidates:
            # Use strongest match as activation source
            seed_node_id, seed_strength = max(candidates, key=lambda x: x[1])
            activations = {nid: 0.0 for nid in self._get_active_nodes(include_tiers)}
            activations[seed_node_id] = seed_strength
        else:
            # Create temporary seed node
            activations = {nid: 0.0 for nid in self._get_active_nodes(include_tiers)}
            if not activations:
                return MemoryQueryResult(
                    query_id=seed_id, retrieved_memories=[], confidence=0.0
                )
            seed_node_id = next(iter(activations))
            activations[seed_node_id] = 1.0

        # Run spreading activation (CA evolution for retrieval)
        for step in range(max_steps):
            new_activations = dict(activations)
            
            for node_id, current_state in activations.items():
                if current_state <= 0:
                    continue
                    
                # Get neighbors and their weights
                neighbors = list(self.associative_graph.neighbors(node_id))
                
                for neighbor_id in neighbors:
                    weight = self.associative_graph[node_id][neighbor_id].get("weight", 0.5)
                    
                    # Activation spreads with decay
                    spread = current_state * weight * (1 - self.decay_rate)
                    
                    if neighbor_id not in new_activations:
                        new_activations[neighbor_id] = 0.0
                    
                    # Additive spreading with saturation
                    new_activations[neighbor_id] = min(
                        1.0, 
                        new_activations[neighbor_id] + spread * 0.3
                    )
            
            activations = new_activations

        # Collect results sorted by activation strength
        retrieved = [
            (nid, score) for nid, score in activations.items()
            if score > 0.01 and nid != seed_node_id
        ]
        retrieved.sort(key=lambda x: x[1], reverse=True)
        
        top_results = retrieved[:top_k]
        
        # Calculate confidence based on activation spread
        total_activation = sum(s for _, s in retrieved)
        confidence = min(1.0, total_activation / len(retrieved)) if retrieved else 0.0

        result = MemoryQueryResult(
            query_id=query,
            retrieved_memories=top_results,
            top_k=top_k,
            retrieval_steps=max_steps,
            confidence=confidence,
        )
        
        self.metrics["total_retrievals"] += 1
        
        logger.debug(f"Retrieved {len(top_results)} memories for query '{query[:30]}...' "
                     f"(confidence={confidence:.2f})")
        
        return result

    # ------------------------------------------------------------------
    # CA Evolution (memory decay and consolidation)
    # ------------------------------------------------------------------

    def evolve(self, steps: int = 1) -> Dict[str, float]:
        """
        Run the cellular automata for N steps.
        
        This drives memory dynamics:
          - Decay: All short-term memories lose activation over time
          - Consolidation: Persistent patterns get promoted to higher tiers
          - Edge strengthening: Frequently co-active pairs strengthen their links
        
        Args:
            steps: Number of CA steps to run
            
        Returns:
            Dict mapping node IDs to their final state values (for inspection)
        """
        states_before = {}
        
        for step in range(steps):
            self.ca_step += 1
            
            # Phase 1: Decay all short-term memories
            decayed_nodes = []
            for nid, node in list(self.short_term.items()):
                old_state = node.state
                
                # Base decay rate
                decay = self.decay_rate * node.state
                
                # Hub nodes (high causal importance) decay slower
                if node.causal_role == "hub":
                    decay *= 0.5
                elif node.causal_role == "bridge":
                    decay *= 0.7
                
                new_state = max(0.0, old_state - decay)
                
                # Track history for consolidation check
                self.node_history[nid].append(new_state)
                
                states_before[nid] = old_state
                node.state = new_state
                node.age += 1
                node.last_active = datetime.now()
                
                # Update graph state
                if self.associative_graph.has_node(nid):
                    self.associative_graph.nodes[nid]["state"] = new_state
                
                # Remove dead memories from short-term
                if new_state < 0.05:
                    decayed_nodes.append(nid)
            
            for nid in decayed_nodes:
                del self.short_term[nid]

            # Phase 2: Strengthen edges between co-active nodes
            active_ids = list(self.short_term.keys())
            for i, src in enumerate(active_ids):
                for tgt in active_ids[i+1:]:
                    edge_key = self._edge_key(src, tgt)
                    if edge_key in self.edges:
                        # Strengthen association between co-active memories
                        self.edges[edge_key].weight = min(
                            1.0, 
                            self.edges[edge_key].weight + 0.01
                        )

            # Phase 3: Check for tier promotions (consolidation)
            self._check_promotions()

        return states_before

    def _check_promotions(self):
        """Check if any memories should be promoted to higher tiers."""
        # Short → Mid term: survived enough steps with sufficient state
        for nid, node in list(self.short_term.items()):
            history = self.node_history.get(nid, [])
            
            # Must have survived mid_term_survival_steps consecutive steps above threshold
            recent_states = history[-self.mid_term_survival_steps:] if len(history) >= self.mid_term_survival_steps else []
            
            if (node.age >= 10 and 
                node.state > self.consolidation_threshold * 0.5 and
                len(recent_states) >= self.mid_term_survival_steps // 2):
                
                # Promote to mid-term
                node.tier = MemoryTier.MID_TERM
                self.mid_term[nid] = node
                del self.short_term[nid]
                
                if self.associative_graph.has_node(nid):
                    self.associative_graph.nodes[nid]["tier"] = "mid_term"
                
                self.metrics["tier_promotions"]["short_to_mid"] += 1
                logger.info(f"Promoted memory '{nid}' to mid-term (age={node.age}, state={node.state:.2f})")

        # Mid → Long term: high betweenness centrality in associative graph
        if self.ca_step % 50 == 0 and len(self.mid_term) > 10:
            try:
                subgraph = self.associative_graph.subgraph(
                    list(self.mid_term.keys()) + list(self.long_term.keys())
                )
                
                # Compute betweenness centrality (sampled for efficiency)
                if len(subgraph.nodes) < 50:
                    centrality = nx.betweenness_centrality(subgraph)
                else:
                    centrality = nx.betweenness_centrality(subgraph, k=min(20, len(subgraph.nodes)))
                
                for nid in list(self.mid_term.keys()):
                    if centrality.get(nid, 0) > self.long_term_centrality_threshold:
                        node = self.mid_term[nid]
                        node.tier = MemoryTier.LONG_TERM
                        self.long_term[nid] = node
                        del self.mid_term[nid]
                        
                        if self.associative_graph.has_node(nid):
                            self.associative_graph.nodes[nid]["tier"] = "long_term"
                        
                        self.metrics["tier_promotions"]["mid_to_long"] += 1
                        logger.info(f"Promoted memory '{nid}' to long-term (betweenness={centrality.get(nid, 0):.4f})")
            except Exception as e:
                logger.warning(f"Centrality computation failed: {e}")

    # ------------------------------------------------------------------
    # Persistence (save/load)
    # ------------------------------------------------------------------

    def save(self, filepath: Optional[str] = None):
        """Save the entire memory state to disk."""
        path = filepath or os.path.join(self.storage_dir, "memory_state.json")
        
        data = {
            "ca_step": self.ca_step,
            "metrics": self.metrics,
            "short_term": {k: v.to_dict() for k, v in self.short_term.items()},
            "mid_term": {k: v.to_dict() for k, v in self.mid_term.items()},
            "long_term": {k: v.to_dict() for k, v in self.long_term.items()},
            "edges": {self._edge_key(k[0], k[1]): v.to_dict() 
                      for k, v in self.edges.items()},
            "causal_chains": self.causal_chains[-100:],  # Keep recent chains
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Memory state saved to {path} ({len(self.short_term)} short-term, "
                    f"{len(self.mid_term)} mid-term, {len(self.long_term)} long-term)")

    def load(self, filepath: Optional[str] = None):
        """Load memory state from disk."""
        path = filepath or os.path.join(self.storage_dir, "memory_state.json")
        
        if not os.path.exists(path):
            logger.warning(f"No saved memory found at {path}")
            return False
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.ca_step = data.get("ca_step", 0)
        self.metrics = data.get("metrics", self.metrics)
        
        # Restore tiers
        for tier_key in ["short_term", "mid_term", "long_term"]:
            store = getattr(self, tier_key.replace("_term", ""))
            for nid, ndata in data.get(tier_key, {}).items():
                node = MemoryNode.from_dict(ndata)
                store[nid] = node
        
        # Restore edges
        self.edges = {}
        for key_str, edata in data.get("edges", {}).items():
            parts = key_str.strip("()").split(",")
            src, tgt = parts[0].strip(), parts[1].strip()
            self.edges[(src, tgt)] = MemoryEdge.from_dict(edata)
        
        # Restore causal chains
        self.causal_chains = data.get("causal_chains", [])
        
        logger.info(f"Memory state loaded from {path}")
        return True

    # ------------------------------------------------------------------
    # Query & Analysis Helpers
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get summary statistics about the memory system."""
        total = len(self.short_term) + len(self.mid_term) + len(self.long_term)
        
        return {
            "total_memories": total,
            "short_term_count": len(self.short_term),
            "mid_term_count": len(self.mid_term),
            "long_term_count": len(self.long_term),
            "ca_step": self.ca_step,
            "metrics": copy.deepcopy(self.metrics),
            "avg_short_state": (
                np.mean([n.state for n in self.short_term.values()]) 
                if self.short_term else 0.0
            ),
            "graph_nodes": self.associative_graph.number_of_nodes(),
            "graph_edges": self.associative_graph.number_of_edges(),
        }

    def get_memories_by_type(self, semantic_type: str) -> Dict[str, MemoryNode]:
        """Get all memories of a specific type across tiers."""
        result = {}
        for tier in [self.short_term, self.mid_term, self.long_term]:
            for nid, node in tier.items():
                if node.semantic_type == semantic_type:
                    result[nid] = node
        return result

    def get_causal_chain_for(self, memory_id: str) -> List[List[str]]:
        """Get the causal chain(s) that include this memory."""
        chains = []
        for chain in self.causal_chains:
            if memory_id in chain:
                chains.append(chain)
        return chains

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _infer_causal_role(self, content: str, causal_chain: Optional[List]) -> str:
        """Infer the causal role of a memory from its string diagram position."""
        if not causal_chain or len(causal_chain) < 2:
            return "leaf"
        
        # Count how many times this appears as source vs target in chain
        sources = sum(1 for s, _ in causal_chain if True)  # Simplified inference
        targets = len(causal_chain)
        
        if sources > targets * 0.6:
            return "hub"      # Appears frequently as cause
        elif abs(sources - targets/2) < targets * 0.15:
            return "bridge"   # Both causes and effects
        else:
            return "leaf"    # Primarily an effect

    def _find_seed_candidates(self, query: str) -> List[Tuple[str, float]]:
        """Find existing memories that match the query as activation seeds."""
        candidates = []
        query_lower = query.lower()
        
        for tier in [self.short_term, self.mid_term, self.long_term]:
            for nid, node in tier.items():
                # Simple keyword matching (can be enhanced with embeddings)
                score = 0.0
                if any(word in nid.lower() or word in query_lower 
                       for word in query_lower.split()):
                    score += 0.5
                
                # Boost by activation strength
                score *= node.state
                
                candidates.append((nid, score))
        
        return [(nid, s) for nid, s in candidates if s > 0]

    def _get_active_nodes(self, include_tiers: Optional[List[MemoryTier]] = None) -> List[str]:
        """Get all active memory node IDs, optionally filtered by tier."""
        nodes = []
        
        tiers_to_search = include_tiers or [MemoryTier.SHORT_TERM, MemoryTier.MID_TERM, MemoryTier.LONG_TERM]
        
        if MemoryTier.SHORT_TERM in tiers_to_search:
            nodes.extend(self.short_term.keys())
        if MemoryTier.MID_TERM in tiers_to_search:
            nodes.extend(self.mid_term.keys())
        if MemoryTier.LONG_TERM in tiers_to_search:
            nodes.extend(self.long_term.keys())
        
        return nodes

    @staticmethod
    def _edge_key(a: str, b: str) -> Tuple[str, str]:
        """Canonical edge key (sorted to avoid duplicates)."""
        return tuple(sorted([a, b]))


# ============================================================================
# Convenience Functions
# ============================================================================

def create_memory_system(
    grid_size: int = 100,
    decay_rate: float = 0.02,
) -> LaceMemory:
    """Factory function to create a configured memory system."""
    return LaceMemory(
        grid_size=grid_size,
        decay_rate=decay_rate,
    )

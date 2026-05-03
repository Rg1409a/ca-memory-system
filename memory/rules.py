"""
Custom CA Rules for Memory Dynamics
=====================================

LACE-compatible cellular automata rules that govern memory behavior:

1. MEMORY_DECAY — Organic forgetting (nodes lose state over time)
2. CONSOLIDATION — Persistent patterns resist decay and strengthen edges  
3. SPREADING_ACTIVATION — For retrieval (activation propagates through graph)
4. ASSOCIATIVE_STRENGTHENING — Co-active pairs form stronger links

These rules can be used standalone or integrated with the LACE engine.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Protocol
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# Rule Metadata
# ============================================================================

@dataclass
class MemoryRuleMetadata:
    """Metadata for memory-specific CA rules."""
    name: str
    description: str
    category: str = "memory"
    author: str = "Nouse Hermes"
    version: str = "1.0"
    
    # Rule parameters
    decay_rate: float = 0.02           # Base memory decay per step
    hub_decay_factor: float = 0.5      # Hub nodes decay at this fraction
    bridge_decay_factor: float = 0.7   # Bridge nodes decay at this fraction  
    consolidation_threshold: float = 0.7
    activation_spread_rate: float = 0.3
    association_strength_gain: float = 0.01
    
    # Tier promotion thresholds
    mid_term_survival_steps: int = 30
    long_term_centrality_threshold: float = 0.05


# ============================================================================
# Rule Implementations
# ============================================================================

class MemoryRuleType(Enum):
    """Types of memory dynamics rules."""
    DECAY = "memory_decay"
    CONSOLIDATION = "consolidation"  
    SPREADING_ACTIVATION = "spreading_activation"
    ASSOCIATIVE_STRENGTHENING = "associative_strengthening"


class MemoryRule:
    """Base class for memory dynamics rules."""
    
    def __init__(self, metadata: MemoryRuleMetadata):
        self.metadata = metadata
        self.name = metadata.name
        
    def apply(
        self, 
        node_states: Dict[str, float],
        edges: Dict[Tuple[str, str], float],
        node_roles: Dict[str, str],  # "hub", "bridge", "leaf"
        step: int,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """Apply the rule and return updated states and edges."""
        raise NotImplementedError


class MemoryDecayRule(MemoryRule):
    """
    Organic forgetting rule.
    
    All memories decay over time at a rate proportional to their current state.
    Hub nodes (high causal importance) and bridge nodes decay more slowly,
    reflecting how important concepts are harder to forget in human memory.
    
    Parameters:
        decay_rate: Base decay per step (default 0.02 = 2% per step)
        hub_decay_factor: Multiplier for hub node decay (default 0.5)
        bridge_decay_factor: Multiplier for bridge node decay (default 0.7)
    """
    
    def apply(
        self,
        node_states: Dict[str, float],
        edges: Dict[Tuple[str, str], float],
        node_roles: Dict[str, str],
        step: int,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        new_states = dict(node_states)
        
        for nid, state in node_states.items():
            if state <= 0.01:
                continue
            
            # Base decay proportional to current state (exponential decay)
            decay = self.metadata.decay_rate * state
            
            # Adjust by causal role
            role = node_roles.get(nid, "leaf")
            if role == "hub":
                decay *= self.metadata.hub_decay_factor
            elif role == "bridge":
                decay *= self.metadata.bridge_decay_factor
            
            new_states[nid] = max(0.0, state - decay)
        
        return new_states, edges


class ConsolidationRule(MemoryRule):
    """
    Memory consolidation rule.
    
    Memories that persist above a threshold for consecutive steps get:
    1. Reduced decay rate (they become more stable)
    2. Strengthened associative links to co-active neighbors
    
    This models how repeated activation leads to long-term memory formation.
    """
    
    def __init__(self, metadata: MemoryRuleMetadata):
        super().__init__(metadata)
        self.persistence_tracker: Dict[str, List[float]] = {}  # nid -> [recent states]
        
    def apply(
        self,
        node_states: Dict[str, float],
        edges: Dict[Tuple[str, str], float],
        node_roles: Dict[str, str],
        step: int,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        new_states = dict(node_states)
        new_edges = dict(edges)
        
        for nid, state in node_states.items():
            # Track persistence history
            if nid not in self.persistence_tracker:
                self.persistence_tracker[nid] = []
            self.persistence_tracker[nid].append(state)
            
            # Keep only recent history
            history = self.persistence_tracker[nid][-self.metadata.mid_term_survival_steps:]
            
            # Check if memory has persisted above threshold
            if len(history) >= 10 and state > self.metadata.consolidation_threshold * 0.5:
                persistent_ratio = sum(1 for s in history if s > self.metadata.consolidation_threshold * 0.3) / len(history)
                
                if persistent_ratio > 0.6:
                    # Strengthen this memory's resistance to future decay
                    new_states[nid] = min(1.0, state + 0.02)  # Small boost
                    
                    # Strengthen edges to co-active neighbors
                    for (src, tgt), weight in list(edges.items()):
                        if src == nid or tgt == nid:
                            other_nid = tgt if src == nid else src
                            if node_states.get(other_nid, 0) > self.metadata.consolidation_threshold * 0.5:
                                new_edges[(src, tgt)] = min(1.0, weight + self.metadata.association_strength_gain)
        
        return new_states, new_edges


class SpreadingActivationRule(MemoryRule):
    """
    Spreading activation rule for memory retrieval.
    
    Activates a seed node and propagates activation through the associative graph.
    Activation spreads proportionally to edge weights, with decay at each hop.
    
    This models how recalling one concept triggers related memories in human cognition.
    """
    
    def __init__(self, metadata: MemoryRuleMetadata):
        super().__init__(metadata)
        
    def apply(
        self,
        node_states: Dict[str, float],
        edges: Dict[Tuple[str, str], float],
        node_roles: Dict[str, str],
        step: int,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        new_states = dict(node_states)
        
        # Calculate total activation for normalization
        total_activation = sum(s for s in node_states.values())
        
        if total_activation == 0:
            return node_states, edges
        
        for nid, state in list(node_states.items()):
            if state <= 0.01:
                continue
            
            # Get neighbors and their edge weights
            neighbor_activations = []
            for (src, tgt), weight in edges.items():
                if src == nid:
                    neighbor_activations.append((tgt, weight))
                elif tgt == nid:
                    neighbor_activations.append((src, weight))
            
            # Spread activation to neighbors
            for neighbor_id, weight in neighbor_activations:
                if neighbor_id not in new_states:
                    continue
                    
                # Activation spreads with decay and weight scaling
                spread = state * weight * self.metadata.activation_spread_rate
                
                # Additive spreading with saturation
                new_states[neighbor_id] = min(1.0, new_states[neighbor_id] + spread)
        
        return new_states, edges


class AssociativeStrengtheningRule(MemoryRule):
    """
    Association strengthening rule.
    
    Co-active memories (both above threshold) strengthen their associative link.
    This models Hebbian learning: "neurons that fire together, wire together."
    
    The strength of the association depends on:
    - How frequently the pair co-activates
    - The causal relationship between them
    """
    
    def apply(
        self,
        node_states: Dict[str, float],
        edges: Dict[Tuple[str, str], float],
        node_roles: Dict[str, str],
        step: int,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        new_edges = dict(edges)
        
        threshold = self.metadata.consolidation_threshold * 0.5
        
        for (src, tgt), weight in list(edges.items()):
            src_state = node_states.get(src, 0)
            tgt_state = node_states.get(tgt, 0)
            
            # Strengthen if both are active
            if src_state > threshold and tgt_state > threshold:
                new_edges[(src, tgt)] = min(1.0, weight + self.metadata.association_strength_gain)
        
        return node_states, new_edges


# ============================================================================
# Rule Factory
# ============================================================================

def create_memory_rules(metadata: Optional[MemoryRuleMetadata] = None) -> Dict[str, MemoryRule]:
    """Create all memory dynamics rules with given metadata."""
    if metadata is None:
        metadata = MemoryRuleMetadata(
            name="memory_dynamics_suite",
            description="Complete suite of memory CA rules for Nouse Hermes",
        )
    
    return {
        "decay": MemoryDecayRule(metadata),
        "consolidation": ConsolidationRule(metadata),
        "spreading_activation": SpreadingActivationRule(metadata),
        "associative_strengthening": AssociativeStrengtheningRule(metadata),
    }


def apply_rules_sequentially(
    node_states: Dict[str, float],
    edges: Dict[Tuple[str, str], float],
    node_roles: Dict[str, str],
    rules: Optional[Dict[str, MemoryRule]] = None,
    step: int = 0,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """
    Apply memory rules in sequence (decay → consolidation → strengthening).
    
    This is the standard evolution order for memory dynamics.
    """
    if rules is None:
        rules = create_memory_rules()
    
    states = node_states
    edge_map = edges
    
    # 1. Decay first (forgetting)
    states, _ = rules["decay"].apply(states, edge_map, node_roles, step)
    
    # 2. Consolidation (strengthen persistent patterns)
    states, edge_map = rules["consolidation"].apply(states, edge_map, node_roles, step)
    
    # 3. Associative strengthening (Hebbian learning)
    _, edge_map = rules["associative_strengthening"].apply(states, edge_map, node_roles, step)
    
    return states, edge_map


def apply_spreading_activation(
    seed_states: Dict[str, float],
    edges: Dict[Tuple[str, str], float],
    max_steps: int = 20,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """
    Run spreading activation from seed nodes for retrieval.
    
    Args:
        seed_states: Initial activations (only non-zero values are seeds)
        edges: Associative graph edges with weights
        max_steps: Maximum propagation steps
        
    Returns:
        Final activated states and updated edge weights
    """
    rule = SpreadingActivationRule(MemoryRuleMetadata(
        name="spreading_activation", description="Retrieval via activation spread"
    ))
    
    states = seed_states
    
    for step in range(max_steps):
        new_states, _ = rule.apply(states, edges, {}, step)
        
        # Check convergence (no significant change)
        max_change = max(abs(new_states.get(nid, 0) - states.get(nid, 0)) 
                        for nid in set(list(states.keys()) + list(new_states.keys())))
        
        if max_change < 0.001:
            break
        
        states = new_states
    
    return states, edges

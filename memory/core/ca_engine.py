"""
Core Cellular Automata Engine (Decoupled from LACE)
=====================================================

A standalone, lightweight CA rule engine for memory dynamics.
This module provides the core simulation loop without depending on
the full LACE framework — making it suitable as a general-purpose
memory substrate that can be used independently or as a backend
for larger systems.

Design Principles:
  - Minimal dependencies (numpy only)
  - Pluggable rule system for custom dynamics
  - Sparse representation for efficiency with large grids
  - Configurable topology (Moore, von Neumann, custom neighborhoods)
  - Deterministic evolution with optional stochastic rules

Usage as standalone memory substrate:
  from ..core.ca_engine import CAEngine
  
  engine = CAEngine(grid_size=(100, 100), neighborhood='moore')
  
  # Add memories as active nodes
  engine.set_node_state('mem_1', (10, 20), state=0.8)
  engine.set_node_state('mem_2', (15, 25), state=0.6)
  
  # Create associative links
  engine.add_edge(('mem_1', 'mem_2'), weight=0.7)
  
  # Register memory dynamics rules
  from ..core.rules import MemoryDecayRule
  engine.register_rule('decay', MemoryDecayRule(decay_rate=0.02))
  
  # Evolve (forget/consolidate)
  states, edges = engine.evolve(steps=10)
  
  # Retrieve via spreading activation
  results = engine.spread_activation(seed_nodes={'mem_1': 1.0}, max_steps=20)

Architecture:
  CAEngine manages the simulation state and rule application loop.
  Rules are pluggable modules that transform node states and edges.
  The engine handles sparse storage, neighborhood computation, and
  boundary conditions — users only need to define rules for their
  specific memory dynamics.
"""

from __future__ import annotations

import math
import logging
import copy
from dataclasses import dataclass, field
from typing import (
    Dict, List, Set, Tuple, Optional, Any, Protocol, Callable, Iterable
)
from collections import defaultdict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class NodeState:
    """Represents a single active node in the CA grid."""
    id: str                              # Unique identifier
    position: Tuple[int, int]            # (row, col) on grid
    state: float                         # Activation value 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra data
    
    @property
    def is_active(self) -> bool:
        return self.state > 1e-6

    def copy(self) -> 'NodeState':
        return NodeState(
            id=self.id,
            position=tuple(self.position),
            state=self.state,
            metadata=dict(self.metadata),
        )


@dataclass
class Edge:
    """Associative link between two nodes."""
    source: str                          # Node ID
    target: str                          # Node ID
    weight: float = 1.0                  # Association strength 0.0-1.0
    
    @property
    def canonical_key(self) -> Tuple[str, str]:
        return tuple(sorted([self.source, self.target]))


# ============================================================================
# Rule Interface (Pluggable Architecture)
# ============================================================================

class CARule(Protocol):
    """Interface for pluggable CA rules."""
    
    def apply(
        self,
        nodes: Dict[str, NodeState],
        edges: Dict[Tuple[str, str], Edge],
        step: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """
        Apply the rule and return updated states/edges.
        
        Args:
            nodes: Current node states {id: NodeState}
            edges: Current edge weights {(src, tgt): weight}
            step: Current evolution step number
            context: Optional shared context between rules
            
        Returns:
            (updated_states, updated_edges) where dicts map IDs/keys to new values.
            Only entries that changed need to be included; unchanged entries are
            preserved from the input.
        """
        ...


# ============================================================================
# Built-in Memory Rules
# ============================================================================

class MemoryDecayRule(CARule):
    """
    Organic forgetting: all nodes decay proportionally to their state.
    
    Hub nodes (identified by high degree) decay slower, modeling how
    important concepts are harder to forget.
    """
    
    def __init__(self, decay_rate: float = 0.02, hub_factor: float = 0.5):
        self.decay_rate = decay_rate
        self.hub_factor = hub_factor
    
    def apply(
        self,
        nodes: Dict[str, NodeState],
        edges: Dict[Tuple[str, str], Edge],
        step: int,
        context: Optional[Dict] = None,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        new_states = {}
        
        # Compute degrees for hub detection
        degrees = defaultdict(int)
        for (src, tgt), edge in edges.items():
            degrees[src] += 1
            degrees[tgt] += 1
        
        max_degree = max(degrees.values()) if degrees else 1
        
        for nid, node in nodes.items():
            if not node.is_active:
                continue
            
            # Hub detection: degree > 50% of max gets hub treatment
            is_hub = degrees.get(nid, 0) > max_degree * 0.5 if max_degree > 0 else False
            
            decay = self.decay_rate * node.state
            if is_hub:
                decay *= self.hub_factor
            
            new_states[nid] = max(0.0, node.state - decay)
        
        return new_states, {}


class ConsolidationRule(CARule):
    """
    Memory consolidation: persistent patterns resist decay and strengthen edges.
    
    Memories that stay above threshold for consecutive steps get a small boost
    and their associative links to co-active neighbors are strengthened.
    """
    
    def __init__(self, threshold: float = 0.7, persistence_window: int = 10):
        self.threshold = threshold
        self.persistence_window = persistence_window
        # Track recent states per node for persistence detection
        self._history: Dict[str, List[float]] = defaultdict(list)
    
    def apply(
        self,
        nodes: Dict[str, NodeState],
        edges: Dict[Tuple[str, str], Edge],
        step: int,
        context: Optional[Dict] = None,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        new_states = {}
        new_edges = {}
        
        for nid, node in nodes.items():
            # Track history
            self._history[nid].append(node.state)
            if len(self._history[nid]) > self.persistence_window * 2:
                self._history[nid] = self._history[nid][-self.persistence_window:]
            
            recent = self._history[nid]
            persistent_ratio = (
                sum(1 for s in recent if s > self.threshold * 0.5) / len(recent)
                if recent else 0
            )
            
            # If consistently above threshold, boost state slightly
            if persistent_ratio > 0.6 and node.state < 1.0:
                new_states[nid] = min(1.0, node.state + 0.02)
            elif node.is_active:
                new_states[nid] = node.state
        
        # Strengthen edges between co-active persistent nodes
        for (src, tgt), edge in edges.items():
            src_state = nodes.get(src, NodeState("", (0, 0), 0)).state
            tgt_state = nodes.get(tgt, NodeState("", (0, 0), 0)).state
            
            if (src_state > self.threshold * 0.5 and 
                tgt_state > self.threshold * 0.5):
                new_edges[(src, tgt)] = min(1.0, edge.weight + 0.01)
            else:
                new_edges[(src, tgt)] = edge.weight
        
        return new_states, new_edges


class SpreadingActivationRule(CARule):
    """
    Spreading activation for retrieval: propagates seed activations through graph.
    
    Activation spreads proportionally to edge weights with decay per hop.
    Used during query processing rather than normal evolution.
    """
    
    def __init__(self, spread_rate: float = 0.3, decay_per_hop: float = 0.9):
        self.spread_rate = spread_rate
        self.decay_per_hop = decay_per_hop
    
    def apply(
        self,
        nodes: Dict[str, NodeState],
        edges: Dict[Tuple[str, str], Edge],
        step: int,
        context: Optional[Dict] = None,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        # Get seed activations from context
        seeds = context.get('seeds', {}) if context else {}
        
        new_states = {nid: node.state for nid, node in nodes.items()}
        
        for nid, current_state in list(new_states.items()):
            if current_state < 1e-6 or nid not in seeds and current_state == 0:
                continue
            
            # Spread to neighbors
            for (src, tgt), edge in edges.items():
                neighbor_id = None
                if src == nid:
                    neighbor_id = tgt
                elif tgt == nid:
                    neighbor_id = src
                
                if neighbor_id is None or neighbor_id not in new_states:
                    continue
                
                spread = current_state * edge.weight * self.spread_rate * self.decay_per_hop
                new_states[neighbor_id] = min(1.0, new_states[neighbor_id] + spread)
        
        return new_states, {}


class AssociativeStrengtheningRule(CARule):
    """
    Hebbian learning: co-active pairs strengthen their associative links.
    
    "Neurons that fire together, wire together." Memories that are both active
    in the same time step get stronger connections between them.
    """
    
    def __init__(self, threshold: float = 0.3, gain: float = 0.01):
        self.threshold = threshold
        self.gain = gain
    
    def apply(
        self,
        nodes: Dict[str, NodeState],
        edges: Dict[Tuple[str, str], Edge],
        step: int,
        context: Optional[Dict] = None,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        new_edges = {}
        
        for (src, tgt), edge in edges.items():
            src_state = nodes.get(src, NodeState("", (0, 0), 0)).state
            tgt_state = nodes.get(tgt, NodeState("", (0, 0), 0)).state
            
            if src_state > self.threshold and tgt_state > self.threshold:
                new_edges[(src, tgt)] = min(1.0, edge.weight + self.gain)
            else:
                new_edges[(src, tgt)] = edge.weight
        
        return {}, new_edges


# ============================================================================
# Neighborhood Definitions
# ============================================================================

class NeighborhoodType(Enum):
    MOORE = "moore"           # 8 neighbors (including diagonals)
    VON_NEUMANN = "von_neumann"  # 4 neighbors (cardinal only)
    CUSTOM = "custom"         # User-defined offsets


def get_moore_offsets() -> List[Tuple[int, int]]:
    return [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

def get_von_neumann_offsets() -> List[Tuple[int, int]]:
    return [(-1,0), (1,0), (0,-1), (0,1)]


# ============================================================================
# Boundary Conditions
# ============================================================================

class BoundaryCondition(Enum):
    BOUNDED = "bounded"       # Dead cells at boundary
    WRAP = "wrap"             # Toroidal wrapping


def apply_boundary(pos: Tuple[int, int], grid_size: Tuple[int, int], 
                   condition: BoundaryCondition) -> Tuple[int, int]:
    """Apply boundary condition to a position."""
    r, c = pos
    rows, cols = grid_size
    
    if condition == BoundaryCondition.WRAP:
        return (r % rows, c % cols)
    else:  # BOUNDED
        return (max(0, min(rows - 1, r)), max(0, min(cols - 1, c)))


# ============================================================================
# Core CA Engine
# ============================================================================

class CAEngine:
    """
    Standalone cellular automata engine for memory dynamics.
    
    This is the core simulation engine — decoupled from LACE but compatible
    with its rule semantics. It manages sparse node storage, neighborhood
    computation, boundary conditions, and pluggable rule application.
    
    The engine treats memories as active nodes on a 2D grid. Edges represent
    associative links between co-active memories. Rules transform states each
    evolution step to model forgetting, consolidation, and spreading activation.
    
    Args:
        grid_size: (rows, cols) of the simulation grid
        neighborhood: Type of neighborhood ('moore', 'von_neumann', or list of offsets)
        boundary: Boundary condition ('bounded' or 'wrap')
        rules: Initial dict of named rules to register
    
    Example:
        engine = CAEngine(grid_size=(100, 100), neighborhood='moore')
        
        # Register memory dynamics rules
        from ..core.rules import MemoryDecayRule
        engine.register_rule('decay', MemoryDecayRule(decay_rate=0.02))
        engine.register_rule('consolidation', ConsolidationRule(threshold=0.7))
        
        # Add memories
        engine.set_node_state('mem_1', (10, 20), state=0.8)
        engine.add_edge(('mem_1', 'mem_2'), weight=0.6)
        
        # Evolve
        states, edges = engine.evolve(steps=10)
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (100, 100),
        neighborhood: str = 'moore',
        boundary: str = 'bounded',
        rules: Optional[Dict[str, CARule]] = None,
    ):
        self.grid_size = grid_size
        
        # Neighborhood configuration
        if neighborhood == 'moore':
            self._neighborhood_offsets = get_moore_offsets()
        elif neighborhood == 'von_neumann':
            self._neighborhood_offsets = get_von_neumann_offsets()
        else:
            self._neighborhood_offsets = neighborhood
        
        # Boundary condition
        self.boundary = BoundaryCondition(boundary) if isinstance(boundary, str) else boundary
        
        # Sparse node storage — only track active nodes
        self.nodes: Dict[str, NodeState] = {}
        
        # Edge storage — sparse adjacency list
        self.edges: Dict[Tuple[str, str], Edge] = {}
        
        # Pluggable rules
        self._rules: Dict[str, CARule] = {}
        if rules:
            for name, rule in rules.items():
                self.register_rule(name, rule)
        
        # Evolution state
        self.step_count: int = 0
        
        # Persistence tracking (for consolidation rule)
        self._node_history: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"CAEngine initialized: grid={grid_size}, "
                    f"neighborhood={neighborhood}, boundary={boundary}")

    # ------------------------------------------------------------------
    # Node Management
    # ------------------------------------------------------------------

    def set_node_state(self, node_id: str, position: Tuple[int, int], 
                       state: float = 0.5, metadata: Optional[Dict] = None):
        """Set or create a node with given activation state."""
        if not (0 <= position[0] < self.grid_size[0]):
            raise ValueError(f"Row {position[0]} out of bounds for grid size {self.grid_size}")
        if not (0 <= position[1] < self.grid_size[1]):
            raise ValueError(f"Col {position[1]} out of bounds for grid size {self.grid_size}")
        
        node = NodeState(
            id=node_id,
            position=position,
            state=max(0.0, min(1.0, state)),
            metadata=dict(metadata) if metadata else {},
        )
        
        self.nodes[node_id] = node
        
        # Update edges to reflect new connections (nodes within neighborhood distance)
        self._update_adjacency(node)

    def remove_node(self, node_id: str):
        """Remove a node and all its edges."""
        if node_id in self.nodes:
            del self.nodes[node_id]
        
        # Remove associated edges
        keys_to_remove = [k for k in self.edges 
                         if k[0] == node_id or k[1] == node_id]
        for key in keys_to_remove:
            del self.edges[key]

    def get_node(self, node_id: str) -> Optional[NodeState]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    # ------------------------------------------------------------------
    # Edge Management
    # ------------------------------------------------------------------

    def add_edge(self, pair: Tuple[str, str], weight: float = 1.0):
        """Add or update an edge between two nodes."""
        src, tgt = pair
        if src not in self.nodes or tgt not in self.nodes:
            logger.warning(f"Cannot add edge ({src}, {tgt}): one or both nodes missing")
            return
        
        key = (min(src, tgt), max(src, tgt))  # Canonical order
        existing = self.edges.get(key)
        
        if existing:
            existing.weight = min(1.0, max(0.0, weight))
        else:
            self.edges[key] = Edge(source=src, target=tgt, weight=min(1.0, max(0.0, weight)))

    def remove_edge(self, pair: Tuple[str, str]):
        """Remove an edge between two nodes."""
        key = (min(pair[0], pair[1]), max(pair[0], pair[1]))
        self.edges.pop(key, None)

    # ------------------------------------------------------------------
    # Rule Management
    # ------------------------------------------------------------------

    def register_rule(self, name: str, rule: CARule):
        """Register a pluggable CA rule."""
        self._rules[name] = rule
        logger.debug(f"Registered rule '{name}': {type(rule).__name__}")

    def unregister_rule(self, name: str):
        """Remove a registered rule."""
        self._rules.pop(name, None)

    def get_rules(self) -> Dict[str, CARule]:
        """Get all registered rules."""
        return dict(self._rules)

    # ------------------------------------------------------------------
    # Evolution Loop
    # ------------------------------------------------------------------

    def evolve(
        self, 
        steps: int = 1,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
        """
        Run the CA for N evolution steps.
        
        Each step applies all registered rules in sequence to transform
        node states and edge weights. This drives memory dynamics:
        forgetting (decay), consolidation (strengthening persistent patterns),
        and associative learning (Hebbian strengthening).
        
        Args:
            steps: Number of evolution steps to run
            context: Optional shared context passed to rules
            
        Returns:
            (final_states, final_edges) — dicts mapping IDs/keys to values
        """
        all_states = {}
        all_edges = {}
        
        for step in range(steps):
            self.step_count += 1
            
            # Apply each rule sequentially
            current_nodes = {nid: node.copy() for nid, node in self.nodes.items()}
            current_edges = dict(self.edges)
            
            for rule_name, rule in self._rules.items():
                try:
                    new_states, new_edge_weights = rule.apply(
                        nodes=current_nodes,
                        edges=current_edges,
                        step=self.step_count,
                        context=context,
                    )
                    
                    # Merge updates
                    if new_states:
                        for nid, state in new_states.items():
                            current_nodes[nid].state = state
                    
                    if new_edge_weights:
                        for key, weight in new_edge_weights.items():
                            if key in current_edges:
                                current_edges[key].weight = weight
                
                except Exception as e:
                    logger.error(f"Rule '{rule_name}' failed at step {self.step_count}: {e}")
            
            # Clean up dead nodes (state ≈ 0)
            dead_nodes = [nid for nid, node in current_nodes.items() 
                         if not node.is_active]
            for nid in dead_nodes:
                del current_nodes[nid]
            
            all_states.update({nid: node.state for nid, node in current_nodes.items()})
            all_edges.update({k: e.weight for k, e in current_edges.items()})
        
        # Update internal state with evolved nodes (NOT original self.nodes!)
        self.nodes = {nid: n for nid, n in current_nodes.items()}
        
        return all_states, all_edges

    def spread_activation(
        self,
        seed_nodes: Dict[str, float],
        max_steps: int = 20,
    ) -> Dict[str, float]:
        """
        Run spreading activation from seed nodes for retrieval.
        
        Activates specified seed nodes and propagates through the graph.
        Nodes with highest final activation are the retrieved memories.
        
        Args:
            seed_nodes: {node_id: initial_activation} — seeds to activate
            max_steps: Maximum propagation steps
            
        Returns:
            Final activation states for all reachable nodes
        """
        activations = {nid: 0.0 for nid in self.nodes}
        
        # Initialize seeds
        for nid, state in seed_nodes.items():
            if nid in activations:
                activations[nid] = max(activations[nid], state)
        
        rule = SpreadingActivationRule(spread_rate=0.3, decay_per_hop=0.9)
        
        for step in range(max_steps):
            new_activations = dict(activations)
            
            for nid, current_state in activations.items():
                if current_state < 1e-6:
                    continue
                
                # Spread to neighbors
                for (src, tgt), edge in self.edges.items():
                    neighbor_id = None
                    if src == nid:
                        neighbor_id = tgt
                    elif tgt == nid:
                        neighbor_id = src
                    
                    if neighbor_id is None or neighbor_id not in new_activations:
                        continue
                    
                    spread = current_state * edge.weight * 0.3 * 0.9
                    new_activations[neighbor_id] = min(1.0, new_activations[neighbor_id] + spread)
            
            activations = new_activations
            
            # Check convergence
            max_change = max(abs(activations[nid] - new_activations.get(nid, 0))
                           for nid in activations if nid in new_activations)
            if max_change < 1e-4:
                break
        
        return activations

    # ------------------------------------------------------------------
    # Adjacency & Neighborhood Helpers
    # ------------------------------------------------------------------

    def _update_adjacency(self, node: NodeState):
        """Update edges for a newly added/modified node based on grid proximity."""
        r, c = node.position
        
        for dr, dc in self._neighborhood_offsets:
            nr, nc = apply_boundary((r + dr, c + dc), self.grid_size, self.boundary)
            
            # Find any active node at this position
            for other_id, other_node in self.nodes.items():
                if other_id == node.id:
                    continue
                if (other_node.position[0] == nr and 
                    other_node.position[1] == nc):
                    
                    key = (min(node.id, other_id), max(node.id, other_id))
                    if key not in self.edges:
                        self.edges[key] = Edge(
                            source=node.id,
                            target=other_id,
                            weight=0.5,  # Default initial association strength
                        )

    def get_neighbors(self, node_id: str) -> List[Tuple[str, float]]:
        """Get all neighbors of a node with their edge weights."""
        neighbors = []
        for (src, tgt), edge in self.edges.items():
            if src == node_id:
                neighbors.append((tgt, edge.weight))
            elif tgt == node_id:
                neighbors.append((src, edge.weight))
        return neighbors

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current CA state."""
        active_states = [n.state for n in self.nodes.values()]
        
        return {
            "step": self.step_count,
            "active_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "avg_activation": float(np.mean(active_states)) if active_states else 0.0,
            "max_activation": float(max(active_states)) if active_states else 0.0,
            "min_activation": float(min(active_states)) if active_states else 0.0,
        }

    def save_state(self) -> Dict[str, Any]:
        """Serialize the full CA state for persistence."""
        return {
            "grid_size": self.grid_size,
            "step_count": self.step_count,
            "nodes": {nid: node.to_dict() if hasattr(node, 'to_dict') else 
                     {"id": nid, "position": node.position, "state": node.state}
                     for nid, node in self.nodes.items()},
            "edges": {f"{k[0]},{k[1]}": e.weight for k, e in self.edges.items()},
        }

    def load_state(self, state: Dict[str, Any]):
        """Restore CA state from serialized data."""
        self.grid_size = tuple(state["grid_size"])
        self.step_count = state.get("step_count", 0)
        
        # Restore nodes
        self.nodes = {}
        for nid, ndata in state.get("nodes", {}).items():
            if isinstance(ndata, dict):
                pos = tuple(ndata.get("position", (0, 0)))
                st = ndata.get("state", 0.5)
            else:
                pos = (0, 0)
                st = float(ndata)
            
            self.nodes[nid] = NodeState(id=nid, position=pos, state=st)
        
        # Restore edges
        self.edges = {}
        for key_str, weight in state.get("edges", {}).items():
            parts = key_str.split(",")
            if len(parts) == 2:
                src, tgt = parts[0], parts[1]
                self.edges[(src, tgt)] = Edge(source=src, target=tgt, weight=float(weight))

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.edges.clear()
        self.step_count = 0
        self._node_history.clear()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_memory_engine(
    grid_size: Tuple[int, int] = (100, 100),
    decay_rate: float = 0.02,
) -> CAEngine:
    """Factory function to create a configured memory engine."""
    engine = CAEngine(grid_size=grid_size, neighborhood='moore')
    
    # Register default memory dynamics rules
    from ..core.rules import MemoryDecayRule, ConsolidationRule
    
    engine.register_rule('decay', MemoryDecayRule(decay_rate=decay_rate))
    engine.register_rule('consolidation', ConsolidationRule(threshold=0.7))
    
    return engine

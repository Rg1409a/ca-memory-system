"""
Causal String Diagram → CA State Encoder
==========================================

Converts causal extraction results (from PDF analysis) into cellular automata
grid states using string diagram mathematics.

String diagrams provide a rigorous mathematical framework for representing
causal relationships as compositional structures:

  Objects (wires):    Entities, variables, or states in the causal chain
  Morphisms (boxes):  Interventions, transformations, or causal mechanisms  
  Composition (∘):    Sequential causality — A causes B causes C
  Tensor product (⊗): Parallel/causal fusion — A and B jointly cause D

This module maps these concepts to CA grid positions:
  - Wire junctions → node coordinates on the grid
  - Causal edges → associative links between active nodes  
  - Monoidal structure → spatial clustering of related concepts
  - Diagram depth → temporal ordering across CA evolution steps

Usage:
  encoder = CausalStringDiagramEncoder()
  
  # From extracted causal triplets (Subject, Relation, Object)
  causal_graph = {
      "nodes": [("X", {"type": "variable"}), ("Y", {"type": "intervention"})],
      "edges": [("X", "Y", {"weight": 0.8})]
  }
  
  ca_state = encoder.encode(causal_graph)
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
# String Diagram Data Model
# ============================================================================

class DiagramElement(Enum):
    """Types of elements in a string diagram."""
    WIRE = "wire"              # Object (entity/variable)
    BOX = "box"                # Morphism (intervention/transformation)
    INPUT = "input"            # Wire input to a box
    OUTPUT = "output"          # Wire output from a box
    JUNCTION = "junction"      # Where wires meet (causal fusion point)


@dataclass
class StringDiagramNode:
    """A node in the string diagram representation."""
    id: str                              # Unique identifier
    element_type: DiagramElement         # wire, box, input, output, junction
    causal_role: str = "leaf"            # root, hub, bridge, leaf
    semantic_type: str = "observation"   # observation, concept, fact, event, rule
    state: float = 0.5                   # Initial activation strength (0-1)
    
    # String diagram position metadata
    depth: int = 0                       # Vertical position in diagram (temporal order)
    width_pos: int = 0                   # Horizontal position (monoidal composition)
    
    # Causal relationships
    parents: List[str] = field(default_factory=list)   # Nodes that cause this one
    children: List[str] = field(default_factory=list)  # Nodes this causes
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "element_type": self.element_type.value,
            "causal_role": self.causal_role,
            "semantic_type": self.semantic_type,
            "state": self.state,
            "depth": self.depth,
            "width_pos": self.width_pos,
            "parents": self.parents,
            "children": self.children,
        }


@dataclass 
class StringDiagramEdge:
    """A causal link between two string diagram nodes."""
    source: str                            # Node ID
    target: str                            # Node ID  
    weight: float = 1.0                    # Causal strength (0-1)
    composition_type: str = "sequential"   # sequential, parallel, fused
    
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "composition_type": self.composition_type,
        }


@dataclass
class StringDiagram:
    """A complete string diagram representing causal relationships."""
    nodes: Dict[str, StringDiagramNode] = field(default_factory=dict)
    edges: List[StringDiagramEdge] = field(default_factory=list)
    
    def add_node(self, node: StringDiagramNode):
        self.nodes[node.id] = node
        
    def add_edge(self, edge: StringDiagramEdge):
        self.edges.append(edge)
        
    def get_children(self, nid: str) -> List[str]:
        return [e.target for e in self.edges if e.source == nid]
    
    def get_parents(self, nid: str) -> List[str]:
        return [e.source for e in self.edges if e.target == nid]


# ============================================================================
# Encoder Implementation
# ============================================================================

class CausalStringDiagramEncoder:
    """
    Converts causal extraction results into CA grid states using string diagram math.
    
    The encoding process follows these steps:
    
    1. Parse causal triplets (Subject, Relation, Object) into a string diagram
    2. Compute monoidal composition structure (⊗ for parallel, ∘ for sequential)
    3. Map diagram nodes to grid coordinates based on categorical position
    4. Generate CA initial conditions from the mapped structure
    5. Return encoding ready for LaceMemory.encode()
    
    Key mapping principles:
      - Sequential composition (∘) → vertical stacking in grid columns
      - Parallel composition (⊗) → horizontal adjacency in grid rows  
      - Junction points (causal fusion) → high-activation nodes
      - Hub morphisms (many inputs/outputs) → long-term memory anchors
    """

    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        
    # ------------------------------------------------------------------
    # Main Encoding Pipeline
    # ------------------------------------------------------------------

    def encode(
        self,
        causal_graph: Dict[str, Any],
        semantic_context: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        """
        Encode a causal graph into CA-compatible memory items.
        
        Args:
            causal_graph: Dict with 'nodes' and 'edges' from causal extraction
                nodes: [{"id": str, "type": str, ...}]
                edges: [{"source": str, "target": str, "weight": float}]
            semantic_context: Optional mapping of node types to semantic categories
            
        Returns:
            (memory_items, edge_pairs) where:
              memory_items: List of dicts ready for LaceMemory.encode()
              edge_pairs: List of (source_id, target_id) for causal chain tracking
        """
        # Step 1: Build string diagram from causal graph
        diagram = self._build_string_diagram(causal_graph)
        
        # Step 2: Compute monoidal composition structure
        self._compute_composition_structure(diagram)
        
        # Step 3: Map to grid coordinates
        grid_mapping = self._map_to_grid(diagram)
        
        # Step 4: Generate memory items from mapped nodes
        memory_items, edge_pairs = self._generate_memory_items(
            diagram, grid_mapping, semantic_context or {}
        )
        
        logger.info(f"Encoded causal graph: {len(memory_items)} memories, "
                    f"{len(edge_pairs)} causal links")
        
        return memory_items, edge_pairs

    def encode_batch(
        self,
        causal_graphs: List[Dict[str, Any]],
        semantic_contexts: Optional[List[Dict[str, str]]] = None,
    ) -> List[Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]]:
        """Encode multiple causal graphs."""
        results = []
        for i, graph in enumerate(causal_graphs):
            ctx = semantic_contexts[i] if semantic_contexts else None
            result = self.encode(graph, ctx)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # String Diagram Construction
    # ------------------------------------------------------------------

    def _build_string_diagram(
        self, 
        causal_graph: Dict[str, Any]
    ) -> StringDiagram:
        """Convert a causal graph (from extraction) into a string diagram."""
        diagram = StringDiagram()
        
        nodes_data = causal_graph.get("nodes", [])
        edges_data = causal_graph.get("edges", [])
        
        # Create nodes
        for node_data in nodes_data:
            nid = node_data["id"]
            
            # Determine element type from node properties
            elem_type = self._infer_element_type(node_data)
            
            # Infer causal role from connectivity
            causal_role = "leaf"  # Will be refined after edges are added
            
            node = StringDiagramNode(
                id=nid,
                element_type=elem_type,
                causal_role=causal_role,
                semantic_type=node_data.get("type", "observation"),
                state=node_data.get("strength", 0.5),
            )
            
            diagram.add_node(node)
        
        # Create edges and update causal roles
        for edge_data in edges_data:
            source = edge_data["source"]
            target = edge_data["target"]
            weight = edge_data.get("weight", 1.0)
            
            edge = StringDiagramEdge(
                source=source,
                target=target,
                weight=weight,
                composition_type="sequential",
            )
            diagram.add_edge(edge)
        
        # Update causal roles based on connectivity
        for nid in diagram.nodes:
            parents = diagram.get_parents(nid)
            children = diagram.get_children(nid)
            
            if len(parents) > 2 and len(children) > 2:
                diagram.nodes[nid].causal_role = "hub"
            elif len(parents) > 0 and len(children) > 0:
                diagram.nodes[nid].causal_role = "bridge"
            elif len(children) > len(parents):
                diagram.nodes[nid].causal_role = "root"
            else:
                diagram.nodes[nid].causal_role = "leaf"
        
        return diagram

    def _infer_element_type(self, node_data: Dict[str, Any]) -> DiagramElement:
        """Infer the string diagram element type from node properties."""
        ntype = node_data.get("type", "").lower()
        
        if "intervention" in ntype or "mechanism" in ntype:
            return DiagramElement.BOX
        elif "variable" in ntype or "entity" in ntype:
            return DiagramElement.WIRE
        elif "junction" in ntype or "fusion" in ntype:
            return DiagramElement.JUNCTION
        else:
            # Default based on connectivity (will be refined later)
            return DiagramElement.BOX

    # ------------------------------------------------------------------
    # Monoidal Composition Structure
    # ------------------------------------------------------------------

    def _compute_composition_structure(self, diagram: StringDiagram):
        """
        Compute the monoidal composition structure of the string diagram.
        
        Determines which nodes are composed sequentially (∘) vs in parallel (⊗),
        and assigns depth/width positions accordingly.
        """
        # Topological sort to determine sequential ordering
        visited = set()
        order = []
        
        def dfs(nid):
            if nid in visited:
                return
            visited.add(nid)
            for child in diagram.get_children(nid):
                dfs(child)
            order.append(nid)
        
        # Find root nodes (no parents) and traverse
        roots = [nid for nid, node in diagram.nodes.items() 
                 if not diagram.get_parents(nid)]
        
        if not roots:
            # No explicit roots — use highest state as starting point
            roots = [max(diagram.nodes.keys(), key=lambda n: diagram.nodes[n].state)]
        
        for root in roots:
            dfs(root)
        
        # Assign depth (temporal order) based on topological position
        for i, nid in enumerate(order):
            diagram.nodes[nid].depth = i
        
        # Compute width positions based on parallel composition groups
        self._assign_width_positions(diagram, order)

    def _assign_width_positions(self, diagram: StringDiagram, order: List[str]):
        """Assign horizontal (width) positions for monoidal composition."""
        # Group nodes by depth level (parallel compositions at same depth)
        depth_groups = defaultdict(list)
        for nid in order:
            depth_groups[diagram.nodes[nid].depth].append(nid)
        
        # Assign width position within each depth group
        for depth, nodes_at_depth in depth_groups.items():
            for i, nid in enumerate(nodes_at_depth):
                diagram.nodes[nid].width_pos = i

    # ------------------------------------------------------------------
    # Grid Mapping
    # ------------------------------------------------------------------

    def _map_to_grid(self, diagram: StringDiagram) -> Dict[str, Tuple[int, int]]:
        """Map string diagram nodes to 2D grid coordinates."""
        mapping = {}
        
        for nid, node in diagram.nodes.items():
            # Map depth → row (temporal dimension)
            max_depth = max(n.depth for n in diagram.nodes.values()) if diagram.nodes else 1
            row = int(node.depth / max(max_depth, 1) * (self.grid_size - 1))
            
            # Map width_pos → column (monoidal composition dimension)  
            depth_group = [n for n in diagram.nodes.values() if n.depth == node.depth]
            max_width = max(len(depth_group), 1)
            col = int(node.width_pos / max(max_width, 1) * (self.grid_size - 1))
            
            # Add some jitter to avoid exact overlaps
            row += hash(nid) % 5 - 2
            col += hash(nid + "jitter") % 5 - 2
            
            # Clamp to grid bounds
            row = max(0, min(self.grid_size - 1, row))
            col = max(0, min(self.grid_size - 1, col))
            
            mapping[nid] = (row, col)
        
        return mapping

    # ------------------------------------------------------------------
    # Memory Item Generation
    # ------------------------------------------------------------------

    def _generate_memory_items(
        self,
        diagram: StringDiagram,
        grid_mapping: Dict[str, Tuple[int, int]],
        semantic_context: Dict[str, str],
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        """Generate CA-compatible memory items from the mapped string diagram."""
        memory_items = []
        edge_pairs = []
        
        for nid, node in diagram.nodes.items():
            # Determine semantic type from context or default
            sem_type = semantic_context.get(nid, node.semantic_type)
            
            # Calculate initial state based on causal role and connectivity
            parents = diagram.get_parents(nid)
            children = diagram.get_children(nid)
            
            # Hub nodes (many connections) get higher initial activation
            base_strength = node.state
            
            if node.causal_role == "hub":
                strength = min(1.0, base_strength + 0.3)
            elif node.causal_role == "bridge":
                strength = min(1.0, base_strength + 0.15)
            else:
                strength = base_strength
            
            # Create memory item for LaceMemory.encode()
            item = {
                "content": f"causal_node:{nid}",
                "semantic_type": sem_type,
                "causal_chain": [(p, nid) for p in parents] + 
                               [(nid, c) for c in children],
                "parent_ids": parents if parents else None,
                "initial_strength": strength,
            }
            
            memory_items.append(item)
        
        # Generate edge pairs from diagram edges
        for edge in diagram.edges:
            edge_pairs.append((edge.source, edge.target))
        
        return memory_items, edge_pairs


# ============================================================================
# Causal Triplet → String Diagram Converter
# ============================================================================

def causal_triplets_to_diagram(
    triplets: List[Tuple[str, str, str]],
) -> Dict[str, Any]:
    """
    Convert extracted causal triplets (Subject, Relation, Object) into a 
    string diagram structure ready for encoding.
    
    Args:
        triplets: List of (subject, relation, object) tuples from PDF extraction
        
    Returns:
        Causal graph dict with 'nodes' and 'edges' keys
    """
    nodes = {}
    edges = []
    
    for subject, relation, obj in triplets:
        # Add nodes if they don't exist
        if subject not in nodes:
            nodes[subject] = {
                "id": subject,
                "type": "variable",
                "strength": 0.5,
            }
        
        if obj not in nodes:
            nodes[obj] = {
                "id": obj,
                "type": "variable", 
                "strength": 0.5,
            }
        
        # Add edge with relation as weight proxy
        edges.append({
            "source": subject,
            "target": obj,
            "weight": 0.7,  # Default causal strength
            "relation": relation,
        })
    
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def create_encoder(grid_size: int = 100) -> CausalStringDiagramEncoder:
    """Factory function to create a configured encoder."""
    return CausalStringDiagramEncoder(grid_size=grid_size)

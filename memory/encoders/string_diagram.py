"""
String Diagram Encoder (Refactored for General Framework)
==========================================================

Converts causal string diagrams into CA grid states. This is a specialized
encoder that inherits from the general MemoryEncoder interface, making it 
compatible with any system using the Nouse memory framework.

The encoder maps monoidal category structure to spatial layout:
  - Sequential composition (∘) → vertical stacking in grid columns
  - Parallel composition (⊗) → horizontal adjacency in grid rows  
  - Junction points (causal fusion) → high-activation nodes
  - Hub morphisms → long-term memory anchors

Usage:
    from ..encoders.string_diagram import StringDiagramEncoder
    
    encoder = StringDiagramEncoder(grid_size=(100, 100))
    
    # From extracted causal triplets (Subject, Relation, Object)
    causal_graph = {
        "nodes": [
            {"id": "X", "type": "variable"},
            {"id": "Y", "type": "intervention"},
            {"id": "Z", "type": "outcome"}
        ],
        "edges": [
            {"source": "X", "target": "Y", "weight": 0.8},
            {"source": "Y", "target": "Z", "weight": 0.7}
        ]
    }
    
    result = encoder.encode(causal_graph)
"""

from __future__ import annotations

import math
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from enum import Enum

from ..encoders.base import (
    MemoryEncoder, EncodingResult, EncodedNode, EncodedEdge,
    SemanticType, CausalRole, register_encoder
)

logger = logging.getLogger(__name__)


# ============================================================================
# String Diagram Data Model (kept for compatibility with causal extraction)
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
    id: str
    element_type: DiagramElement
    causal_role: str = "leaf"
    semantic_type: str = "observation"
    state: float = 0.5
    depth: int = 0               # Vertical position (temporal order)
    width_pos: int = 0           # Horizontal position (monoidal composition)
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)


@dataclass
class StringDiagramEdge:
    """A causal link between two string diagram nodes."""
    source: str
    target: str
    weight: float = 1.0
    composition_type: str = "sequential"


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
# String Diagram Encoder (implements MemoryEncoder interface)
# ============================================================================

@register_encoder("string_diagram")
class StringDiagramEncoder(MemoryEncoder):
    """
    Converts causal string diagrams into CA grid states.
    
    Inherits from MemoryEncoder to be compatible with the general framework,
    while providing specialized mapping for monoidal category structures.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (100, 100)):
        super().__init__(grid_size=grid_size)
    
    def encode(
        self,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EncodingResult:
        """
        Encode a causal graph (from extraction) into CA grid state.
        
        Args:
            data: Dict with 'nodes' and 'edges' from causal extraction pipeline
            metadata: Optional semantic context mapping
            
        Returns:
            EncodingResult ready for LaceMemory.encode_batch() or CAEngine placement
        """
        if not isinstance(data, dict):
            logger.warning(f"StringDiagramEncoder expects dict input, got {type(data)}")
            return EncodingResult(source_type="causal_graph", encoding_method="string_diagram")
        
        # Build string diagram from causal graph data
        diagram = self._build_string_diagram(data)
        
        # Compute monoidal composition structure
        self._compute_composition_structure(diagram)
        
        # Map to grid coordinates
        grid_mapping = self._map_to_grid(diagram)
        
        # Generate encoding result
        return self._generate_encoding_result(diagram, grid_mapping, metadata or {})
    
    def encode_batch(
        self,
        items: List[Any],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> EncodingResult:
        """Encode multiple causal graphs."""
        merged = EncodingResult(source_type="causal_graphs", encoding_method="string_diagram_batch")
        
        for i, item in enumerate(items):
            meta = metadata_list[i] if metadata_list else {}
            result = self.encode(item, meta)
            
            # Merge nodes (deduplicate by ID)
            existing_ids = {n.id for n in merged.nodes}
            for node in result.nodes:
                if node.id not in existing_ids:
                    merged.nodes.append(node)
                    existing_ids.add(node.id)
            
            merged.edges.extend(result.edges)
        
        return merged
    
    # ------------------------------------------------------------------
    # String Diagram Construction
    # ------------------------------------------------------------------

    def _build_string_diagram(self, causal_graph: Dict[str, Any]) -> StringDiagram:
        """Convert a causal graph dict into a string diagram."""
        diagram = StringDiagram()
        
        nodes_data = causal_graph.get("nodes", [])
        edges_data = causal_graph.get("edges", [])
        
        # Create nodes
        for node_data in nodes_data:
            nid = node_data["id"]
            elem_type = self._infer_element_type(node_data)
            
            node = StringDiagramNode(
                id=nid,
                element_type=elem_type,
                causal_role="leaf",  # Will be refined after edges added
                semantic_type=node_data.get("type", "observation"),
                state=node_data.get("strength", 0.5),
            )
            diagram.add_node(node)
        
        # Create edges and update connectivity
        for edge_data in edges_data:
            source = edge_data["source"]
            target = edge_data["target"]
            
            edge = StringDiagramEdge(
                source=source,
                target=target,
                weight=edge_data.get("weight", 1.0),
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
        
        return diagram
    
    def _infer_element_type(self, node_data: Dict[str, Any]) -> DiagramElement:
        """Infer string diagram element type from node properties."""
        ntype = node_data.get("type", "").lower()
        
        if "intervention" in ntype or "mechanism" in ntype:
            return DiagramElement.BOX
        elif "variable" in ntype or "entity" in ntype:
            return DiagramElement.WIRE
        elif "junction" in ntype or "fusion" in ntype:
            return DiagramElement.JUNCTION
        else:
            return DiagramElement.BOX
    
    # ------------------------------------------------------------------
    # Monoidal Composition Structure
    # ------------------------------------------------------------------

    def _compute_composition_structure(self, diagram: StringDiagram):
        """Compute monoidal composition and assign depth/width positions."""
        # Topological sort for sequential ordering
        visited = set()
        order = []
        
        def dfs(nid):
            if nid in visited:
                return
            visited.add(nid)
            for child in diagram.get_children(nid):
                dfs(child)
            order.append(nid)
        
        roots = [nid for nid, node in diagram.nodes.items() 
                 if not diagram.get_parents(nid)]
        
        if not roots:
            roots = [max(diagram.nodes.keys(), key=lambda n: diagram.nodes[n].state)]
        
        for root in roots:
            dfs(root)
        
        # Assign depth (temporal order)
        for i, nid in enumerate(order):
            diagram.nodes[nid].depth = i
        
        # Assign width positions within each depth level
        depth_groups = defaultdict(list)
        for nid in order:
            depth_groups[diagram.nodes[nid].depth].append(nid)
        
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
            max_depth = max(n.depth for n in diagram.nodes.values()) if diagram.nodes else 1
            
            # Depth → row (temporal dimension)
            row = int(node.depth / max(max_depth, 1) * (self.grid_size[0] - 1))
            
            # Width position → column (monoidal composition dimension)
            depth_group = [n for n in diagram.nodes.values() if n.depth == node.depth]
            max_width = max(len(depth_group), 1)
            col = int(node.width_pos / max(max_width, 1) * (self.grid_size[1] - 1))
            
            # Add jitter to avoid exact overlaps
            row += hash(nid) % 5 - 2
            col += hash(nid + "jitter") % 5 - 2
            
            mapping[nid] = self._clamp_position((row, col))
        
        return mapping
    
    # ------------------------------------------------------------------
    # Encoding Result Generation
    # ------------------------------------------------------------------

    def _generate_encoding_result(
        self,
        diagram: StringDiagram,
        grid_mapping: Dict[str, Tuple[int, int]],
        semantic_context: Dict[str, str],
    ) -> EncodingResult:
        """Generate CA-compatible encoding result from mapped string diagram."""
        nodes = []
        edges = []
        
        for nid, node in diagram.nodes.items():
            sem_type_str = semantic_context.get(nid, node.semantic_type)
            
            try:
                sem_type = SemanticType(sem_type_str)
            except ValueError:
                sem_type = SemanticType.OBSERVATION
            
            causal_role_map = {
                "root": CausalRole.ROOT,
                "hub": CausalRole.HUB, 
                "bridge": CausalRole.BRIDGE,
                "leaf": CausalRole.LEAF,
            }
            causal_role = causal_role_map.get(node.causal_role, CausalRole.LEAF)
            
            # Calculate initial state based on causal role
            base_strength = node.state
            if node.causal_role == "hub":
                strength = min(1.0, base_strength + 0.3)
            elif node.causal_role == "bridge":
                strength = min(1.0, base_strength + 0.15)
            else:
                strength = base_strength
            
            position = grid_mapping.get(nid, (0, 0))
            
            nodes.append(EncodedNode(
                id=f"sd_{nid}",
                content=nid,
                position=position,
                state=strength,
                semantic_type=sem_type,
                causal_role=causal_role,
                metadata={"element_type": node.element_type.value},
            ))
        
        # Generate edges from diagram edges
        for edge in diagram.edges:
            edges.append(EncodedEdge(
                source=f"sd_{edge.source}",
                target=f"sd_{edge.target}",
                weight=edge.weight,
                causal_direction="forward",  # Causal chains are directional
            ))
        
        return EncodingResult(
            nodes=nodes,
            edges=edges,
            source_type="string_diagram",
            encoding_method="monoidal_mapping",
        )


# ============================================================================
# Convenience Functions (backward compatibility)
# ============================================================================

def causal_triplets_to_diagram(triplets: List[Tuple[str, str, str]]) -> Dict[str, Any]:
    """Convert extracted causal triplets into string diagram structure."""
    nodes = {}
    edges = []
    
    for subject, relation, obj in triplets:
        if subject not in nodes:
            nodes[subject] = {"id": subject, "type": "variable", "strength": 0.5}
        if obj not in nodes:
            nodes[obj] = {"id": obj, "type": "variable", "strength": 0.5}
        
        edges.append({
            "source": subject,
            "target": obj,
            "weight": 0.7,
            "relation": relation,
        })
    
    return {"nodes": list(nodes.values()), "edges": edges}


def create_string_diagram_encoder(grid_size: Tuple[int, int] = (100, 100)) -> StringDiagramEncoder:
    """Factory function to create a configured string diagram encoder."""
    return StringDiagramEncoder(grid_size=grid_size)

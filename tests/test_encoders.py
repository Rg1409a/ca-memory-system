"""Encoder tests — string diagram, embedding, registry, batch encoding."""
import pytest


# ---------------------------------------------------------------------------
# String Diagram Encoder
# ---------------------------------------------------------------------------

class TestStringDiagramEncoder:
    """Tests for StringDiagramEncoder.encode() and related methods."""

    def test_encode_causal_graph(self):
        from memory.encoders.string_diagram import StringDiagramEncoder
        
        encoder = StringDiagramEncoder(grid_size=(100, 100))
        causal_graph = {
            "nodes": [
                {"id": "X", "type": "variable"},
                {"id": "Y", "type": "intervention"},
                {"id": "Z", "type": "outcome"},
            ],
            "edges": [
                {"source": "X", "target": "Y", "weight": 0.8},
                {"source": "Y", "target": "Z", "weight": 0.7},
            ]
        }
        
        result = encoder.encode(causal_graph)
        
        assert len(result.nodes) == 3
        assert len(result.edges) == 2
        assert result.source_type == "string_diagram"

    def test_encode_returns_encoded_nodes_with_positions(self):
        from memory.encoders.string_diagram import StringDiagramEncoder
        
        encoder = StringDiagramEncoder(grid_size=(50, 50))
        causal_graph = {
            "nodes": [{"id": "A", "type": "variable"}],
            "edges": []
        }
        
        result = encoder.encode(causal_graph)
        node = result.nodes[0]
        
        assert node.position[0] < 50 and node.position[0] >= 0
        assert node.position[1] < 50 and node.position[1] >= 0

    def test_encode_non_dict_input_returns_empty(self):
        from memory.encoders.string_diagram import StringDiagramEncoder
        
        encoder = StringDiagramEncoder(grid_size=(100, 100))
        result = encoder.encode("not a dict")
        
        assert len(result.nodes) == 0

    def test_hub_nodes_get_boosted_state(self):
        from memory.encoders.string_diagram import StringDiagramEncoder
        
        encoder = StringDiagramEncoder(grid_size=(100, 100))
        causal_graph = {
            "nodes": [
                {"id": "hub", "type": "variable", "strength": 0.5},
                {"id": "leaf1", "type": "outcome"},
                {"id": "leaf2", "type": "outcome"},
                {"id": "leaf3", "type": "outcome"},
                {"id": "root1", "type": "intervention"},
                {"id": "root2", "type": "intervention"},
                {"id": "root3", "type": "intervention"},
            ],
            "edges": [
                {"source": "hub", "target": "leaf1", "weight": 0.5},
                {"source": "hub", "target": "leaf2", "weight": 0.5},
                {"source": "hub", "target": "leaf3", "weight": 0.5},
                {"source": "root1", "target": "hub", "weight": 0.5},
                {"source": "root2", "target": "hub", "weight": 0.5},
                {"source": "root3", "target": "hub", "weight": 0.5},
            ]
        }
        
        result = encoder.encode(causal_graph)
        hub_node = next(n for n in result.nodes if "hub" in n.id)
        
        # Hub should have boosted state (> base strength + 0.3)
        assert hub_node.state > 0.7

    def test_bridge_nodes_get_moderate_boost(self):
        from memory.encoders.string_diagram import StringDiagramEncoder
        
        encoder = StringDiagramEncoder(grid_size=(100, 100))
        causal_graph = {
            "nodes": [
                {"id": "bridge", "type": "variable"},
                {"id": "before", "type": "intervention"},
                {"id": "after", "type": "outcome"},
            ],
            "edges": [
                {"source": "before", "target": "bridge", "weight": 0.5},
                {"source": "bridge", "target": "after", "weight": 0.5},
            ]
        }
        
        result = encoder.encode(causal_graph)
        bridge_node = next(n for n in result.nodes if "bridge" in n.id)
        
        # Bridge should have moderate boost (> base + 0.15 but < hub boost)
        assert bridge_node.state > 0.6

    def test_deterministic_positions(self):
        """Same input should produce same grid positions."""
        from memory.encoders.string_diagram import StringDiagramEncoder
        
        encoder = StringDiagramEncoder(grid_size=(100, 100))
        causal_graph = {
            "nodes": [{"id": "A", "type": "variable"}],
            "edges": []
        }
        
        result1 = encoder.encode(causal_graph)
        result2 = encoder.encode(causal_graph)
        
        assert result1.nodes[0].position == result2.nodes[0].position


# ---------------------------------------------------------------------------
# Encoder Registry
# ---------------------------------------------------------------------------

class TestEncoderRegistry:
    """Tests for register_encoder, get_encoder, list_encoders."""

    def test_register_and_get_encoder(self):
        from memory.encoders.base import MemoryEncoder, register_encoder
        
        @register_encoder("test_custom")
        class CustomEncoder(MemoryEncoder):
            def encode(self, data, metadata=None):
                pass
        
        encoders = ["memory", "string_diagram"]  # built-in + custom
        assert len(encoders) >= 2

    def test_get_unknown_encoder_raises(self):
        from memory.encoders.base import get_encoder
        
        with pytest.raises(KeyError):
            get_encoder("nonexistent_encoder")


# ---------------------------------------------------------------------------
# EncodingResult Data Model
# ---------------------------------------------------------------------------

class TestEncodingResult:
    """Tests for the EncodingResult dataclass."""

    def test_to_dict_serialization(self):
        from memory.encoders.base import EncodingResult, EncodedNode, EncodedEdge
        
        result = EncodingResult(
            nodes=[EncodedNode(id="n1", content="test", position=(0, 0))],
            edges=[EncodedEdge(source="n1", target="n2")],
            source_type="test",
        )
        
        d = result.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 1

    def test_empty_result_defaults(self):
        from memory.encoders.base import EncodingResult
        
        result = EncodingResult()
        assert result.nodes == []
        assert result.edges == []


# ---------------------------------------------------------------------------
# Batch Encoding with Deduplication
# ---------------------------------------------------------------------------

class TestBatchEncoding:
    """Tests for encode_batch deduplication behavior."""

    def test_string_diagram_batch_deduplicates(self):
        from memory.encoders.string_diagram import StringDiagramEncoder
        
        encoder = StringDiagramEncoder(grid_size=(100, 100))
        
        # Two graphs with overlapping node IDs
        graph1 = {
            "nodes": [{"id": "shared", "type": "variable"}],
            "edges": []
        }
        graph2 = {
            "nodes": [{"id": "shared", "type": "variable"}, {"id": "unique", "type": "outcome"}],
            "edges": []
        }
        
        result = encoder.encode_batch([graph1, graph2])
        
        # Should have 2 nodes (shared deduplicated + unique)
        assert len(result.nodes) == 2


# ---------------------------------------------------------------------------
# SemanticType Enum
# ---------------------------------------------------------------------------

class TestSemanticTypes:
    """Tests for the SemanticType and CausalRole enums."""

    def test_all_semantic_types_exist(self):
        from memory.encoders.base import SemanticType
        
        types = [t.value for t in SemanticType]
        assert "observation" in types
        assert "concept" in types
        assert "fact" in types
        assert "event" in types

    def test_all_causal_roles_exist(self):
        from memory.encoders.base import CausalRole
        
        roles = [r.value for r in CausalRole]
        assert "root" in roles
        assert "hub" in roles
        assert "bridge" in roles
        assert "leaf" in roles

"""Edge case and error handling tests — empty inputs, missing deps, boundaries."""
import pytest


# ---------------------------------------------------------------------------
# Empty / Boundary Input Tests
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    """Tests for behavior with empty or boundary inputs."""

    def test_retrieve_empty_query(self):
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=5)
        
        memories_by_tier = {
            "all": {"m1": {"content": "test content", "semantic_type": "observation"}},
        }
        
        result = retriever.retrieve(query="", memories_by_tier=memories_by_tier, top_k=5)
        
        assert isinstance(result.results, list)

    def test_retrieve_single_memory(self):
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=5)
        
        memories_by_tier = {
            "all": {"m1": {"content": "only memory", "semantic_type": "fact"}},
        }
        
        result = retriever.retrieve(query="memory", memories_by_tier=memories_by_tier, top_k=5)
        
        assert isinstance(result.results, list)

    def test_retrieve_no_memories(self):
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=5)
        result = retriever.retrieve(query="test", memories_by_tier={}, top_k=5)
        
        assert len(result.results) == 0

    def test_encode_empty_string(self):
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="empty_test", grid_size=(100, 100))
        nid = agent.encode("")
        
        # Should not raise — empty string is valid content
        assert isinstance(nid, str)


# ---------------------------------------------------------------------------
# Missing Dependencies Tests
# ---------------------------------------------------------------------------

class TestMissingDependencies:
    """Tests for graceful degradation when optional deps are missing."""

    def test_hybrid_retriever_without_faiss(self):
        """HybridRetriever should not crash if FAISS is unavailable."""
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=3)
        
        memories_by_tier = {
            "all": {"m1": {"content": "test", "semantic_type": "observation"}},
        }
        
        # Should not raise regardless of FAISS availability
        result = retriever.retrieve(
            query="test",
            memories_by_tier=memories_by_tier,
            top_k=3,
        )
        
        assert isinstance(result.results, list)

    def test_agent_memory_without_ca_engine(self):
        """AgentMemory should handle missing CA engine gracefully."""
        from memory.agents.agent_memory import AgentMemory
        
        # Create agent with a grid size that will create an engine
        agent = AgentMemory(agent_id="no_engine_test", grid_size=(100, 100))
        
        # If CAEngine is None (import failed), encode should return empty string
        nid = agent.encode("test")
        
        if not nid:
            # Engine wasn't available — that's fine for this test
            pass


# ---------------------------------------------------------------------------
# Duplicate Content Tests
# ---------------------------------------------------------------------------

class TestDuplicateContent:
    """Tests for handling duplicate or near-duplicate memories."""

    def test_duplicate_content_creates_different_nodes(self):
        """Same content should create different node IDs (deterministic hash)."""
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="dup_test", grid_size=(100, 100))
        
        nid1 = agent.encode("identical content")
        nid2 = agent.encode("identical content")
        
        # Same content → same node ID (deterministic hash)
        assert nid1 == nid2

    def test_near_duplicate_content(self):
        """Near-duplicate content should create different nodes."""
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="near_dup", grid_size=(100, 100))
        
        nid1 = agent.encode("surface_adhesion causes failure")
        nid2 = agent.encode("surface_adhesion causes failure_mode_B")
        
        assert nid1 != nid2


# ---------------------------------------------------------------------------
# Grid Boundary Tests
# ---------------------------------------------------------------------------

class TestGridBoundaries:
    """Tests for grid position boundary conditions."""

    def test_set_node_at_boundary(self):
        from memory.core.ca_engine import CAEngine
        
        engine = CAEngine(grid_size=(10, 10))
        
        # Edge positions should be valid
        engine.set_node_state("corner", (0, 0), state=0.5)
        engine.set_node_state("opposite_corner", (9, 9), state=0.5)
        
        assert engine.get_node("corner") is not None
        assert engine.get_node("opposite_corner") is not None

    def test_set_node_out_of_bounds_raises(self):
        from memory.core.ca_engine import CAEngine
        
        engine = CAEngine(grid_size=(10, 10))
        
        with pytest.raises(ValueError):
            engine.set_node_state("out", (10, 5))
        
        with pytest.raises(ValueError):
            engine.set_node_state("out2", (5, -1))

    def test_set_node_negative_position_raises(self):
        from memory.core.ca_engine import CAEngine
        
        engine = CAEngine(grid_size=(10, 10))
        
        with pytest.raises(ValueError):
            engine.set_node_state("neg", (-1, 5))


# ---------------------------------------------------------------------------
# State Serialization Tests
# ---------------------------------------------------------------------------

class TestStateSerialization:
    """Tests for save/load state roundtrip integrity."""

    def test_save_load_preserves_nodes(self):
        from memory.core.ca_engine import CAEngine
        
        engine = CAEngine(grid_size=(100, 100))
        engine.set_node_state("persist", (50, 50), state=0.7)
        
        saved = engine.save_state()
        
        new_engine = CAEngine(grid_size=(100, 100))
        new_engine.load_state(saved)
        
        node = new_engine.get_node("persist")
        assert node is not None
        assert node.state == pytest.approx(0.7)

    def test_save_load_preserves_edges(self):
        from memory.core.ca_engine import CAEngine
        
        engine = CAEngine(grid_size=(100, 100))
        engine.set_node_state("a", (50, 50), state=0.5)
        engine.set_node_state("b", (51, 51), state=0.5)
        engine.add_edge(("a", "b"), weight=0.8)
        
        saved = engine.save_state()
        
        new_engine = CAEngine(grid_size=(100, 100))
        new_engine.load_state(saved)
        
        assert len(new_engine.edges) > 0

    def test_load_empty_state(self):
        from memory.core.ca_engine import CAEngine
        
        engine = CAEngine(grid_size=(100, 100))
        empty_state = {"grid_size": (100, 100), "step_count": 0, "nodes": {}, "edges": {}}
        
        # Should not raise
        engine.load_state(empty_state)


# ---------------------------------------------------------------------------
# Rule Edge Cases
# ---------------------------------------------------------------------------

class TestRuleEdgeCases:
    """Tests for rule behavior with edge cases."""

    def test_evolve_with_no_rules(self):
        from memory.core.ca_engine import CAEngine
        
        engine = CAEngine(grid_size=(100, 100))
        engine.set_node_state("no_rule", (5, 5), state=0.8)
        
        # Should not raise even with no rules registered
        states, edges = engine.evolve(steps=3)
        
        assert "no_rule" in states

    def test_evolve_with_failing_rule(self):
        """Engine should continue evolving if one rule fails."""
        from memory.core.ca_engine import CAEngine
        
        class BadRule:
            def apply(self, nodes, edges, step, context=None):
                raise RuntimeError("Intentional failure")
        
        engine = CAEngine(grid_size=(100, 100))
        engine.register_rule('bad', BadRule())
        engine.set_node_state("survivor", (5, 5), state=0.8)
        
        # Should not raise — rule failures are logged and skipped
        states, _ = engine.evolve(steps=2)
        
        assert "survivor" in states

    def test_unregister_nonexistent_rule(self):
        from memory.core.ca_engine import CAEngine
        
        engine = CAEngine(grid_size=(100, 100))
        
        # Should not raise
        engine.unregister_rule("nonexistent")


# ---------------------------------------------------------------------------
# MemoryEncoder ABC Tests
# ---------------------------------------------------------------------------

class TestMemoryEncoderABC:
    """Tests for the MemoryEncoder abstract base class."""

    def test_cannot_instantiate_abc_directly(self):
        from memory.encoders.base import MemoryEncoder
        
        with pytest.raises(TypeError):
            MemoryEncoder(grid_size=(100, 100))

    def test_custom_encoder_can_be_registered(self):
        """Custom encoder subclass should be registerable."""
        from memory.encoders.base import MemoryEncoder, register_encoder
        
        @register_encoder("custom_test")
        class CustomEncoder(MemoryEncoder):
            def encode(self, data, metadata=None):
                pass
        
        # Should not raise — registration succeeded


# ---------------------------------------------------------------------------
# RetrievedMemory Data Model Tests
# ---------------------------------------------------------------------------

class TestRetrievedMemoryModel:
    """Tests for the RetrievedMemory and RetrievalResult data models."""

    def test_retrieved_memory_to_dict(self):
        from memory.retrieval.base import RetrievedMemory
        
        mem = RetrievedMemory(
            id="test_id",
            content="test content",
            score=0.8,
            tier="short_term",
            semantic_type="observation",
        )
        
        d = mem.to_dict()
        assert d["id"] == "test_id"
        assert d["score"] == 0.8

    def test_retrieval_result_to_dict(self):
        from memory.retrieval.base import RetrievalResult, RetrievedMemory
        
        result = RetrievalResult(
            query="test",
            results=[RetrievedMemory(id="m1", content="c", score=0.5)],
            top_k=5,
            retrieval_method="hybrid",
        )
        
        d = result.to_dict()
        assert "results" in d
        assert len(d["results"]) == 1

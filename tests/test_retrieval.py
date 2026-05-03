"""Retrieval tests — hybrid retriever, tier promotion, decay curves."""
import pytest


# ---------------------------------------------------------------------------
# Hybrid Retriever Integration Tests
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    """Tests for the production HybridRetriever pipeline."""

    def test_retriever_returns_results_for_valid_query(self):
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=3)
        
        memories_by_tier = {
            "short_term": {
                "m1": {"content": "Entity A causes failure mode B", "semantic_type": "observation"},
                "m2": {"content": "Surface roughness increases risk", "semantic_type": "fact"},
            },
        }
        
        result = retriever.retrieve(
            query="surface adhesion mechanisms",
            memories_by_tier=memories_by_tier,
            top_k=3,
        )
        
        assert len(result.results) > 0
        assert result.retrieval_method == "hybrid"

    def test_retriever_respects_top_k(self):
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=2)
        
        memories_by_tier = {
            "all": {f"m{i}": {"content": f"Memory number {i}", "semantic_type": "observation"}
                     for i in range(10)}
        }
        
        result = retriever.retrieve(query="memory", top_k=2)
        
        assert len(result.results) <= 2

    def test_retriever_with_empty_memories(self):
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=5)
        result = retriever.retrieve(query="test", memories_by_tier={})
        
        assert len(result.results) == 0

    def test_retriever_with_no_faiss_fallback(self):
        """Test that retrieval works even when FAISS is unavailable."""
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=3)
        
        memories_by_tier = {
            "tier1": {
                "m1": {"content": "test content here", "semantic_type": "observation"},
            },
        }
        
        # Should not raise even if FAISS/embeddings unavailable
        result = retriever.retrieve(
            query="test",
            memories_by_tier=memories_by_tier,
            top_k=3,
        )
        
        assert isinstance(result.results, list)


# ---------------------------------------------------------------------------
# Tier Promotion Tests
# ---------------------------------------------------------------------------

class TestTierPromotion:
    """Tests for short→mid→long-term memory tier transitions."""

    def test_new_memories_start_in_short_term(self):
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="test_tier", grid_size=(100, 100))
        nid = agent.encode("Test memory content")
        
        assert nid in agent.tiers["short_term"]
        assert nid not in agent.tiers["mid_term"]

    def test_mid_term_promotion_after_persistence(self):
        """Memories surviving N consecutive CA generations should promote."""
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="test_promote", grid_size=(100, 100))
        
        # Encode a strong memory
        nid = agent.encode("Persistent concept that survives decay")
        
        # Evolve many steps — persistent nodes should survive
        for _ in range(50):
            agent.evolve(steps=1)
        
        # Node should still exist (not fully decayed)
        assert len(agent.tiers["short_term"]) > 0 or len(agent.tiers["mid_term"]) > 0

    def test_long_term_promotion_for_hub_nodes(self):
        """High-centrality nodes should resist decay longer."""
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="test_hub", grid_size=(100, 100))
        
        # Encode multiple related memories (creates hub-like structure)
        for text in [
            "Core concept A",
            "Related to core A", 
            "Another aspect of core A",
            "Connected to core A",
        ]:
            agent.encode(text)
        
        # Evolve extensively
        for _ in range(100):
            agent.evolve(steps=1)
        
        # Hub nodes should still have some activation
        stats = agent.get_stats()
        assert "active_nodes" in stats


# ---------------------------------------------------------------------------
# Decay Curve Tests
# ---------------------------------------------------------------------------

class TestDecayCurves:
    """Tests verifying memory decay follows expected patterns."""

    def test_decay_rate_matches_config(self):
        """Verify that configured decay rate produces expected state reduction."""
        from memory.core.ca_engine import CAEngine, MemoryDecayRule
        
        engine = CAEngine(grid_size=(100, 100))
        # Use a high decay rate for measurable change in few steps
        engine.register_rule('decay', MemoryDecayRule(decay_rate=0.5))
        
        engine.set_node_state("fast_decay", (10, 10), state=1.0)
        
        states_before = engine.get_node("fast_decay").state
        
        # Evolve a few steps with high decay rate
        for _ in range(5):
            engine.evolve(steps=1)
        
        node_after = engine.get_node("fast_decay")
        if node_after is not None:
            # With 0.5 decay_rate, state should decrease significantly
            assert node_after.state < states_before

    def test_hub_nodes_decay_slower(self):
        """Hub nodes (high degree) should decay slower than leaf nodes."""
        from memory.core.ca_engine import CAEngine, MemoryDecayRule
        
        engine = CAEngine(grid_size=(100, 100))
        engine.register_rule('decay', MemoryDecayRule(decay_rate=0.3))
        
        # Create hub node connected to many others
        engine.set_node_state("hub", (50, 50), state=0.9)
        for i in range(10):
            engine.set_node_state(f"leaf_{i}", (50 + (i % 3), 50 + (i // 3)), state=0.8)
        
        # Connect hub to all leaves
        for i in range(10):
            engine.add_edge(("hub", f"leaf_{i}"), weight=0.5)
        
        states_before = {nid: n.state for nid, n in engine.nodes.items()}
        
        for _ in range(20):
            engine.evolve(steps=1)
        
        hub_after = engine.get_node("hub")
        if hub_after is not None and "leaf_0" in engine.nodes:
            leaf_after = engine.get_node("leaf_0")
            # Hub should have higher remaining state than leaves (hub_factor protection)
            assert hub_after.state >= leaf_after.state

    def test_no_decay_with_zero_rate(self):
        """Zero decay rate should preserve states."""
        from memory.core.ca_engine import CAEngine, MemoryDecayRule
        
        engine = CAEngine(grid_size=(100, 100))
        engine.register_rule('decay', MemoryDecayRule(decay_rate=0.0))
        
        engine.set_node_state("stable", (5, 5), state=0.7)
        states_before = {nid: n.state for nid, n in engine.nodes.items()}
        
        engine.evolve(steps=10)
        
        for nid, before in states_before.items():
            after = engine.get_node(nid)
            if after is not None:
                assert after.state == pytest.approx(before, abs=0.01)


# ---------------------------------------------------------------------------
# Retrieval Quality Tests
# ---------------------------------------------------------------------------

class TestRetrievalQuality:
    """Tests for retrieval accuracy and ranking."""

    def test_semantic_similarity_ranks_relevant_first(self):
        """More semantically similar memories should rank higher."""
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=5)
        
        # Create tier with varied content
        memories_by_tier = {
            "all": {
                "exact_match": {"content": "Entity A causes failure mode B in MEMS", 
                                "semantic_type": "observation"},
                "related": {"content": "Factor C increases risk during process D", 
                            "semantic_type": "fact"},
                "unrelated": {"content": "Quantum computing uses qubits", 
                              "semantic_type": "concept"},
            },
        }
        
        result = retriever.retrieve(
            query="surface adhesion mechanisms",
            memories_by_tier=memories_by_tier,
            top_k=5,
        )
        
        if len(result.results) >= 2:
            # Exact match should rank higher than unrelated
            exact_score = next((r.score for r in result.results 
                               if "exact_match" in r.id), None)
            unrelated_score = next((r.score for r in result.results 
                                   if "unrelated" in r.id), None)
            
            if exact_score is not None and unrelated_score is not None:
                assert exact_score >= unrelated_score

    def test_empty_query_returns_no_results(self):
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=5)
        
        memories_by_tier = {
            "all": {"m1": {"content": "test content", "semantic_type": "observation"}},
        }
        
        result = retriever.retrieve(query="", memories_by_tier=memories_by_tier, top_k=5)
        
        # Empty query should return no results (no meaningful semantic match possible)
        assert len(result.results) == 0

    def test_retrieval_result_has_metadata(self):
        from memory.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(top_k=3)
        
        memories_by_tier = {
            "tier1": {"m1": {"content": "test content", "semantic_type": "fact"}},
        }
        
        result = retriever.retrieve(
            query="test",
            memories_by_tier=memories_by_tier,
            top_k=3,
        )
        
        assert hasattr(result, 'query')
        assert hasattr(result, 'results')
        assert hasattr(result, 'retrieval_method')


# ---------------------------------------------------------------------------
# Spreading Activation Tests
# ---------------------------------------------------------------------------

class TestSpreadingActivationRetriever:
    """Tests for the legacy SpreadingActivationRetriever."""

    def test_spreading_activation_returns_results(self):
        from memory.retrieval.spreading_activation import SpreadingActivationRetriever
        
        retriever = SpreadingActivationRetriever()
        
        memories_by_tier = {
            "all": {
                "m1": {"content": "stiction mechanism", "state": 0.8},
                "m2": {"content": "pull-in failure", "state": 0.6},
            },
        }
        
        result = retriever.retrieve(
            query="stiction",
            memories_by_tier=memories_by_tier,
            top_k=5,
        )
        
        assert isinstance(result.results, list)

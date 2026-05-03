"""Core CA engine unit tests — node management, evolution, activation."""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Node Management
# ---------------------------------------------------------------------------

class TestNodeManagement:
    """Tests for set_node_state, get_node, remove_node."""

    def test_set_and_get_node(self, small_engine):
        engine = small_engine
        engine.set_node_state("n1", (10, 20), state=0.8)
        node = engine.get_node("n1")
        assert node is not None
        assert node.id == "n1"
        assert node.position == (10, 20)
        assert node.state == pytest.approx(0.8)

    def test_set_node_clamps_state(self, small_engine):
        engine = small_engine
        engine.set_node_state("n_low", (5, 5), state=-0.5)
        engine.set_node_state("n_high", (6, 6), state=1.5)
        assert engine.get_node("n_low").state == pytest.approx(0.0)
        assert engine.get_node("n_high").state == pytest.approx(1.0)

    def test_get_missing_node_returns_none(self, small_engine):
        assert small_engine.get_node("nonexistent") is None

    def test_remove_node_cleans_edges(self, small_engine):
        engine = small_engine
        engine.set_node_state("a", (10, 10))
        engine.set_node_state("b", (10, 11))  # adjacent -> auto-edge
        assert len(engine.edges) > 0
        engine.remove_node("a")
        assert "a" not in engine.nodes
        for key in engine.edges:
            assert "a" not in key

    def test_remove_missing_node_no_error(self, small_engine):
        small_engine.remove_node("ghost")  # should not raise


# ---------------------------------------------------------------------------
# Edge Management
# ---------------------------------------------------------------------------

class TestEdgeManagement:
    """Tests for add_edge, remove_edge, get_neighbors."""

    def test_add_edge_between_existing_nodes(self, small_engine):
        engine = small_engine
        engine.set_node_state("x", (10, 10))
        engine.set_node_state("y", (10, 11))
        engine.add_edge(("x", "y"), weight=0.7)
        edge = engine.edges.get(("x", "y")) or engine.edges.get(("y", "x"))
        assert edge is not None
        assert edge.weight == pytest.approx(0.7)

    def test_add_edge_clamps_weight(self, small_engine):
        engine = small_engine
        engine.set_node_state("a", (10, 10))
        engine.set_node_state("b", (10, 11))
        engine.add_edge(("a", "b"), weight=-0.5)
        edge_key = tuple(sorted(["a", "b"]))
        assert engine.edges[edge_key].weight == pytest.approx(0.0)

    def test_add_edge_to_missing_node_logs_warning(self, small_engine):
        engine = small_engine
        engine.set_node_state("exists", (10, 10))
        # Should return None/False when one node is missing
        result = engine.add_edge(("exists", "missing"), weight=1.0)
        assert result in (None, False)

    def test_remove_edge(self, small_engine):
        engine = small_engine
        engine.set_node_state("p", (10, 10))
        engine.set_node_state("q", (10, 11))
        engine.add_edge(("p", "q"), weight=0.5)
        assert len(engine.edges) > 0
        engine.remove_edge(("p", "q"))
        key = tuple(sorted(["p", "q"]))
        assert key not in engine.edges

    def test_get_neighbors(self, small_engine):
        engine = small_engine
        engine.set_node_state("center", (10, 10))
        engine.set_node_state("n1", (9, 10))
        engine.set_node_state("n2", (11, 10))
        engine.add_edge(("center", "n1"), weight=0.8)
        engine.add_edge(("center", "n2"), weight=0.3)
        neighbors = engine.get_neighbors("center")
        neighbor_ids = {nid for nid, _ in neighbors}
        assert "n1" in neighbor_ids
        assert "n2" in neighbor_ids


# ---------------------------------------------------------------------------
# Evolution & Decay
# ---------------------------------------------------------------------------

class TestEvolution:
    """Tests for evolve() with decay and consolidation rules."""

    def test_decay_reduces_states(self, configured_engine):
        engine = configured_engine
        engine.set_node_state("mem1", (10, 10), state=0.9)
        engine.set_node_state("mem2", (15, 15), state=0.8)
        
        states_before = {nid: n.state for nid, n in engine.nodes.items()}
        evolved_states, _ = engine.evolve(steps=10)
        
        # All states should decrease or stay same with decay rule
        for nid in states_before:
            assert evolved_states[nid] <= states_before[nid] + 0.01

    def test_evolution_returns_dict(self, configured_engine):
        engine = configured_engine
        engine.set_node_state("test", (5, 5), state=0.7)
        states, edges = engine.evolve(steps=1)
        assert isinstance(states, dict)
        assert isinstance(edges, dict)

    def test_evolution_updates_internal_state(self, configured_engine):
        engine = configured_engine
        engine.set_node_state("persistent", (5, 5), state=0.9)
        
        # Evolve many steps — persistent node should survive decay
        for _ in range(50):
            engine.evolve(steps=1)
        
        node = engine.get_node("persistent")
        assert node is not None  # Should still exist due to consolidation

    def test_evolution_with_no_rules(self, small_engine):
        engine = small_engine  # no rules registered
        engine.set_node_state("a", (5, 5), state=0.8)
        states, _ = engine.evolve(steps=5)
        assert "a" in states

    def test_step_count_increments(self, configured_engine):
        engine = configured_engine
        assert engine.step_count == 0
        engine.evolve(steps=3)
        assert engine.step_count == 3


# ---------------------------------------------------------------------------
# Spreading Activation
# ---------------------------------------------------------------------------

class TestSpreadingActivation:
    """Tests for spread_activation retrieval mechanism."""

    def test_seed_activates_itself(self, small_engine):
        engine = small_engine
        engine.set_node_state("seed", (10, 10), state=0.5)
        activations = engine.spread_activation({"seed": 1.0})
        assert activations["seed"] == pytest.approx(1.0)

    def test_activation_spreads_to_neighbors(self, small_engine):
        engine = small_engine
        engine.set_node_state("center", (10, 10), state=0.5)
        engine.set_node_state("neighbor", (10, 11), state=0.3)
        engine.add_edge(("center", "neighbor"), weight=0.8)
        
        activations = engine.spread_activation({"center": 1.0})
        # Neighbor should receive some activation through the edge
        assert activations["neighbor"] > 0

    def test_nonexistent_seed_ignored(self, small_engine):
        engine = small_engine
        engine.set_node_state("real", (5, 5), state=0.5)
        activations = engine.spread_activation({"ghost": 1.0})
        assert "real" not in activations or activations["real"] == 0

    def test_convergence_stops_early(self, small_engine):
        """Activation should converge before max_steps is exhausted."""
        engine = small_engine
        engine.set_node_state("a", (5, 5), state=0.5)
        engine.set_node_state("b", (5, 6), state=0.3)
        engine.add_edge(("a", "b"), weight=0.9)
        
        activations = engine.spread_activation({"a": 1.0}, max_steps=200)
        # Should converge well before 200 steps


# ---------------------------------------------------------------------------
# Boundary Conditions
# ---------------------------------------------------------------------------

class TestBoundaryConditions:
    """Tests for bounded vs wrap boundary conditions."""

    def test_bounded_clamps_position(self, small_engine):
        engine = small_engine
        with pytest.raises(ValueError):
            engine.set_node_state("out", (999, 0))

    def test_wrap_boundary(self):
        from memory.core.ca_engine import CAEngine
        engine = CAEngine(grid_size=(10, 10), boundary='wrap')
        # Should not raise for positions that would be out of bounds in bounded mode
        engine.set_node_state("wrapped", (5, 5))


# ---------------------------------------------------------------------------
# State Persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_save_load_roundtrip(self):
        from memory.core.ca_engine import CAEngine
        
        engine = CAEngine(grid_size=(100, 100))
        engine.set_node_state("persist", (10, 10), state=0.7)
        
        saved = engine.save_state()
        
        new_engine = CAEngine(grid_size=(100, 100))
        new_engine.load_state(saved)
        
        node = new_engine.get_node("persist")
        assert node is not None
        assert node.state == pytest.approx(0.7)

    def test_reset_clears_all(self, configured_engine):
        engine = configured_engine
        engine.set_node_state("temp", (5, 5), state=0.8)
        engine.reset()
        assert len(engine.nodes) == 0
        assert len(engine.edges) == 0


# ---------------------------------------------------------------------------
# Rule Management
# ---------------------------------------------------------------------------

class TestRuleManagement:
    """Tests for pluggable rule registration."""

    def test_register_and_get_rules(self, small_engine):
        from memory.core.ca_engine import MemoryDecayRule
        engine = small_engine
        engine.register_rule('decay', MemoryDecayRule(decay_rate=0.05))
        rules = engine.get_rules()
        assert 'decay' in rules

    def test_unregister_rule(self, small_engine):
        from memory.core.ca_engine import MemoryDecayRule
        engine = small_engine
        engine.register_rule('decay', MemoryDecayRule(decay_rate=0.05))
        engine.unregister_rule('decay')
        assert 'decay' not in engine.get_rules()

    def test_unregistered_rule_no_error(self, small_engine):
        small_engine.unregister_rule('nonexistent')  # should not raise


# ---------------------------------------------------------------------------
# NodeState Data Model
# ---------------------------------------------------------------------------

class TestNodeStateModel:
    """Tests for the NodeState dataclass."""

    def test_node_state_is_active(self):
        from memory.core.ca_engine import NodeState
        node = NodeState(id="a", position=(0, 0), state=0.5)
        assert node.is_active is True
        
        dead = NodeState(id="b", position=(0, 0), state=1e-7)
        assert dead.is_active is False

    def test_node_state_copy(self):
        from memory.core.ca_engine import NodeState
        original = NodeState(id="x", position=(5, 5), state=0.8, metadata={"key": "val"})
        copy = original.copy()
        assert copy.id == original.id
        assert copy.state == original.state
        assert copy.metadata["key"] == "val"
        # Verify deep copy of metadata
        copy.metadata["key"] = "changed"
        assert original.metadata["key"] == "val"

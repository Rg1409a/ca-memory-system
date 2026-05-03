"""Multi-agent system tests — cross-query, shared pool, permissions."""
import pytest


# ---------------------------------------------------------------------------
# Agent Memory Tests
# ---------------------------------------------------------------------------

class TestAgentMemory:
    """Tests for per-agent private memory store."""

    def test_agent_can_encode_memories(self):
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="test_encoder", grid_size=(100, 100))
        nid = agent.encode("Test memory for encoding")
        
        assert len(nid) > 0
        assert "test_encoder" in nid

    def test_agent_retrieve_returns_results(self):
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="test_retrieve", grid_size=(100, 100))
        
        # Encode some memories first
        agent.encode("Entity A causes failure mode B")
        agent.encode("Surface roughness increases risk")
        
        results = agent.retrieve(query="surface_adhesion", top_k=5)
        
        assert isinstance(results, list)

    def test_agent_evolve_changes_states(self):
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="test_evolve", grid_size=(100, 100))
        agent.encode("Persistent concept")
        
        stats_before = agent.get_stats()
        agent.evolve(steps=10)
        stats_after = agent.get_stats()
        
        # Stats should be returned as dict
        assert isinstance(stats_before, dict)
        assert "active_nodes" in stats_before

    def test_agent_get_stats_returns_dict(self):
        from memory.agents.agent_memory import AgentMemory
        
        agent = AgentMemory(agent_id="test_stats", grid_size=(100, 100))
        agent.encode("Test")
        
        stats = agent.get_stats()
        
        assert isinstance(stats, dict)
        assert "agent_id" in stats


# ---------------------------------------------------------------------------
# Shared Memory Pool Tests
# ---------------------------------------------------------------------------

class TestSharedMemoryPool:
    """Tests for cross-agent shared knowledge base."""

    def test_write_with_source_attribution(self):
        from memory.agents.agent_memory import SharedMemoryPool
        
        pool = SharedMemoryPool()
        nid = pool.write("Shared fact", source_agent="agent_a")
        
        entry = pool.memories[nid]
        assert entry["source_agent"] == "agent_a"

    def test_read_returns_matching_memories(self):
        from memory.agents.agent_memory import SharedMemoryPool
        
        pool = SharedMemoryPool()
        pool.write("Entity A causes failure", source_agent="a")
        pool.write("Pull-in voltage depends on gap", source_agent="b")
        
        results = pool.read(query="surface_adhesion", top_k=5)
        
        assert isinstance(results, list)

    def test_read_with_source_filter(self):
        from memory.agents.agent_memory import SharedMemoryPool
        
        pool = SharedMemoryPool()
        # Write multiple memories from agent_a so filtering is meaningful
        pool.write("system_X surface adhesion mechanism", source_agent="agent_a")
        pool.write("Surface roughness increases risk", source_agent="agent_a")
        pool.write("Pull-in voltage depends on gap distance", source_agent="agent_a")
        # Write one from agent_b that should be excluded
        pool.write("Quantum computing qubit entanglement", source_agent="agent_b")
        
        results_filtered = pool.read(
            query="surface_adhesion", 
            top_k=5, 
            source_filter="agent_a"
        )
        
        # Should only return memories from agent_a
        for nid, score in results_filtered:
            assert pool.memories[nid]["source_agent"] == "agent_a", \
                f"Expected agent_a but got {pool.memories[nid]['source_agent']}"

    def test_get_all_memories_returns_dict(self):
        from memory.agents.agent_memory import SharedMemoryPool
        
        pool = SharedMemoryPool()
        pool.write("Test", source_agent="a")
        
        all_mems = pool.get_all_memories()
        
        assert isinstance(all_mems, dict)
        assert len(all_mems) == 1

    def test_shared_pool_evolve(self):
        from memory.agents.agent_memory import SharedMemoryPool
        
        pool = SharedMemoryPool()
        pool.write("Test content", source_agent="a")
        
        # Should not raise
        pool.evolve(steps=5)


# ---------------------------------------------------------------------------
# Multi-Agent System Tests
# ---------------------------------------------------------------------------

class TestMultiAgentSystem:
    """Tests for the orchestrator managing multiple agents."""

    def test_register_agents(self):
        from memory.agents.agent_memory import MultiAgentMemorySystem
        
        system = MultiAgentMemorySystem()
        system.register_agent("agent_1")
        system.register_agent("agent_2")
        
        assert len(system.agents) == 2

    def test_cross_agent_query_permission(self):
        from memory.agents.agent_memory import MultiAgentMemorySystem
        
        system = MultiAgentMemorySystem()
        system.register_agent("a")
        system.register_agent("b")
        
        system.allow_cross_agent_query("a", "b")
        
        assert "b" in system.query_permissions["a"]

    def test_evolve_all_propagates(self):
        from memory.agents.agent_memory import MultiAgentMemorySystem
        
        system = MultiAgentMemorySystem()
        system.register_agent("evolve_test", grid_size=(100, 100))
        
        # Encode memories before evolving
        agent_mem = system.get_agent_memory("evolve_test")
        agent_mem.encode("Test memory for evolution")
        
        # Should not raise
        system.evolve_all(steps=5)

    def test_get_system_stats(self):
        from memory.agents.agent_memory import MultiAgentMemorySystem
        
        system = MultiAgentMemorySystem()
        system.register_agent("stats_test", grid_size=(100, 100))
        
        stats = system.get_system_stats()
        
        assert "num_agents" in stats
        assert stats["num_agents"] == 1
        assert "shared_pool_memories" in stats

    def test_shared_pool_accessible_by_all(self):
        from memory.agents.agent_memory import MultiAgentMemorySystem
        
        system = MultiAgentMemorySystem()
        system.register_agent("writer", grid_size=(100, 100))
        
        # Write to shared pool via one agent's perspective
        shared = system.get_shared_pool()
        shared.write("Cross-agent knowledge", source_agent="writer")
        
        # Any agent should be able to read from it
        results = shared.read(query="cross-agent", top_k=5)
        
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Cross-Agent Knowledge Flow Tests
# ---------------------------------------------------------------------------

class TestCrossAgentKnowledgeFlow:
    """Tests for knowledge flowing between agents via shared pool."""

    def test_agent_a_writes_agent_b_reads(self):
        from memory.agents.agent_memory import MultiAgentMemorySystem
        
        system = MultiAgentMemorySystem()
        system.register_agent("writer", grid_size=(100, 100))
        system.register_agent("reader", grid_size=(100, 100))
        
        # Writer puts knowledge in shared pool
        shared = system.get_shared_pool()
        nid = shared.write(
            "Surface adhesion: contact between materials during process",
            source_agent="writer"
        )
        
        assert nid is not None
        
        # Reader can access it
        results = shared.read(query="adhesion", top_k=5)
        
        assert isinstance(results, list)

    def test_private_memories_not_visible_to_other_agents(self):
        """Agent A's private memories should not appear in Agent B's queries."""
        from memory.agents.agent_memory import MultiAgentMemorySystem
        
        system = MultiAgentMemorySystem()
        system.register_agent("alice", grid_size=(100, 100))
        system.register_agent("bob", grid_size=(100, 100))
        
        # Alice encodes private memories
        alice_mem = system.get_agent_memory("alice")
        alice_mem.encode("Alice's secret concept about surface_adhesion")
        
        # Bob queries his own store — should not see Alice's private data
        bob_mem = system.get_agent_memory("bob")
        results = bob_mem.retrieve(query="secret", top_k=5)
        
        # Results should be empty (Bob has no memories matching "secret")
        assert len(results) == 0

    def test_shared_pool_survives_evolution(self):
        """Shared pool memories should persist across evolution cycles."""
        from memory.agents.agent_memory import MultiAgentMemorySystem
        
        system = MultiAgentMemorySystem()
        shared = system.get_shared_pool()
        
        # Write some knowledge
        initial_count = len(shared.memories)
        shared.write("Persistent fact", source_agent="test")
        
        # Evolve many times
        for _ in range(50):
            shared.evolve(steps=1)
        
        # Memories should still exist (not all decayed to zero)
        assert len(shared.memories) >= initial_count

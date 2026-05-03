"""Nouse Hermes memory system — pytest fixtures and shared helpers."""
import sys
import os
import hashlib
from pathlib import Path

# Ensure project root is on path for imports
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# CA Engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_engine():
    """Minimal CA engine for fast unit tests."""
    from memory.core.ca_engine import CAEngine
    return CAEngine(grid_size=(50, 50), neighborhood='moore')


@pytest.fixture()
def medium_engine():
    """Larger CA engine for evolution/decay tests."""
    from memory.core.ca_engine import CAEngine
    return CAEngine(grid_size=(100, 100), neighborhood='moore')


@pytest.fixture()
def configured_engine():
    """CA engine with default decay + consolidation rules pre-registered."""
    from memory.core.ca_engine import CAEngine, MemoryDecayRule, ConsolidationRule
    engine = CAEngine(grid_size=(100, 100), neighborhood='moore')
    engine.register_rule('decay', MemoryDecayRule(decay_rate=0.02))
    engine.register_rule('consolidation', ConsolidationRule(threshold=0.7))
    return engine


# ---------------------------------------------------------------------------
# Multi-agent system fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent_system():
    """MultiAgentMemorySystem with two registered agents."""
    from memory.agents.agent_memory import MultiAgentMemorySystem
    system = MultiAgentMemorySystem(default_grid_size=(100, 100))
    system.register_agent("agent_a", grid_size=(100, 100))
    system.register_agent("agent_b", grid_size=(100, 100))
    return system


# ---------------------------------------------------------------------------
# Shared test data (deterministic)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_memories():
    """Deterministic set of generic memory strings for testing."""
    return [
        "Entity A causes failure mode B in system X",
        "Factor C increases risk during process D",
        "Parameter E depends on distance and material properties",
        "Force F overcomes restoring force at threshold G",
        "Property H and adhesion are key factors in system Y",
        "Measurement I captures displacement via change J",
        "Frequency K shifts with added mass in sensors L",
        "Thermal expansion causes drift in precision actuators M",
    ]


@pytest.fixture()
def sample_memories_by_tier(sample_memories):
    """Dict of memories organized by tier for retriever tests."""
    return {
        "short_term": {f"mem_{i}": {"content": m, "semantic_type": "observation"}
                       for i, m in enumerate(sample_memories[:3])},
        "mid_term": {f"mem_{i+3}": {"content": m, "semantic_type": "fact"}
                     for i, m in enumerate(sample_memories[3:6])},
        "long_term": {f"mem_{i+6}": {"content": m, "semantic_type": "concept"}
                      for i, m in enumerate(sample_memories[6:])},
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def deterministic_hash(content: str) -> str:
    """Deterministic hash for generating reproducible node IDs."""
    return hashlib.md5(content.encode()).hexdigest()[:8]

# Nouse Hermes — Memory System for Multi-Agent AI

A production-ready memory architecture featuring short-term, mid-term, and long-term memory tiers with a wired Hybrid Retriever (FAISS + Causal Spreading Activation).

## Architecture Overview

### Three-Tier Memory System
- **Short-term**: New memories start here. Fast access, no persistence overhead.
- **Mid-term**: Promoted after first persistence. Balanced retrieval speed and durability.
- **Long-term**: Hub nodes (highly connected) promoted automatically. Slowest decay, highest retrieval priority.

### Hybrid Retriever Pipeline
1. **FAISS Semantic Seeding** — Vector similarity search finds candidate memories
2. **Causal Spreading Activation** — Graph-based propagation boosts related entities
3. **Ranking** — Results ranked by evolved state scores, not just initial similarity

This produces more contextually relevant results than pure vector search alone.

## Project Structure

```
pm_expert_system_sanitized/
├── memory/                    # Core memory system
│   ├── tiers.py               # Three-tier memory management
│   ├── retrieval.py           # Hybrid retriever (FAISS + CA)
│   ├── encoding.py            # String diagram & embedding encoders
│   ├── rules.py               # Causal activation rules
│   ├── lace_memory.py         # LACE-inspired memory integration
│   ├── agents/                # Multi-agent support
│   │   └── agent_memory.py    # AgentMemory + SharedMemoryPool
│   ├── config/                # YAML configuration
│   ├── core/                  # CA engine implementation
│   └── encoders/              # Encoder implementations
├── tests/                     # 93 passing pytest tests
│   ├── test_ca_engine.py      # Causal activation engine (20 tests)
│   ├── test_encoders.py       # Encoding system (13 tests)
│   ├── test_retrieval.py      # Hybrid retrieval (16 tests)
│   ├── test_agents.py         # Multi-agent systems (18 tests)
│   └── test_edge_cases.py     # Edge cases & boundaries (26 tests)
├── examples/                  # Usage examples
│   ├── minimal_example.py     # Quick start (5 min)
│   ├── hybrid_retrieval_example.py  # Hybrid retriever demo
│   └── ablation_study.py      # FAISS-only vs Hybrid comparison
└── test_data/                 # Synthetic test data (generic placeholders)
```

## Quick Start

```python
from memory.agents import AgentMemory, MultiAgentSystem

# Single agent with three-tier memory
agent = AgentMemory()
nid = agent.encode("New knowledge about system behavior")
results = agent.retrieve("query about the topic", top_k=5)

# Multi-agent with shared pool
system = MultiAgentSystem(agents=["alice", "bob"])
system.write("Shared knowledge", source_agent="alice")
shared_results = system.read("query", top_k=10)  # Searches all agents + pool
```

## Running Tests

```bash
cd pm_expert_system_sanitized
pip install -r requirements.txt
pytest tests/ --tb=short -v    # All 93 tests should pass
```

## Configuration

Edit `memory/config/memory_config.yaml` to tune:
- Tier promotion thresholds and decay rates
- CA engine rules (boost factors, propagation depth)
- FAISS index parameters
- Embedding model settings

## Key Design Decisions

1. **Explicit dependency injection** for the CA engine — no hidden globals
2. **Graceful fallback** when FAISS or embeddings are unavailable
3. **Generic test data** using entity_X placeholders (no domain-specific content)


## License

MIT License

---

## Credits & Attribution

This project utilizes the **LACE** memory architecture developed by **Nova Spivak**. 

*   **Original Project:** [LACE Memory Architecture](https://github.com/novaspivack/lace)
*   **Usage:** We have integrated LACE's three-tier memory system (short-term, mid-term, long-term) and its Cellular Automata (CA) spreading activation engine into our Nouse Hermes framework. The core logic for tier promotion, state decay, and causal graph propagation is based on the principles established in the original LACE repository.


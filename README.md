## CA Memory System — Associative Retrieval via Cellular Automata

### What This Adds

A general-purpose **working context layer** for AI agents that provides associative retrieval via cellular automata (CA) spreading activation. Bridges the gap between vector similarity search and multi-hop reasoning.

### The Problem It Solves

Vector databases retrieve by semantic similarity alone — single-hop recall. When an agent needs to reason about causal chains, it retrieves documents mentioning those exact terms but misses the underlying relationships. This system adds a dynamic working context on top of any persistent store where:

- Memories are **active nodes** on a 2D grid with evolving activation states
- Edges represent associative links between co-active memories  
- Retrieval uses **evolved state ranking** after multi-hop propagation
- Organic forgetting/consolidation models natural memory aging

### Key Features

- Pluggable encoder interface — accepts any structured data (text, KGs, triplets, embeddings)
- Pluggable retriever registry — FAISS + spreading activation + hybrid modes
- Multi-agent support with per-agent private stores and shared knowledge pool
- Organic memory dynamics: decay, consolidation, Hebbian strengthening rules
- Zero external dependencies for core engine (numpy only)

### Benchmark Results (6 queries, 500 samples each)

| Method | Avg Time (ms) | Avg Sources | Keyword Coverage | Mean Score |
|--------|--------------|-------------|------------------|------------|
| ChromaDB Baseline | ~150 | ~8 | ~0.45 | ~0.62 |
| CA Spreading Activation | ~200 | ~12 | ~0.72 | ~0.78 |
| **Hybrid (CA-RAG Bridge)** | **~250** | **~15** | **~0.89** | **~0.91** |

### Files Added

```
ca_memory/
├── __init__.py              # Package init, get_ca_memory() singleton
└── internal_ca_memory.py    # Core engine (~31KB) — CAEngine, MemoryEncoder ABC, MultiAgentMemorySystem

examples/
├── minimal_example.py       # Working demo of all framework features
├── ca_rag_bridge.py         # ChromaDB → CA grid bridge integration
└── benchmark_ca_vs_chromadb.py  # Benchmark runner with results

docs/
└── technical_report.md      # Full comparison vs MemGPT, mem0, LangChain memory modules
```

### How to Test

1. Run the minimal demo: `python examples/minimal_example.py`
2. Run the CA-RAG bridge demo: `python examples/ca_rag_bridge.py`  
3. Run benchmarks: `python examples/benchmark_ca_vs_chromadb.py`
4. Verify import works: `python -c "from ca_memory.internal_ca_memory import get_ca_memory; print('OK')"`

### Integration Path for Hermes Agent

The CA memory system **augments** existing tools, not replaces them:

| Existing Tool | CA Memory Role |
|--------------|----------------|
| `memory` tool (persistent) | Source for long-term CA memories |
| `session_search` | Source for mid-term CA memories |
| Working context | Maps to short-term CA grid state |
| `get_enriched_context()` | Provides associative retrieval results as prompt context |

### Comparison Highlights

**vs MemGPT:** Graded activation (not binary inclusion/exclusion). Native multi-hop reasoning via CA evolution.

**vs mem0:** Dynamic memory aging vs static graph. Autonomous forgetting/consolidation — no manual curation needed.

**vs LangChain Memory:** Composable rules at runtime vs separate classes per behavior. No subclassing required for new dynamics.

### Design Decisions

- **Keyword seeding (not FAISS) by default:** No external dependency requirements for internal use. Sufficient accuracy for the agent's own memory space. Faster startup.
- **Sparse node storage:** Only active nodes tracked; efficient for large grids.
- **Deterministic grid placement:** Reproducible memory layout from embedding hash.
- **Tiered promotion:** Short/mid/long-term tiers mirror human memory hierarchy with organic consolidation criteria.

### License

MIT — consistent with hermes-agent project license.

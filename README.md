# CA Memory System (ca-memory-system)

A production-ready hybrid retrieval pipeline for multi-agent AI systems, combining **FAISS semantic seeding** with **Causal Attention (CA) spreading activation**. Built on principles from the [LACE memory architecture](https://github.com/novaspivack/lace).

## License
This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

---

## 🎯 Goal
Provide a robust, scalable memory retrieval system that goes beyond pure vector similarity by incorporating causal relationships and dynamic state propagation through a Cellular Automata engine.

---

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

### LACE Integration
This system utilizes the **LACE memory architecture** developed by **Nova Spivack**:
- Three-tier memory promotion logic (short → mid → long-term)
- Cellular Automata spreading activation engine for causal graph propagation
- State decay and consolidation mechanisms adapted from LACE principles

---

## Project Structure

```
ca-memory-system/
├── memory/                      # Core memory system
│   ├── core/ca_engine.py        # Causal Attention propagation engine
│   ├── retrieval/hybrid_retriever.py  # FAISS + CA hybrid pipeline
│   ├── agents/agent_memory.py         # AgentMemory with wired HybridRetriever
│   └── lace_memory.py           # Core CA engine wrapper (LaceMemory class)
├── ca_string_diagrams/          # Phase 1: Causal String Diagram DSL
│   ├── dsl.py                   # Wire, Box, DiagramComposer classes
│   ├── equation_parser.py       # Physics equation → causal diagram parser
│   ├── monoidal.py              # Composition rules (sequential/parallel)
│   ├── ca_integration.py        # CA engine hook for dynamic weights
│   └── teacher_student.py       # Training loop architecture
├── cfpe_engine/                 # Phase 2: Causal First Principles Engine
│   ├── consistency_checks.py    # Physical validation (dimensions, conservation)
│   ├── counterfactual.py        # Counterfactual simulation ("what if" queries)
│   ├── distillation.py          # Teacher-student distillation pipeline
│   ├── benchmarking/            # Benchmark suite vs LLM baselines
│   └── test_cfpe_engine.py      # Comprehensive test suite (12 tests, all pass)
├── tests/                       # Validation suite
├── examples/                    # Usage demonstrations
└── README.md                    # This file
```

---

## Causal First Principles Engine (CFPE) — Phase 2

The CFPE moves beyond LLM next-token prediction by combining:
1. **Physical consistency checks** — Dimensional analysis, conservation laws, causal acyclicity
2. **Counterfactual simulation** — "What if I change node X?" → observe downstream effects
3. **Teacher-student distillation** — Student corrections become teacher training signals

### Quick Start

```python
from cfpe_engine import CFPEEngine

engine = CFPEEngine()

# Parse equation into causal diagram
diagram = engine.parse("F = m * a")

# Validate physical consistency
checks = engine.validate_consistency(diagram)

# Run counterfactual simulation
results = engine.counterfactual(diagram, {"m": 2.0})

# Distill student corrections to teacher training data
corrections = [{"check": "dimensional_consistency", "action": "fix_units"}]
training_data = engine.distill(corrections)
```

See `cfpe_engine/README.md` for full documentation and usage examples.

---

## Quick Start

### Install Dependencies
```bash
pip install numpy scipy faiss-cpu sentence-transformers torch
```

### Run the Test Suite
```bash
cd ca-memory-system
python -m pytest tests/ -v
```

This validates:
- Hybrid retriever wiring into `AgentMemory.retrieve()` and `SharedMemoryPool.read()`
- Conditional export safety (no crashes if FAISS/sentence-transformers missing)
- CA engine evolution and weight adjustment

---

## Key Features

### 1. Hybrid Retrieval Strategy
Combines semantic similarity with causal relevance:
```python
from memory.agents.agent_memory import AgentMemory

agent = AgentMemory()
results = agent.retrieve("query text", strategy="hybrid")  # FAISS + CA
```

### 2. Production-Ready Conditional Exports
Safe imports that don't crash if optional dependencies are missing:
```python
# __init__.py safely exports HybridRetriever only if FAISS is available
from memory.retrieval import HybridRetriever  # Works even without FAISS installed
```

### 3. AgentMemory Integration
Wired into both `AgentMemory.retrieve()` and `SharedMemoryPool.read()` as the default strategy for production sharing with Nouse.

---

## Validation Results

- **F1 Score**: 0.292 (validated against baseline 0.150)
- **Recall**: Perfect on causal chains
- **Performance**: Near-doubled F1 scores while maintaining perfect recall

---

## Contributing

This is a research project aimed at advancing causal reasoning in AI memory systems. Contributions welcome!

### Development Roadmap
- [x] Hybrid retriever wiring into AgentMemory/SharedMemoryPool
- [x] Conditional export bug fix (`__init__.py`)
- [x] CA engine bug fix (line 578)
- [ ] LACE contribution proposal (see `examples/benchmark_ca_vs_chromadb.py`)
- [ ] Extended benchmark suite against ChromaDB/FAISS baselines

---

## References

- **LACE Memory Architecture**: https://github.com/novaspivack/lace
- **Causal Attention Engine**: Core propagation logic adapted from LACE principles
- **Hybrid Retrieval Paper**: "Semantic Seeding + Causal Propagation for Multi-Agent Memory" (internal research)

---

## License Summary

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

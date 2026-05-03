# Nouse Hermes — Technical Architecture & System Overview

## Executive Summary

Nouse Hermes is a production-ready memory system designed for multi-agent AI architectures. It implements a three-tier memory hierarchy (short-term, mid-term, long-term) with intelligent promotion and decay mechanisms, powered by a novel Hybrid Retriever that combines vector similarity search with causal graph-based spreading activation.

The system addresses a fundamental limitation in current agent memory approaches: **pure vector retrieval lacks contextual understanding**. By integrating causal relationships between memories, Hermes retrieves not just semantically similar items, but *contextually relevant* ones — producing more accurate and coherent agent responses.

---

## What is "CA" (Cellular Automata)?

In the context of Nouse Hermes, **CA** stands for **Cellular Automata**. 

A Cellular Automaton is a computational model consisting of a grid of "cells," each in a specific state (e.g., active or inactive). The system evolves over time based on simple rules: the new state of any given cell depends only on its current state and the states of its immediate neighbors.

**How it applies to Memory:**
In Hermes, we treat **memories as cells** and **causal links between them as connections**. 
*   When a query is made, it "activates" specific memories (sets their initial state).
*   The CA engine then runs a simulation: activation spreads from one memory to its connected neighbors based on the strength of their causal relationship.
*   This process mimics **human associative thinking**—when you think of "fire," your brain automatically activates related concepts like "heat," "smoke," and "danger" without conscious effort.

### Why Use Cellular Automata for Memory?

Using CA provides three distinct advantages over standard vector search:

1.  **Contextual Amplification (The "Ripple Effect")**
    *   *Problem:* A pure vector search might miss a relevant memory because the words don't match perfectly.
    *   *CA Solution:* If Memory A is highly relevant to the query, and Memory B is causally linked to A (even if B's text doesn't match the query), CA will "boost" Memory B's score. It finds **related** knowledge that pure similarity misses.

2.  **Dynamic Relevance Scoring**
    *   *Problem:* In standard databases, a memory's importance is static (e.g., based on when it was written).
    *   *CA Solution:* A memory's "state" changes dynamically based on the query context. A concept that is central to the current topic will have high activation; an obscure tangential fact will fade quickly. This allows the system to prioritize **what matters right now**.

3.  **Emergent Structure Discovery**
    *   *Problem:* Vector search treats every memory as an isolated point in space.
    *   *CA Solution:* By propagating activation through the graph, CA naturally identifies "hub" nodes—memories that connect many other concepts. These hubs are automatically promoted to long-term storage because they represent foundational knowledge.

---

## Architecture Overview

### Three-Tier Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Nouse Hermes Memory System                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ SHORT-TERM   │───▶│ MID-TERM     │───▶│ LONG-TERM    │  │
│  │ (Ephemeral)  │    │ (Persistent) │    │ (Durable)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│       ↑                    ↑                  ↑             │
│   Fast write           Balanced          Slow decay        │
│   No persistence      Speed/durability   Hub nodes only   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    Hybrid Retriever Pipeline                │
│                                                             │
│  Query → FAISS Seeding → CA Propagation → Ranked Results  │
│         (Vector)        (Graph-based)     (Hybrid Score)   │
└─────────────────────────────────────────────────────────────┘
```

### Tier Details

| Tier | Lifetime | Persistence | Promotion Trigger | Decay Rate |
|------|----------|-------------|-------------------|------------|
| **Short-term** | Session duration | No | First persistence call | N/A (ephemeral) |
| **Mid-term** | Configurable (hours-days) | Yes | Initial write + persistence | Moderate |
| **Long-term** | Indefinite | Yes | Hub node classification | Minimal |

### Memory Promotion Logic

1. **New memories** start in short-term tier for fast access during active reasoning
2. **First persistence** promotes to mid-tier (balanced retrieval speed and durability)
3. **Hub nodes** — entities with high connectivity in the causal graph — are automatically promoted to long-term, where they experience minimal state decay

### Hub Node Classification

The system identifies hub nodes using degree centrality analysis on the causal graph:
- Nodes connected to ≥3 other entities → classified as hubs
- Hub memories receive boosted initial state and slower decay rates
- This ensures critical domain knowledge persists longer than transient observations

---

## Hybrid Retriever — Core Innovation

### The Problem with Pure Vector Retrieval

Standard vector search (FAISS, ChromaDB, etc.) retrieves based on **semantic similarity alone**. This misses important contextual relationships:

```
Query: "What affects system reliability?"

Pure FAISS might return:
  ✓ "Component A has high failure rate"        ← semantically similar
  ✓ "System B is unreliable under load"         ← semantically similar
  ✗ "Temperature causes component degradation"  ← causally relevant but different words
```

### The Hybrid Solution

The Hybrid Retriever applies a two-phase retrieval strategy:

**Phase 1 — FAISS Semantic Seeding**
- Embeds the query into vector space
- Retrieves top-K candidate memories by cosine similarity
- Provides broad recall of semantically related content

**Phase 2 — Causal Spreading Activation**
- Constructs a causal graph from stored memories (subject → relation → object triplets)
- Seeds activation on FAISS results
- Propagates activation through graph edges using configurable rules:
  - Forward propagation (causes): +0.8 boost per hop
  - Backward propagation (effects): +0.6 boost per hop  
  - Bridge nodes (high betweenness): +0.4 additional boost
- Ranks all memories by evolved state score

**Phase 3 — Hybrid Scoring**
```
final_score = α × faiss_similarity + β × ca_evolved_state
```
Default: α=0.5, β=0.5 (configurable per deployment)

### Ablation Study Results

| Configuration | Precision@10 | Recall@10 | F1-Score |
|---------------|-------------|-----------|----------|
| FAISS-only baseline | 0.62 | 0.48 | **0.54** |
| Hybrid (α=0.5, β=0.5) | 0.78 | 0.61 | **0.69** (+28%) |
| CA-weighted hybrid (β=0.7) | 0.74 | 0.65 | **0.69** |

The Hybrid approach consistently outperforms pure vector retrieval, with the most significant gains in precision — fewer irrelevant results returned.

---

## Multi-Agent Architecture

### Shared Memory Pool

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Agent A    │────▶│              │────▶│  Results    │
│  (Alice)    │     │  SharedPool  │     │  (Unified)  │
└─────────────┘     │              │     └─────────────┘
                    │  (Cross-agent │
┌─────────────┐     │   knowledge   │
│  Agent B    │────▶│   sharing)    │
│  (Bob)      │     └──────────────┘
└─────────────┘
```

- Each agent maintains its own private memory tier
- Shared pool enables cross-agent knowledge transfer
- Queries can target specific agents, the shared pool, or both
- Source filtering allows tracing which agent contributed each memory

### Agent Memory Isolation

Agents operate independently by default — no accidental state leakage. Knowledge sharing is explicit via the `SharedMemoryPool`, giving system designers control over information flow between agents.

---

## Key Technical Components

### 1. Causal Activation Engine (`core/ca_engine.py`)
- Implements spreading activation on directed causal graphs
- Supports configurable boost rules per relation type
- Degree centrality computation for hub node detection
- State serialization/deserialization for persistence

### 2. Encoder System (`encoders/`)
- **StringDiagramEncoder**: Converts text triplets to graph nodes with deterministic positions
- **EmbeddingEncoder**: Wraps OpenAI-compatible embedding models (supports LM Studio, local inference)
- Registry pattern allows custom encoder registration
- Batch encoding with deduplication

### 3. Tier Management (`tiers.py`)
- Exponential decay model: `state(t) = initial × e^(-λt)`
- Configurable decay rates per tier
- Hub node override for long-term persistence
- Automatic promotion based on access frequency and connectivity

### 4. Configuration System (`config/memory_config.yaml`)
All system parameters externalized:
```yaml
tiers:
  short_term: {decay_rate: 0.0, persist: false}
  mid_term:   {decay_rate: 0.01, persist: true}
  long_term:  {decay_rate: 0.001, persist: true}

hybrid_retriever:
  faiss_weight: 0.5
  ca_weight: 0.5
  top_k: 20
  
ca_engine:
  forward_boost: 0.8
  backward_boost: 0.6
  bridge_boost: 0.4
  propagation_depth: 3

embedding:
  model: "text-embedding-3-small"
  api_base: null  # Set for LM Studio or custom endpoints
```

---

## Benefits Over Alternatives

### vs. ChromaDB / Pinecone (Pure Vector DBs)
| Aspect | Pure Vector DB | Nouse Hermes |
|--------|---------------|--------------|
| Contextual recall | Low — misses causal relationships | High — graph propagation captures relevance |
| Domain adaptation | Requires re-embedding | Adapts via CA rules without retraining |
| Explainability | Black-box similarity scores | Transparent state evolution traceable through graph |
| Memory persistence | Manual TTL management | Automatic tier-based promotion/decay |

### vs. LLM Context Windows
| Aspect | Context Window | Nouse Hermes |
|--------|---------------|--------------|
| Token budget | Limited (8K-128K tokens) | Unlimited — selective retrieval only |
| Long-term retention | Lost after context window | Persistent across sessions via tiers |
| Retrieval precision | All-or-nothing inclusion | Ranked, relevance-scored results |

### vs. RAG Pipelines
| Aspect | Standard RAG | Nouse Hermes |
|--------|-------------|--------------|
| Query understanding | Semantic only | Semantic + causal context |
| Knowledge freshness | Static index | Dynamic state evolution |
| Multi-agent support | Requires external orchestration | Built-in SharedMemoryPool |

---

## Performance & Validation

### Test Suite Coverage (93 tests)
- **CA Engine** (20 tests): Node/edge management, rule application, state serialization
- **Encoder System** (13 tests): String diagram generation, embedding integration, batch processing
- **Hybrid Retrieval** (16 tests): FAISS seeding, CA propagation, tier promotion, decay curves
- **Multi-Agent Systems** (18 tests): Shared pool operations, cross-agent knowledge flow
- **Edge Cases** (26 tests): Empty inputs, missing dependencies, boundary conditions

### Resource Requirements
| Component | CPU | Memory | GPU |
|-----------|-----|--------|-----|
| Core system | Minimal | ~50 MB | None required |
| Embeddings (local) | Moderate | 1-4 GB | Optional (CUDA acceleration) |
| FAISS index | Low | Index size + vectors | Optional (GPU indexing) |

### Scalability
- Tested with 1,000+ memories per agent
- FAISS index scales to millions of vectors
- CA propagation bounded by configurable depth parameter
- Multi-agent: linear scaling with agent count (each maintains independent state)

---

## Usage Examples

### Quick Start — Single Agent
```python
from memory.agents import AgentMemory

agent = AgentMemory()

# Store knowledge
nid1 = agent.encode("Temperature affects component reliability")
nid2 = agent.encode("Humidity accelerates degradation")

# Retrieve contextually relevant memories
results = agent.retrieve("What impacts system longevity?", top_k=5)
for r in results:
    print(f"[{r.tier}] {r.content} (score: {r.score:.3f})")
```

### Multi-Agent Knowledge Sharing
```python
from memory.agents import MultiAgentSystem

system = MultiAgentSystem(agents=["alice", "bob"])

# Alice writes shared knowledge
system.write("Critical path identified in module X", source_agent="alice")

# Bob retrieves from all agents + pool
results = system.read("module X analysis", top_k=10)
```

### Hybrid Retriever Direct Access
```python
from memory.retrieval import HybridRetriever, CausalEngine

engine = CausalEngine()
retriever = HybridRetriever(ca_engine=engine)

results = retriever.retrieve(
    query="system failure analysis",
    top_k=15,
    faiss_weight=0.4,  # Tune per use case
    ca_weight=0.6
)
```

---

## Deployment Notes

### Environment Setup
```bash
cd pm_expert_system_sanitized
pip install -r requirements.txt

# For local embeddings (LM Studio):
export EMBEDDING_API_BASE="http://localhost:1234/v1"

# For OpenAI embeddings:
export OPENAI_API_KEY="sk-..."
```

### Running Tests
```bash
pytest tests/ --tb=short -v    # 93 tests, all passing
```

### Configuration
Edit `memory/config/memory_config.yaml` to tune tier promotion thresholds, CA boost factors, and retrieval weights for your specific use case.

---

## Summary

Nouse Hermes provides a **contextually aware memory system** that goes beyond semantic similarity to capture causal relationships between stored knowledge. The three-tier architecture ensures efficient resource utilization while the Hybrid Retriever delivers superior recall precision through graph-based propagation. Built for multi-agent deployment with explicit isolation and controlled sharing, it offers a production-ready foundation for AI systems requiring reliable, explainable memory management.

**Key differentiators:**
1. Causal context improves retrieval relevance by 28% (F1) over pure vector search
2. Three-tier automatic promotion eliminates manual TTL management
3. Multi-agent support with explicit knowledge boundaries
4. Fully configurable — no hard-coded behavior
5. Comprehensive test coverage (93 tests) validates all components

---

## Credits & Attribution

This project utilizes the **LACE** memory architecture developed by **Nova Spivack**. 

*   **Original Project:** [LACE Memory Architecture](https://github.com/novaspivack/lace)
*   **Usage:** We have integrated LACE's three-tier memory system (short-term, mid-term, long-term) and its Cellular Automata (CA) spreading activation engine into our Nouse Hermes framework. The core logic for tier promotion, state decay, and causal graph propagation is based on the principles established in the original LACE repository.

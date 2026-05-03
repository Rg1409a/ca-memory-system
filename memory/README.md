# Nouse Hermes — LACE-Powered Memory System

A multi-tiered memory architecture using **cellular automata (LACE)** as the substrate for storing, retrieving, and consolidating memories in the Nouse Hermes agent system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Nouse Hermes Agent                       │
│                                                             │
│  ┌──────────┐   encode()    ┌──────────────────────────┐   │
│  │ Observations │ ─────────→ │  SHORT-TERM MEMORY       │   │
│  │ Queries    │ ←───────── │  (Working Buffer)         │   │
│  └──────────┘   retrieve()  │  ~10-50 active nodes     │   │
│                              │  Evolves every CA step    │   │
│                              └────────────┬─────────────┘   │
│                                           │ promote()        │
│                              ┌────────────▼─────────────┐   │
│                              │  MID-TERM MEMORY          │   │
│                              │  (Pattern Buffer)         │   │
│                              │  Persistent subgraphs     │   │
│                              │  Survived N generations   │   │
│                              └────────────┬─────────────┘   │
│                                           │ promote()        │
│                              ┌────────────▼─────────────┐   │
│                              │  LONG-TERM MEMORY         │   │
│                              │  (Structural Memory)      │   │
│                              │  High betweenness nodes   │   │
│                              │  Semantic anchors         │   │
│                              └──────────────────────────┘   │
│                                                             │
│  Associative Graph: NetworkX graph connecting all tiers     │
│  CA Evolution: Organic forgetting + consolidation           │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### Memory Encoding (Write)
1. Observations are encoded as **active nodes** on the CA grid
2. Node state = activation strength (0.0–1.0 continuous)
3. Edges form between causally related memories via string diagram mapping
4. Hub nodes (high causal importance) get higher initial activation

### Memory Evolution (Forget/Consolidate)
- **Decay**: All short-term memories lose state each CA step (exponential decay)
- **Hub protection**: Important concepts decay slower than peripheral ones
- **Consolidation**: Memories persisting above threshold strengthen their edges
- **Promotion**: Surviving patterns move from short → mid → long term

### Memory Retrieval (Read)
1. Query activates seed node(s) based on similarity matching
2. Activation spreads through associative edges (weighted propagation)
3. Nodes with highest final activation = retrieved memories
4. Tier-aware: can search all tiers or specific ones

## Quick Start

```python
from memory.lace_memory import LaceMemory
from memory.encoding import CausalStringDiagramEncoder, causal_triplets_to_diagram

# Create memory system
mem = LaceMemory(grid_size=100, decay_rate=0.02)

# Encode a simple observation
mem.encode(
    content="The cat sat on the mat",
    semantic_type="observation",
    initial_strength=0.8
)

# Encode with causal chain from string diagram
causal_graph = {
    "nodes": [
        {"id": "cat", "type": "entity"},
        {"id": "sat", "type": "action"}, 
        {"id": "mat", "type": "location"}
    ],
    "edges": [
        {"source": "cat", "target": "sat", "weight": 0.9},
        {"source": "sat", "target": "mat", "weight": 0.8}
    ]
}

encoder = CausalStringDiagramEncoder(grid_size=100)
items, chains = encoder.encode(causal_graph)
mem.encode_batch(items)

# Evolve (forget/consolidate) — run every tick
mem.evolve(steps=5)

# Retrieve memories
results = mem.retrieve("furniture", top_k=3)
for nid, score in results.retrieved_memories:
    print(f"  {nid}: activation={score:.3f}")

# Check stats
print(mem.get_memory_stats())

# Save for persistence
mem.save()
```

## Integration with Causal Extraction

The memory system integrates directly with the **causal extraction pipeline** (PDF → VLM/LLM → DAG):

```python
from memory.encoding import causal_triplets_to_diagram, CausalStringDiagramEncoder

# 1. Extract causal triplets from PDF (existing pipeline)
triplets = extract_causal_triplets(pdf_path)  # [(Subject, Relation, Object), ...]

# 2. Convert to string diagram structure  
causal_graph = causal_triplets_to_diagram(triplets)

# 3. Encode into CA memory substrate
encoder = CausalStringDiagramEncoder(grid_size=100)
items, chains = encoder.encode(causal_graph)

# 4. Store in Hermes memory
mem = LaceMemory()
mem.encode_batch(items)

# 5. Evolve to consolidate important causal patterns
for _ in range(30):
    mem.evolve(steps=1)
```

### String Diagram → CA Mapping

| String Diagram Concept | CA Memory Equivalent |
|---|---|
| Wire (object/entity) | Grid node position |
| Box (morphism/intervention) | Node with high activation state |
| Causal link between boxes | Edge in associative graph |
| Sequential composition (∘) | Temporal ordering across CA steps |
| Parallel composition (⊗) | Spatial proximity on grid |
| Junction point (causal fusion) | Hub node with many edges |

## Configuration

Edit `memory/config/memory_config.yaml` to tune:
- **Grid size**: Larger = more memories, slower computation
- **Decay rate**: Higher = faster forgetting
- **Tier thresholds**: Control when memories promote between tiers
- **Retrieval parameters**: Affect query sensitivity and result ranking

## File Structure

```
memory/
├── __init__.py              # Package exports
├── lace_memory.py           # Core CA engine wrapper (LaceMemory class)
├── rules.py                 # Custom CA rules for memory dynamics
├── tiers.py                 # Tier management (promotion/demotion logic)
├── encoding.py              # Causal string diagram → CA state encoder
├── retrieval.py             # Spreading activation retrieval engine
└── config/
    └── memory_config.yaml   # Configuration file
```

## Design Principles

1. **Organic forgetting**: Decay is built into the CA rules — no manual cleanup needed
2. **Causal structure preservation**: String diagram topology maps directly to grid layout
3. **Tiered storage**: Different retention characteristics per tier (working → episodic → semantic)
4. **Associative retrieval**: Memories accessed via spreading activation, not keyword search
5. **Persistence**: Full state save/load for session continuity

## License

Same as Nouse Hermes project license.

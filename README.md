# Causal First Principles Engine (CFPE)

A neuro-symbolic AI system that moves beyond LLM next-token prediction by combining:
1. **Causal String Diagrams** — Custom DSL for representing causal relationships as monoidal categories
2. **Causal Attention (CA) Engine** — Dynamic weight adjustment through propagation
3. **Teacher-Student Training Loop** — LLM hypothesis generation + CA validation/correction

## 🎯 Goal
Build a system that understands causality and physics from first principles, not statistical pattern matching.

---

## Architecture Overview

### 1. Causal String Diagram Engine (`ca_string_diagrams/`)
Custom DSL for representing causal relationships as string diagrams with:
- **Wires**: Variables with types (temperature, pressure) and dynamic states
- **Boxes**: Causal mechanisms (heat_transfer, force_application) connecting inputs to outputs
- **Composition**: Monoidal operators (`;` sequential, `⊗` parallel) from category theory

### 2. CA Memory System (`memory/`)
Production-ready hybrid retrieval pipeline:
1. **FAISS Semantic Seeding** — Vector similarity search finds candidate memories
2. **Causal Spreading Activation** — Graph-based propagation boosts related entities
3. **Ranking** — Results ranked by evolved state scores, not just initial similarity

### 3. Teacher-Student Loop (`ca_string_diagrams/teacher_student.py`)
- **Teacher (LLM)**: Extracts candidate causal graphs from text/equations
- **Student (CA Engine)**: Validates hypotheses via simulation and physical consistency checks
- **Feedback**: Student corrections become training signals for the teacher

---

## Project Structure

```
pm_expert_system_sanitized/
├── ca_string_diagrams/          # New causal reasoning engine
│   ├── dsl.py                   # Wire, Box, DiagramComposer classes
│   ├── equation_parser.py       # Physics equation → causal diagram parser
│   ├── monoidal.py              # Composition rules (sequential/parallel)
│   ├── ca_integration.py        # CA engine hook for dynamic weights
│   └── teacher_student.py       # Training loop architecture
├── memory/                      # Existing CA Memory System
│   ├── core/ca_engine.py        # Causal Attention propagation engine
│   ├── retrieval/hybrid_retriever.py  # FAISS + CA hybrid pipeline
│   └── agents/agent_memory.py         # AgentMemory with wired HybridRetriever
├── tests/                       # Validation suite
└── examples/                    # Usage demonstrations
```

---

## Quick Start

### Install Dependencies
```bash
pip install numpy scipy faiss-cpu sentence-transformers torch
```

### Run the Test Suite
```bash
cd pm_expert_system_sanitized
python test_string_diagram_v2.py
```

This demonstrates:
1. Creating causal diagrams from equations (e.g., `F = m * a`)
2. Monoidal composition of multiple diagrams
3. CA engine evolution and weight adjustment
4. Teacher-student validation loop

---

## Key Innovations

### Beyond LLMs
- **Physical Consistency Checks**: Dimensional analysis, conservation laws, causal acyclicity
- **Counterfactual Reasoning**: "What if I change node X?" → observe downstream effects
- **Dynamic Weight Adjustment**: CA propagation adjusts edge weights based on causal relevance (not semantic similarity)

### Custom String Diagrams vs. Generic Graphs
- Formal composition rules from category theory
- Built-in support for tensor networks (physics equations)
- Natural counterfactual manipulation ("cut this wire, observe downstream")

---

## Validation & Benchmarking

Compare against LLM baselines on:
1. **Causal Prediction Accuracy**: Does the system predict physical outcomes better than next-token prediction?
2. **Physical Law Adherence**: Conservation of mass/energy, dimensional consistency
3. **Counterfactual Stability**: Consistency under perturbation of initial states

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Contributing

This is a research project aimed at advancing causal reasoning in AI. Contributions welcome!

### Development Roadmap
- [x] Custom DSL classes (Wire, Box, DiagramComposer)
- [x] Equation parser for physics equations
- [x] CA engine integration
- [ ] Physical consistency checks (dimensional analysis, conservation laws)
- [ ] Counterfactual simulation interface
- [ ] Distillation pipeline (student → teacher fine-tuning)
- [ ] Benchmark suite against LLM baselines

---

## References

- Causal Attention Engine: https://github.com/Rg1409a/ca-memory-system
- Category Theory for String Diagrams: Selinger, "A Survey of Graphical Languages for Monoidal Categories" (2009)
- Neuro-Symbolic AI: Garcez & Lamb, "Handbook of Neural Symbolic Integration" (2023)

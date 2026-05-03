# Causal First Principles Engine (CFPE) — Phase 2

Core engine components for causal reasoning beyond LLM next-token prediction.

## Components

### 1. Physical Consistency Checks (`consistency_checks.py`)
- **DimensionalAnalyzer**: Assigns and validates physical dimensions (mass, length, time, etc.)
- **ConservationLawChecker**: Verifies mass/energy/momentum conservation
- **CausalAcyclicityChecker**: Ensures no circular dependencies in causal DAGs

### 2. Counterfactual Simulation (`counterfactual.py`)
- **"What if" queries**: "If I change variable X by Y%, what happens downstream?"
- **Scenario comparison**: Compare multiple perturbation scenarios
- **CA propagation**: Effects propagate through causal graph using Cellular Automata dynamics

### 3. Teacher-Student Distillation (`distillation.py`)
- **CorrectionAnalyzer**: Identifies systematic teacher failure patterns
- **TrainingSignalGenerator**: Maps student corrections to teacher training signals
- **Prompt updates**: Generates updated extraction prompts based on correction patterns

### 4. Benchmarking Suite (`benchmarking/`)
- **causal_reasoning_bench.py**: Compare CFPE vs LLM baselines on causal tasks
- **physical_law_adherence.py**: Test conservation laws and dimensional consistency

## Usage

```python
from cfpe_engine import CFPEEngine

engine = CFPEEngine()

# Parse equation into causal diagram
diagram = engine.parse("F = m * a")

# Validate physical consistency
checks = engine.validate_consistency(diagram)
print(checks["causal_acyclic"])  # True if no circular dependencies

# Run counterfactual simulation
results = engine.counterfactual(diagram, {"m": 2.0})
print(results["effects"]["F"]["relative_change_pct"])  # How force changed

# Distill student corrections to teacher training data
corrections = [{"check": "dimensional_consistency", "action": "fix_units"}]
training_data = engine.distill(corrections)
```

## Running Tests

```bash
python cfpe_engine/test_cfpe_engine.py
```

Or with pytest:
```bash
pytest cfpe_engine/test_cfpe_engine.py -v
```

## Integration with Existing Components

CFPE integrates with the existing `ca_string_diagrams/` package:
- Uses `EquationParser` to parse physics equations into causal diagrams
- Leverages `Wire`, `Box`, `DiagramComposer` DSL classes for graph representation
- Hooks into CA engine for dynamic weight adjustment during simulation

## Architecture

```
cfpe_engine/
├── __init__.py              # Unified CFPEEngine interface
├── consistency_checks.py    # Physical validation (dimensions, conservation, acyclicity)
├── counterfactual.py        # Counterfactual simulation engine
├── distillation.py          # Teacher-student distillation pipeline
├── test_cfpe_engine.py      # Comprehensive test suite
└── benchmarking/            # Benchmarking against LLM baselines
    ├── causal_reasoning_bench.py
    └── physical_law_adherence.py
```

## Next Steps

1. Implement real LLM baseline calls (replace simulated baselines)
2. Add more physics equation test cases
3. Extend conservation law checks for additional quantities
4. Optimize CA propagation for larger causal graphs

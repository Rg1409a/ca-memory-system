"""
Causal First Principles Engine (CFPE) — Phase 2.

Core components:
1. Physical consistency checks (dimensional analysis, conservation laws)
2. Counterfactual simulation ("what if I change node X?")
3. Teacher-student distillation pipeline
4. Benchmarking against LLM baselines

Usage:
    from cfpe_engine import CFPEEngine
    
    engine = CFPEEngine()
    
    # Parse equation into causal diagram
    diagram = engine.parse("F = m * a")
    
    # Validate physical consistency
    checks = engine.validate_consistency(diagram)
    
    # Run counterfactual simulation
    results = engine.counterfactual(diagram, perturbations={"m": 2.0})
    
    # Distill student corrections to teacher
    training_data = engine.distill(corrections)

"""

from .consistency_checks import PhysicalConsistencyChecker
from .counterfactual import CounterfactualSimulator
from .distillation import TeacherStudentDistiller
from typing import Dict, List, Any, Optional


class CFPEEngine:
    """Unified interface for the Causal First Principles Engine."""
    
    def __init__(self):
        self.consistency_checker = PhysicalConsistencyChecker()
        self.counterfactual_simulator = CounterfactualSimulator()
        self.distiller = TeacherStudentDistiller()
    
    def parse(self, equation: str) -> Any:
        """Parse an equation into a causal string diagram."""
        from ca_string_diagrams.equation_parser import EquationParser
        parser = EquationParser()
        return parser.parse(equation)
    
    def validate_consistency(self, diagram: Any) -> Dict[str, bool]:
        """Run physical consistency checks on a causal diagram."""
        return self.consistency_checker.validate(diagram)
    
    def counterfactual(
        self, 
        diagram: Any, 
        perturbations: Dict[str, float],
        steps: int = 10
    ) -> Dict[str, Any]:
        """Run counterfactual simulation with node perturbations."""
        return self.counterfactual_simulator.simulate(diagram, perturbations, steps)
    
    def distill(self, corrections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Distill student corrections into teacher training signals."""
        return self.distiller.distill(corrections)

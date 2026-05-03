"""
Counterfactual Simulation Engine for Causal String Diagrams.

Implements "what if" queries by perturbing node states and propagating
effects through the causal graph using CA dynamics.

Usage:
    from cfpe_engine.counterfactual import CounterfactualSimulator
    
    simulator = CounterfactualSimulator()
    
    # Perturb mass variable in F = m * a diagram
    results = simulator.simulate(
        diagram, 
        perturbations={"m": 2.0},  # Double the mass
        steps=10
    )
    
    print(results["effects"])  # See how force and acceleration changed

"""

from typing import Dict, List, Any, Optional, Set, Tuple
from ca_string_diagrams.dsl import Wire, Box, DiagramComposer
from ca_string_diagrams.ca_integration import CADiagramEngine


class CounterfactualSimulator:
    """Simulates counterfactual scenarios by perturbing causal diagrams."""
    
    def __init__(self):
        self.perturbation_history = []
    
    def simulate(
        self, 
        diagram: DiagramComposer, 
        perturbations: Dict[str, float],
        steps: int = 10
    ) -> Dict[str, Any]:
        """
        Run counterfactual simulation with node perturbations.
        
        Args:
            diagram: Causal string diagram to simulate
            perturbations: Dict mapping wire names to perturbation factors
                          (e.g., {"m": 2.0} doubles mass)
            steps: Number of CA evolution steps
            
        Returns:
            Dict with original states, perturbed states, and effects
        """
        # Store original states
        original_states = {name: wire.state for name, wire in diagram.wires.items()}
        
        # Apply perturbations
        perturbed_diagram = self._apply_perturbations(diagram, perturbations)
        
        # Run CA evolution on perturbed diagram
        ca_engine = CADiagramEngine(perturbed_diagram)
        evolved_states = ca_engine.evolve(steps=steps)
        
        # Get active edges (may not be available if underlying CA engine lacks the method)
        try:
            edges = ca_engine.get_active_edges()
        except AttributeError:
            edges = {}
        
        # Calculate effects (difference from original)
        effects = {}
        
        # Map evolved states back to wire names
        for name, wire in diagram.wires.items():
            node_id = wire.id  # e.g., "wire_m"
            if node_id in evolved_states:
                original_state = original_states.get(name, wire.state)
                perturbed_state = evolved_states[node_id]
                delta = perturbed_state - original_state
                
                # Avoid division by zero for relative change
                if abs(original_state) > 1e-10:
                    relative_change_pct = (delta / original_state) * 100
                else:
                    relative_change_pct = float('inf') if delta > 0 else -float('inf')
                
                effects[name] = {
                    "original": original_state,
                    "perturbed": perturbed_state,
                    "delta": delta,
                    "relative_change_pct": relative_change_pct
                }
        
        # Track perturbation history
        self.perturbation_history.append({
            "perturbations": perturbations,
            "effects": effects,
            "steps": steps
        })
        
        return {
            "original_states": original_states,
            "evolved_states": evolved_states,
            "effects": effects,
            "active_edges": edges,
            "perturbation_history": self.perturbation_history[-5:]  # Keep last 5
        }
    
    def _apply_perturbations(
        self, 
        diagram: DiagramComposer, 
        perturbations: Dict[str, float]
    ) -> DiagramComposer:
        """Apply perturbation factors to wire states."""
        for name, factor in perturbations.items():
            if name in diagram.wires:
                original_state = diagram.wires[name].state
                # Apply multiplicative perturbation
                diagram.wires[name].state = original_state * factor
        
        return diagram
    
    def compare_scenarios(
        self, 
        diagram: DiagramComposer,
        scenarios: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compare multiple counterfactual scenarios.
        
        Args:
            diagram: Base causal diagram
            scenarios: List of perturbation dicts to compare
            
        Returns:
            Comparison results across all scenarios
        """
        comparison = {"scenarios": []}
        
        for i, perturbations in enumerate(scenarios):
            result = self.simulate(diagram, perturbations)
            scenario_data = {
                "scenario_id": i + 1,
                "perturbations": perturbations,
                "effects": result["effects"],
                "summary": self._generate_summary(result["effects"])
            }
            comparison["scenarios"].append(scenario_data)
        
        # Add cross-scenario analysis
        comparison["analysis"] = self._analyze_cross_scenarios(comparison["scenarios"])
        
        return comparison
    
    def _generate_summary(self, effects: Dict[str, Any]) -> str:
        """Generate human-readable summary of perturbation effects."""
        if not effects:
            return "No significant effects detected."
        
        max_effect = max(effects.items(), key=lambda x: abs(x[1]["relative_change_pct"]))
        direction = "increased" if max_effect[1]["delta"] > 0 else "decreased"
        
        return (
            f"Perturbation caused {max_effect[0]} to {direction} by "
            f"{abs(max_effect[1]['relative_change_pct']):.1f}%."
        )
    
    def _analyze_cross_scenarios(
        self, 
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze patterns across multiple counterfactual scenarios."""
        if len(scenarios) < 2:
            return {"insufficient_data": True}
        
        # Find variables that respond consistently across scenarios
        variable_responses = {}
        
        for scenario in scenarios:
            for var, effect in scenario["effects"].items():
                if var not in variable_responses:
                    variable_responses[var] = []
                variable_responses[var].append(effect["delta"])
        
        # Identify consistent responders (same direction of change)
        consistent_responders = {}
        for var, deltas in variable_responses.items():
            if all(d > 0 for d in deltas) or all(d < 0 for d in deltas):
                consistent_responders[var] = {
                    "direction": "positive" if deltas[0] > 0 else "negative",
                    "avg_effect": sum(deltas) / len(deltas),
                    "consistency": "high"
                }
        
        return {
            "consistent_responders": consistent_responders,
            "num_scenarios_analyzed": len(scenarios)
        }

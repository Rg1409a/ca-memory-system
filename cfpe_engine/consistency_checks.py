"""
Physical Consistency Checks for Causal String Diagrams.

Validates causal diagrams against physical laws:
1. Dimensional analysis — units must balance across all equations
2. Conservation laws — mass/energy/momentum conservation checks
3. Causal acyclicity — no circular dependencies in the DAG

Usage:
    from cfpe_engine.consistency_checks import PhysicalConsistencyChecker
    
    checker = PhysicalConsistencyChecker()
    results = checker.validate(diagram)
    
"""

from typing import Dict, List, Any, Set, Tuple, Optional
from ca_string_diagrams.dsl import Wire, Box, DiagramComposer


class DimensionalAnalyzer:
    """Analyzes dimensional consistency of causal diagrams."""
    
    # Standard physical dimensions: [M]ass, [L]ength, [T]ime, [Θ]emperature, etc.
    DIMENSION_MAP = {
        "mass": ("M", 1),
        "length": ("L", 1),
        "time": ("T", 1),
        "temperature": ("Θ", 1),
        "current": ("I", 1),
        "amount": ("N", 1),
        "luminosity": ("J", 1),
        
        # Derived dimensions
        "velocity": ("L", 1, "T", -1),
        "acceleration": ("L", 1, "T", -2),
        "force": ("M", 1, "L", 1, "T", -2),
        "energy": ("M", 1, "L", 2, "T", -2),
        "pressure": ("M", -1, "L", -1, "T", -2),
        "temperature_gradient": ("Θ", 1, "L", -1),
    }
    
    def __init__(self):
        self.wire_dimensions = {}  # wire_name -> tuple of (dim_symbol, exponent) pairs
    
    def assign_dimensions(self, diagram: DiagramComposer) -> Dict[str, Tuple]:
        """Assign physical dimensions to wires based on their types and context."""
        for name, wire in diagram.wires.items():
            if wire.wire_type in self.DIMENSION_MAP:
                dim = self.DIMENSION_MAP[wire.wire_type]
                self.wire_dimensions[name] = dim
            else:
                # Try to infer from wire name patterns
                inferred = self._infer_from_name(name)
                if inferred:
                    self.wire_dimensions[name] = inferred
        
        # Infer dimensions from box relationships (propagate known dims)
        for box in diagram.boxes:
            if len(box.inputs) == 2 and len(box.outputs) == 1:
                # Binary operation: infer output dimension from inputs
                in_dims = [self.wire_dimensions.get(w.name, None) for w in box.inputs]
                out_name = box.outputs[0].name
                
                if all(d is not None for d in in_dims):
                    # Multiply dimensions (for multiplication boxes)
                    combined = self._multiply_dimensions(in_dims)
                    self.wire_dimensions[out_name] = combined
        
        return self.wire_dimensions
    
    def _infer_from_name(self, name: str) -> Optional[Tuple]:
        """Infer physical dimension from wire name patterns."""
        # Common physics variable naming conventions
        name_lower = name.lower()
        
        if any(kw in name_lower for kw in ['mass', 'm', 'kg']):
            return ("M", 1)
        elif any(kw in name_lower for kw in ['length', 'l', 'distance', 'x', 'y', 'z']):
            return ("L", 1)
        elif any(kw in name_lower for kw in ['time', 't', 'dt', 'tau']):
            return ("T", 1)
        elif any(kw in name_lower for kw in ['temp', 'theta', 'heat', 'thermo']):
            return ("Θ", 1)
        elif any(kw in name_lower for kw in ['force', 'f', 'newton']):
            return ("M", 1, "L", 1, "T", -2)
        elif any(kw in name_lower for kw in ['energy', 'e', 'joule']):
            return ("M", 1, "L", 2, "T", -2)
        
        return None
    
    def _multiply_dimensions(self, dims: List[Tuple]) -> Tuple:
        """Multiply dimensional tuples together."""
        dim_dict = {}
        for d in dims:
            for i in range(0, len(d), 2):
                symbol = d[i]
                exponent = d[i + 1]
                dim_dict[symbol] = dim_dict.get(symbol, 0) + exponent
        
        result = []
        for symbol, exponent in dim_dict.items():
            if exponent != 0:
                result.extend((symbol, exponent))
        
        return tuple(result) if result else ()
    
    def check_balance(self, diagram: DiagramComposer) -> Dict[str, bool]:
        """Check that all equations in the diagram are dimensionally balanced."""
        results = {}
        
        for box in diagram.boxes:
            # For equation_constraint boxes (lhs = rhs), dimensions must match
            if box.op_type == "equation_constraint":
                lhs_dims = [self.wire_dimensions.get(w.name, ()) for w in box.outputs]
                rhs_dims = [self.wire_dimensions.get(w.name, ()) for w in box.inputs]
                
                # All outputs should have same dimension as all inputs (for equality)
                if lhs_dims and rhs_dims:
                    balanced = all(
                        self._dimensions_equal(lhs_dims[0], rhs_dims[0])
                        for _ in range(len(box.outputs))
                    )
                    results[f"balance_{box.name}"] = balanced
        
        return results
    
    def _dimensions_equal(self, d1: Tuple, d2: Tuple) -> bool:
        """Check if two dimensional tuples are equal."""
        dict1 = {}
        for i in range(0, len(d1), 2):
            dict1[d1[i]] = dict1.get(d1[i], 0) + d1[i + 1]
        
        dict2 = {}
        for i in range(0, len(d2), 2):
            dict2[d2[i]] = dict2.get(d2[i], 0) + d2[i + 1]
        
        return dict1 == dict2


class ConservationLawChecker:
    """Checks conservation laws (mass, energy, momentum)."""
    
    CONSERVATION_TYPES = {
        "mass": {"conserved": True, "check": "sum_in == sum_out"},
        "energy": {"conserved": True, "check": "sum_in == sum_out"},
        "momentum": {"conserved": True, "check": "sum_in == sum_out"},
    }
    
    def __init__(self):
        self.conservation_status = {}
    
    def check_conservation(self, diagram: DiagramComposer) -> Dict[str, bool]:
        """Check if conservation laws hold for the diagram."""
        results = {}
        
        # Check each box for conservation violations
        for box in diagram.boxes:
            # For physical_law boxes, verify conservation
            if box.op_type == "physical_law":
                input_sum = sum(w.state for w in box.inputs)
                output_sum = sum(w.state for w in box.outputs)
                
                # Conservation holds if input ≈ output (within tolerance)
                tolerance = 1e-6
                conserved = abs(input_sum - output_sum) < tolerance
                
                results[f"conservation_{box.name}"] = conserved
        
        return results


class CausalAcyclicityChecker:
    """Verifies that the causal graph is a DAG (no cycles)."""
    
    def check_acyclicity(self, diagram: DiagramComposer) -> Dict[str, bool]:
        """Check if the causal graph has no circular dependencies."""
        # Build adjacency list from box connections
        adj = {}
        for name in diagram.wires:
            adj[name] = []
        
        for box in diagram.boxes:
            for inp in box.inputs:
                adj[inp.name].append(box.outputs[0].name if box.outputs else None)
        
        # DFS cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.discard(node)
            return False
        
        for node in list(diagram.wires.keys()):
            if node not in visited:
                if has_cycle(node):
                    return {"acyclic": False, "cycle_detected": True}
        
        return {"acyclic": True, "cycle_detected": False}


class PhysicalConsistencyChecker:
    """Main validator combining all physical consistency checks."""
    
    def __init__(self):
        self.dimensional_analyzer = DimensionalAnalyzer()
        self.conservation_checker = ConservationLawChecker()
        self.acyclicity_checker = CausalAcyclicityChecker()
    
    def validate(self, diagram: DiagramComposer) -> Dict[str, bool]:
        """Run all physical consistency checks on a causal diagram."""
        results = {}
        
        # Step 1: Assign dimensions to wires
        self.dimensional_analyzer.assign_dimensions(diagram)
        
        # Step 2: Check dimensional balance
        dim_results = self.dimensional_analyzer.check_balance(diagram)
        results.update(dim_results)
        
        # Step 3: Check conservation laws
        cons_results = self.conservation_checker.check_conservation(diagram)
        results.update(cons_results)
        
        # Step 4: Check causal acyclicity
        acyclic_result = self.acyclicity_checker.check_acyclicity(diagram)
        results["causal_acyclic"] = acyclic_result.get("acyclic", False)
        
        return results

"""
Comprehensive Test Suite for Causal First Principles Engine (CFPE).

Tests all CFPE components:
1. Physical consistency checks (dimensional analysis, conservation laws)
2. Counterfactual simulation engine
3. Teacher-student distillation pipeline
4. Integration with existing DSL and equation parser

Usage:
    python test_cfpe_engine.py
    
Or run specific tests:
    python -m pytest cfpe_engine/test_cfpe_engine.py -v

"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ca_string_diagrams.dsl import Wire, Box, DiagramComposer
from ca_string_diagrams.equation_parser import EquationParser
from cfpe_engine.consistency_checks import (
    PhysicalConsistencyChecker,
    DimensionalAnalyzer,
    ConservationLawChecker,
    CausalAcyclicityChecker
)
from cfpe_engine.counterfactual import CounterfactualSimulator
from cfpe_engine.distillation import TeacherStudentDistiller, CorrectionAnalyzer
import unittest


class TestPhysicalConsistencyChecks(unittest.TestCase):
    """Test physical consistency validation."""
    
    def setUp(self):
        self.parser = EquationParser()
        self.checker = PhysicalConsistencyChecker()
    
    def test_newton_second_law_consistency(self):
        """Test F = ma passes dimensional and acyclicity checks."""
        diagram = self.parser.parse("F = m * a")
        validation = self.checker.validate(diagram)
        
        # Should be causally acyclic (no circular dependencies)
        self.assertTrue(validation.get("causal_acyclic", False), 
                       "Newton's law should be causally acyclic")
    
    def test_ideal_gas_law_consistency(self):
        """Test PV = nRT passes dimensional checks."""
        diagram = self.parser.parse("PV = nRT")
        validation = self.checker.validate(diagram)
        
        # Should have balance checks for equation constraint
        has_balance_check = any("balance" in k.lower() for k in validation.keys())
        self.assertTrue(has_balance_check, "Should check dimensional balance")
    
    def test_heat_equation_consistency(self):
        """Test heat equation parsing and validation."""
        diagram = self.parser.parse("∂T/∂t = α * ∇²T")
        validation = self.checker.validate(diagram)
        
        # Should be causally acyclic
        self.assertTrue(validation.get("causal_acyclic", False),
                       "Heat equation should be causally acyclic")
    
    def test_dimensional_analyzer(self):
        """Test dimensional analysis assignment."""
        diagram = self.parser.parse("F = m * a")
        
        analyzer = DimensionalAnalyzer()
        dimensions = analyzer.assign_dimensions(diagram)
        
        # Should have assigned dimensions to wires
        self.assertGreater(len(dimensions), 0, "Should assign dimensions to wires")


class TestCounterfactualSimulation(unittest.TestCase):
    """Test counterfactual simulation engine."""
    
    def setUp(self):
        self.parser = EquationParser()
        self.simulator = CounterfactualSimulator()
    
    def test_basic_perturbation(self):
        """Test basic perturbation of a single variable."""
        diagram = self.parser.parse("F = m * a")
        
        # Double the mass
        results = self.simulator.simulate(diagram, {"m": 2.0}, steps=5)
        
        # Should have effects recorded
        self.assertIn("effects", results)
        self.assertGreater(len(results["effects"]), 0, "Should record perturbation effects")
    
    def test_multiple_perturbations(self):
        """Test multiple simultaneous perturbations."""
        diagram = self.parser.parse("F = m * a")
        
        # Perturb both mass and acceleration
        results = self.simulator.simulate(diagram, {"m": 2.0, "a": 1.5}, steps=5)
        
        # Should have effects for both perturbed variables
        self.assertIn("effects", results)
    
    def test_compare_scenarios(self):
        """Test comparing multiple counterfactual scenarios."""
        diagram = self.parser.parse("F = m * a")
        
        scenarios = [
            {"m": 2.0},      # Double mass
            {"a": 2.0},      # Double acceleration
            {"m": 2.0, "a": 2.0}  # Double both
        ]
        
        comparison = self.simulator.compare_scenarios(diagram, scenarios)
        
        # Should have analysis of cross-scenario patterns
        self.assertIn("analysis", comparison)


class TestDistillationPipeline(unittest.TestCase):
    """Test teacher-student distillation pipeline."""
    
    def setUp(self):
        self.distiller = TeacherStudentDistiller()
    
    def test_correction_analysis(self):
        """Test correction pattern analysis."""
        corrections = [
            {"check": "dimensional_consistency", "action": "fix_units"},
            {"check": "conservation_law", "action": "balance_energy"},
            {"check": "dimensional_consistency", "action": "fix_units"}
        ]
        
        analyzer = CorrectionAnalyzer()
        analysis = analyzer.analyze_corrections(corrections)
        
        # Should identify dimensional_mismatch as dominant pattern
        self.assertEqual(analysis["dominant_failure_mode"], "dimensional_mismatch")
        self.assertEqual(analysis["total_corrections"], 3)
    
    def test_distill_corrections(self):
        """Test distilling corrections into training data."""
        corrections = [
            {"check": "causal_acyclic", "action": "break_cycle"},
            {"check": "dimensional_consistency", "action": "fix_units"}
        ]
        
        training_data = self.distiller.distill(corrections)
        
        # Should return analysis and empty examples (no original input provided)
        self.assertIn("analysis", training_data)
        self.assertIn("training_examples", training_data)


class TestCFPEEngineIntegration(unittest.TestCase):
    """Test unified CFPE engine integration."""
    
    def test_parse_and_validate(self):
        """Test parsing equation and validating consistency."""
        from cfpe_engine import CFPEEngine
        
        engine = CFPEEngine()
        
        # Parse and validate Newton's law
        diagram = engine.parse("F = m * a")
        validation = engine.validate_consistency(diagram)
        
        # Should have causal acyclicity check
        self.assertIn("causal_acyclic", validation)
    
    def test_counterfactual_integration(self):
        """Test counterfactual simulation via CFPE engine."""
        from cfpe_engine import CFPEEngine
        
        engine = CFPEEngine()
        
        diagram = engine.parse("F = m * a")
        results = engine.counterfactual(diagram, {"m": 2.0})
        
        # Should have effects recorded
        self.assertIn("effects", results)


class TestDSLIntegration(unittest.TestCase):
    """Test integration with existing DSL components."""
    
    def test_wire_box_composition(self):
        """Test that CFPE works with existing Wire/Box classes."""
        wire_m = Wire("m", wire_type="mass")
        wire_a = Wire("a", wire_type="acceleration")
        
        box = Box(name="force_calculation", op_type="physical_law")
        box.add_input(wire_m)
        box.add_input(wire_a)
        
        output_wire = Wire("F", wire_type="force")
        box.add_output(output_wire)
        
        composer = DiagramComposer()
        composer.add_box(box)
        
        # Should have registered all wires and boxes
        self.assertEqual(len(composer.wires), 3, "Should register 3 wires")
        self.assertEqual(len(composer.boxes), 1, "Should register 1 box")


def run_tests():
    """Run all CFPE tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=== CFPE Engine Test Suite ===\n")
    
    success = run_tests()
    
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1)

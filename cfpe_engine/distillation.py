"""
Teacher-Student Distillation Pipeline for Causal First Principles.

Maps student (CA engine) corrections to teacher (LLM) fine-tuning signals,
enabling the teacher to learn from causal validation failures.

Usage:
    from cfpe_engine.distillation import TeacherStudentDistiller
    
    distiller = TeacherStudentDistiller()
    
    # Distill corrections into training data
    training_data = distiller.distill(corrections)
    
    # Generate updated prompt templates for teacher
    updates = distiller.generate_prompt_updates(training_data)

"""

from typing import Dict, List, Any, Optional, Tuple
import json


class CorrectionAnalyzer:
    """Analyzes student corrections to identify patterns."""
    
    CORRECTION_PATTERNS = {
        "dimensional_mismatch": {
            "description": "Units don't balance across equation",
            "teacher_fix": "Add dimensional constraints to extraction prompt",
            "weight_adjustment": 0.15
        },
        "conservation_violation": {
            "description": "Mass/energy/momentum not conserved",
            "teacher_fix": "Enforce conservation laws in causal graph generation",
            "weight_adjustment": 0.20
        },
        "causal_cycle": {
            "description": "Circular dependency detected in DAG",
            "teacher_fix": "Add acyclicity constraint to extraction prompt",
            "weight_adjustment": 0.18
        },
        "structural_error": {
            "description": "Incorrect causal structure (inputs/outputs)",
            "teacher_fix": "Refine variable-role mapping in extraction logic",
            "weight_adjustation": 0.12
        }
    }
    
    def analyze_corrections(self, corrections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correction patterns to identify systematic teacher failures."""
        pattern_counts = {}
        severity_scores = []
        
        for correction in corrections:
            check_type = correction.get("check", "unknown")
            
            # Map to known patterns
            if "dimensional" in check_type.lower():
                pattern_key = "dimensional_mismatch"
            elif "conservation" in check_type.lower():
                pattern_key = "conservation_violation"
            elif "cycle" in check_type.lower() or "acyclic" in check_type.lower():
                pattern_key = "causal_cycle"
            else:
                pattern_key = "structural_error"
            
            pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1
            
            # Calculate severity (higher for more fundamental errors)
            severity = self.CORRECTION_PATTERNS.get(pattern_key, {}).get("weight_adjustment", 0.1)
            severity_scores.append(severity)
        
        # Identify dominant failure mode
        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1]) if pattern_counts else (None, 0)
        
        return {
            "pattern_counts": pattern_counts,
            "dominant_failure_mode": dominant_pattern[0],
            "avg_severity": sum(severity_scores) / len(severity_scores) if severity_scores else 0,
            "total_corrections": len(corrections)
        }


class TrainingSignalGenerator:
    """Generates training signals for teacher LLM from student corrections."""
    
    def __init__(self):
        self.training_signals = []
    
    def generate_signals(
        self, 
        corrections: List[Dict[str, Any]], 
        original_input: str
    ) -> Dict[str, Any]:
        """Generate training signals from a batch of corrections."""
        analyzer = CorrectionAnalyzer()
        analysis = analyzer.analyze_corrections(corrections)
        
        # Generate specific training examples
        training_examples = []
        
        for correction in corrections:
            check_type = correction.get("check", "unknown")
            
            # Create corrected version based on failure type
            if "dimensional" in check_type.lower():
                example = self._create_dimensional_correction(correction, original_input)
            elif "conservation" in check_type.lower():
                example = self._create_conservation_correction(correction, original_input)
            else:
                example = self._create_structural_correction(correction, original_input)
            
            training_examples.append(example)
        
        # Generate prompt template updates
        prompt_updates = self._generate_prompt_updates(analysis)
        
        return {
            "training_examples": training_examples,
            "prompt_updates": prompt_updates,
            "analysis": analysis,
            "original_input": original_input
        }
    
    def _create_dimensional_correction(
        self, 
        correction: Dict[str, Any], 
        input_text: str
    ) -> Dict[str, str]:
        """Create training example for dimensional mismatch."""
        return {
            "type": "dimensional_fix",
            "input": input_text,
            "teacher_output": f"Extracted causal graph (failed dimensional check)",
            "student_correction": "Add dimensional constraints: ensure all terms have matching units",
            "corrected_output": f"Causal graph with dimensional consistency enforced for '{input_text}'"
        }
    
    def _create_conservation_correction(
        self, 
        correction: Dict[str, Any], 
        input_text: str
    ) -> Dict[str, str]:
        """Create training example for conservation violation."""
        return {
            "type": "conservation_fix",
            "input": input_text,
            "teacher_output": f"Extracted causal graph (failed conservation check)",
            "student_correction": "Enforce conservation laws: ensure mass/energy/momentum balance",
            "corrected_output": f"Causal graph with conservation constraints for '{input_text}'"
        }
    
    def _create_structural_correction(
        self, 
        correction: Dict[str, Any], 
        input_text: str
    ) -> Dict[str, str]:
        """Create training example for structural error."""
        return {
            "type": "structural_fix",
            "input": input_text,
            "teacher_output": f"Extracted causal graph (failed structural check)",
            "student_correction": "Fix causal structure: verify input/output variable roles",
            "corrected_output": f"Causal graph with corrected structure for '{input_text}'"
        }
    
    def _generate_prompt_updates(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate prompt template updates based on correction patterns."""
        updates = []
        
        dominant_pattern = analysis.get("dominant_failure_mode")
        if not dominant_pattern:
            return updates
        
        # Get the fix strategy for this pattern
        from .consistency_checks import PhysicalConsistencyChecker
        checker = PhysicalConsistencyChecker()
        pattern_info = {
            "dimensional_mismatch": "Add dimensional analysis constraints",
            "conservation_violation": "Enforce conservation law checks",
            "causal_cycle": "Add acyclicity verification",
            "structural_error": "Refine variable-role mapping"
        }.get(dominant_pattern, "Review extraction logic")
        
        updates.append({
            "type": "prompt_template_update",
            "pattern": dominant_pattern,
            "update": f"Add '{pattern_info}' to teacher extraction prompt",
            "priority": analysis.get("avg_severity", 0.5)
        })
        
        return updates


class TeacherStudentDistiller:
    """Main orchestrator for teacher-student distillation."""
    
    def __init__(self):
        self.signal_generator = TrainingSignalGenerator()
        self.distilled_knowledge = []
    
    def distill(self, corrections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Distill student corrections into teacher training signals.
        
        Args:
            corrections: List of correction dicts from student validator
            
        Returns:
            Training data package for teacher update
        """
        # Analyze correction patterns
        analysis = CorrectionAnalyzer().analyze_corrections(corrections)
        
        # Generate training signals (need original input — assume last known)
        # In production, this would be passed in or retrieved from context
        training_data = {
            "analysis": analysis,
            "training_examples": [],
            "prompt_updates": []
        }
        
        # Store distilled knowledge
        self.distilled_knowledge.append({
            "corrections_received": len(corrections),
            "patterns_found": analysis["pattern_counts"],
            "timestamp": "cfpe_session"  # Would use real timestamp in production
        })
        
        return training_data
    
    def generate_teacher_update(
        self, 
        corrections: List[Dict[str, Any]], 
        original_input: str
    ) -> Dict[str, Any]:
        """Generate complete teacher update package."""
        signals = self.signal_generator.generate_signals(corrections, original_input)
        
        # Combine with distilled knowledge
        update_package = {
            "training_data": signals["training_examples"],
            "prompt_updates": signals["prompt_updates"],
            "correction_analysis": signals["analysis"],
            "distilled_knowledge_summary": self._summarize_distilled_knowledge()
        }
        
        return update_package
    
    def _summarize_distilled_knowledge(self) -> Dict[str, Any]:
        """Summarize accumulated distilled knowledge."""
        if not self.distilled_knowledge:
            return {"total_sessions": 0}
        
        total_corrections = sum(
            session["corrections_received"] 
            for session in self.distilled_knowledge
        )
        
        pattern_totals = {}
        for session in self.distilled_knowledge:
            for pattern, count in session["patterns_found"].items():
                pattern_totals[pattern] = pattern_totals.get(pattern, 0) + count
        
        return {
            "total_sessions": len(self.distilled_knowledge),
            "total_corrections_processed": total_corrections,
            "dominant_patterns": dict(
                sorted(pattern_totals.items(), key=lambda x: x[1], reverse=True)
            )
        }

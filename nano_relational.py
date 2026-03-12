"""
Nano-Relational Engine
======================
Implements advanced field triangulation and constraint propagation.
Based on the "Nano-Relational" and "Thermodynamic Argument" canon documents.

Core Logic:
1. Parallax Triangulation: Using ratios as vantage points to locate the field.
2. Constraint Propagation: Reducing uncertainty of unknown variables from knowns.
3. Tiling Error Detection: Finding where the multiversal echoes diverge.

"A paint by numbers done with matchsticks." — S. Cross, 2026.
"""

from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class FieldState:
    triangulated_center: float  # The resultant vector of the field
    uncertainty_budget: float   # How much of the field is unconstrained
    tiling_error_magnitude: float
    missing_string_indices: List[str]

class NanoRelationalEngine:
    """
    Translates the nano-relational theory into computable field mechanics.
    """

    def __init__(self, n_contexts: int = 8):
        self.n_contexts = n_contexts
        self.context_labels = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

    def calculate_triangulation(self, context_scores: Dict[str, float]) -> float:
        """
        Treats context scores as vantage points on a single terrarium energy.
        Calculates the central field vector (triangulated center).
        
        Equation: Σ (score_i * confidence_i) / Σ confidence_i
        (Weighted mean of all contexts as a simple triangulation proxy)
        """
        vals = []
        weights = []
        for c in self.context_labels:
            if c in context_scores:
                vals.append(context_scores[c])
                # Confidence is derived from the proximity to the ZPB (0.0 center)
                # or can be provided. Here we use 1.0 as default.
                weights.append(1.0)
        
        if not vals:
            return 0.0
        
        return float(np.average(vals, weights=weights))

    def propagate_constraints(self, context_scores: Dict[str, float], confirmed_vars: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Crystallography principle: confirmed relationships constrain the unknown.
        Returns {context: (min_probable, max_probable)} for unknown variables.
        """
        constraints = {}
        # Simple implementation: using the mean and variance of confirmed vars
        # to bound the unknown ones in a closed terrarium (conservation).
        if not confirmed_vars:
            return {c: (-1.0, 1.0) for c in self.context_labels}

        confirmed_vals = [context_scores[c] for c in confirmed_vars if c in context_scores]
        if not confirmed_vals:
            return {c: (-1.0, 1.0) for c in self.context_labels}

        mean_confirmed = np.mean(confirmed_vals)
        std_confirmed = np.std(confirmed_vals) if len(confirmed_vals) > 1 else 0.2

        for c in self.context_labels:
            if c not in confirmed_vars:
                # unknown vars are constrained to be within 2 sigma of the confirmed field
                constraints[c] = (max(-1.0, mean_confirmed - 2*std_confirmed), 
                                  min(1.0, mean_confirmed + 2*std_confirmed))
        
        return constraints

    def detect_tiling_errors(self, ratios: Dict[str, float]) -> float:
        """
        Checks if ratio(A,B) * ratio(B,C) * ratio(C,A) ≈ 1.0.
        Divergence from 1.0 = Tiling Error (Inconsistency in the multiversal echoes).
        """
        errors = []
        # Check cycles of 3 across the new 8-context field
        triplets = [
            ("C1", "C2", "C3"), ("C2", "C3", "C4"), ("C3", "C4", "C5"), 
            ("C4", "C5", "C6"), ("C5", "C6", "C7"), ("C6", "C7", "C8")
        ]
        
        for a, b, c in triplets:
            r_ab = ratios.get(f"{a}_{b}", 1.0)
            r_bc = ratios.get(f"{b}_{c}", 1.0)
            r_ca = 1.0 / ratios.get(f"{a}_{c}", 1.0) if f"{a}_{c}" in ratios else 1.0
            
            # In a perfectly tiled field, product is 1.0
            cycle_product = r_ab * r_bc * r_ca
            errors.append(abs(1.0 - cycle_product))
            
        return float(np.mean(errors)) if errors else 0.0

    def compute_field_state(self, context_scores: Dict[str, float], ratios: Dict[str, float]) -> FieldState:
        """
        Synthesizes the full nano-relational state.
        """
        center = self.calculate_triangulation(context_scores)
        tiling_error = self.detect_tiling_errors(ratios)
        
        # Uncertainty budget: reduced as more contexts are active
        uncertainty = 1.0 - (len(context_scores) / self.n_contexts)
        
        # Missing strings: where scores deviate most from the triangulated center
        missing = []
        for c, val in context_scores.items():
            if abs(val - center) > 0.5:
                missing.append(c)
                
        return FieldState(
            triangulated_center = center,
            uncertainty_budget = uncertainty,
            tiling_error_magnitude = tiling_error,
            missing_string_indices = missing
        )

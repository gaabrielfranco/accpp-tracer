"""ACC++ circuit tracer for mechanistic interpretability of transformer attention heads."""

__version__ = "0.3.0"

from .circuit import Tracer, get_seeds, get_seeds_prob
from .decomposition import (
    compute_weight_pseudoinverses,
    get_omega_decomposition,
    load_decomposition_cache,
    save_decomposition_cache,
)
from .intervention import EdgeSpec, InterventionResult, edges_from_graph
from .models import ModelConfig, get_model_config
from .rope import get_rotation_matrices, get_rotation_matrix, get_rotary_matrix
from .signals import component_label_to_id, get_component_output
from .tracing import trace_firing

__all__ = [
    # Circuit tracer (Level 2 & 3 API)
    "Tracer",
    "get_seeds",
    "get_seeds_prob",
    # Per-firing tracing (Level 1 API)
    "trace_firing",
    # Signal extraction utilities
    "get_component_output",
    "component_label_to_id",
    # Intervention API
    "EdgeSpec",
    "InterventionResult",
    "edges_from_graph",
    # Decomposition
    "get_omega_decomposition",
    "compute_weight_pseudoinverses",
    "load_decomposition_cache",
    "save_decomposition_cache",
    # RoPE
    "get_rotation_matrices",
    "get_rotation_matrix",
    "get_rotary_matrix",
    # Model config (derived from TransformerLens model.cfg)
    "ModelConfig",
    "get_model_config",
]

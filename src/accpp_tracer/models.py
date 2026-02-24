"""Model configuration for ACC++ circuit tracing.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformer_lens import HookedTransformer


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for ACC++ tracing, derived from TransformerLens model.cfg.

    Attributes:
        has_rope: Whether the model uses Rotary Position Embeddings.
        has_gqa: Whether the model uses Grouped Query Attention.
        gqa_repeats: Number of query heads per KV head group.
        has_post_attn_ln: Whether the model has post-attention layer norm.
        use_numpy_svd: Whether to use numpy SVD (for numerical stability).
    """

    has_rope: bool
    has_gqa: bool
    gqa_repeats: int
    has_post_attn_ln: bool
    use_numpy_svd: bool = False


def get_model_config(
    model: HookedTransformer,
    use_numpy_svd: bool = False,
) -> ModelConfig:
    """Derive ACC++ configuration from a TransformerLens model.

    All architectural flags are read from model.cfg, so any model
    supported by TransformerLens can be traced without registration.

    Args:
        model: A HookedTransformer model instance.
        use_numpy_svd: Use numpy for SVD computation (more numerically
            stable for some models like Pythia). Default: False.

    Returns:
        ModelConfig with all fields derived from model.cfg.

    Raises:
        ValueError: If the model has a configuration that ACC++ cannot handle.
    """
    cfg = model.cfg

    # --- Validation ---
    # Positional embeddings: only "standard" (absolute) and "rotary" (RoPE)
    # are supported. Other types (shortformer, alibi, sinusoidal) would need
    # different attention score decomposition.
    if cfg.positional_embedding_type not in ("standard", "rotary"):
        raise ValueError(
            f"ACC++ only supports 'standard' and 'rotary' positional embeddings, "
            f"got '{cfg.positional_embedding_type}'. "
            f"Model: {cfg.model_name}."
        )

    # Attention direction: must be causal. The recursive tracing assumes
    # src_token <= dest_token (autoregressive masking).
    if cfg.attention_dir != "causal":
        raise ValueError(
            f"ACC++ only supports causal (unidirectional) attention, "
            f"got '{cfg.attention_dir}'. "
            f"Model: {cfg.model_name}."
        )

    # Attention-only models: the residual stream decomposition includes MLP
    # contributions. Without MLPs, the cache keys would be missing.
    if cfg.attn_only:
        raise ValueError(
            f"ACC++ requires MLP layers (attn_only=True is not supported). "
            f"Model: {cfg.model_name}."
        )

    # MoE: mixture-of-experts routing breaks the additive MLP decomposition
    # because the output depends on a non-linear gating function.
    if cfg.num_experts is not None:
        raise ValueError(
            f"ACC++ does not support Mixture-of-Experts models "
            f"(num_experts={cfg.num_experts}). "
            f"Model: {cfg.model_name}."
        )

    # QK normalization: applies RMSNorm to Q and K before computing attention
    # scores, introducing a nonlinearity that breaks the bilinear Omega = W_Q @ W_K^T
    # decomposition.
    if getattr(cfg, "use_qk_norm", False):
        raise ValueError(
            f"ACC++ does not support QK normalization (use_qk_norm=True). "
            f"The bilinear Omega decomposition requires linear Q and K projections. "
            f"Model: {cfg.model_name}."
        )

    # Different RoPE bases per layer type (e.g., Gemma-3 uses rotary_base=1M
    # for global layers and rotary_base_local=10k for local layers). Our
    # get_rotation_matrix uses a single rotary_base for all layers.
    if getattr(cfg, "rotary_base_local", None) is not None:
        raise ValueError(
            f"ACC++ does not support per-layer RoPE bases "
            f"(rotary_base_local={cfg.rotary_base_local}). "
            f"Model: {cfg.model_name}."
        )

    # --- Derive config ---
    has_gqa = (
        cfg.n_key_value_heads is not None
        and cfg.n_key_value_heads < cfg.n_heads
    )
    return ModelConfig(
        has_rope=(cfg.positional_embedding_type == "rotary"),
        has_gqa=has_gqa,
        gqa_repeats=(cfg.n_heads // cfg.n_key_value_heads) if has_gqa else 1,
        has_post_attn_ln=getattr(
            cfg, "use_normalization_before_and_after", False
        ),
        use_numpy_svd=use_numpy_svd,
    )

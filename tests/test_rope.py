"""Tests for RoPE rotation matrix computation."""

import numpy as np
import torch

from accpp_tracer import get_rotation_matrix

from conftest import GOLDEN_DIR

N_GOLDEN_POSITIONS = 8


def test_rotation_matrix_properties(pythia_model):
    d_head = pythia_model.cfg.d_head
    rotary_dim = pythia_model.cfg.rotary_dim

    R0 = get_rotation_matrix(pythia_model, 0, "cpu")
    assert R0.shape == (d_head, d_head)
    # Position 0: zero angles -> identity rotation
    assert torch.allclose(R0, torch.eye(d_head), atol=1e-6)

    R5 = get_rotation_matrix(pythia_model, 5, "cpu")
    # Rotation matrices are orthogonal
    assert torch.allclose(R5 @ R5.T, torch.eye(d_head), atol=1e-5)
    # Non-rotary dims are untouched (identity block)
    assert torch.allclose(
        R5[rotary_dim:, rotary_dim:],
        torch.eye(d_head - rotary_dim),
        atol=1e-6,
    )
    assert R5[:rotary_dim, rotary_dim:].abs().max().item() < 1e-6
    assert R5[rotary_dim:, :rotary_dim].abs().max().item() < 1e-6


def test_rotation_consistency_with_tl_cache(pythia_model):
    """R must reproduce TL's own rotary application: q_rot ~= R @ q_pre.

    TransformerLens computes hook_q pre-rotation internally; we check that
    applying our R to W_Q x + b_Q matches the attention scores implied by
    the model's own sin/cos tables, via the calculate_sin_cos_rotary output.
    """
    sin, cos = pythia_model.blocks[0].attn.calculate_sin_cos_rotary(
        pythia_model.cfg.rotary_dim,
        pythia_model.cfg.n_ctx,
        base=pythia_model.cfg.rotary_base,
        dtype=torch.float32,
    )
    d_head = pythia_model.cfg.d_head
    rotary_dim = pythia_model.cfg.rotary_dim
    pos = 7
    torch.manual_seed(0)
    q = torch.randn(d_head)

    # TL applies: rotated = q * cos + rotate_every_two(q) * sin on the
    # rotary slice (adjacent_pairs=False -> "rotate half" convention).
    q_rot = q[:rotary_dim]
    n = rotary_dim // 2
    rot = torch.cat([-q_rot[n:], q_rot[:n]])
    expected = torch.cat(
        [q_rot * cos[pos] + rot * sin[pos], q[rotary_dim:]]
    )

    R = get_rotation_matrix(pythia_model, pos, "cpu")
    assert torch.allclose(R @ q, expected, atol=1e-5)


def test_golden_rotation_stack(pythia_model, update_golden):
    R = torch.stack(
        [
            get_rotation_matrix(pythia_model, i, "cpu")
            for i in range(N_GOLDEN_POSITIONS)
        ]
    ).numpy()
    path = GOLDEN_DIR / "rope_pythia.npz"
    if update_golden:
        np.savez_compressed(path, R=R)
        return
    golden = np.load(path)["R"]
    assert R.shape == golden.shape
    assert np.max(np.abs(R - golden)) < 1e-6

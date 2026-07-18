"""Tests for the Omega SVD decomposition and its disk cache."""

import numpy as np
import pytest
import torch

from accpp_tracer import (
    compute_weight_pseudoinverses,
    get_model_config,
    get_omega_decomposition,
    load_decomposition_cache,
    save_decomposition_cache,
)

from conftest import GOLDEN_DIR
from golden_utils import assert_decomposition_matches, decomposition_sketch


@pytest.mark.parametrize("model_name", ["gpt2", "pythia"])
def test_omega_svd_reconstruction(model_name, gpt2_model, pythia_model):
    """U @ diag(S) @ VT must reconstruct Omega = W_Q @ W_K^T for every head."""
    model = gpt2_model if model_name == "gpt2" else pythia_model
    config = get_model_config(model)
    U, S, VT = get_omega_decomposition(model, config, "cpu")

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    d_model, d_head = model.cfg.d_model, model.cfg.d_head
    assert U.shape == (n_layers, n_heads, d_model, d_head)
    assert S.shape == (n_layers, n_heads, d_head)
    assert VT.shape == (n_layers, n_heads, d_head, d_model)
    assert U.dtype == torch.float32

    # Singular values non-negative and descending
    assert (S >= 0).all()
    assert (S[..., :-1] - S[..., 1:] >= -1e-5).all()

    # Reconstruction of Omega (rank <= d_head, so top-d_head SVD is exact)
    omega = torch.einsum(
        "lhmk,lhnk->lhmn", model.W_Q.float(), model.W_K.float()
    )
    recon = (U * S.unsqueeze(-2)) @ VT
    max_err = (recon - omega).abs().max().item()
    assert max_err < 1e-3, f"Omega reconstruction max err {max_err}"


@pytest.mark.parametrize("model_name", ["gpt2", "pythia"])
def test_pseudoinverses(model_name, gpt2_model, pythia_model):
    """W_pinv @ W must be the d_head identity (W has full column rank)."""
    model = gpt2_model if model_name == "gpt2" else pythia_model
    config = get_model_config(model)
    W_Q_pinv, W_K_pinv = compute_weight_pseudoinverses(model, config, "cpu")

    d_head = model.cfg.d_head
    eye = torch.eye(d_head)
    for pinv, W in ((W_Q_pinv, model.W_Q.float()), (W_K_pinv, model.W_K.float())):
        prod = pinv @ W  # (n_layers, n_heads, d_head, d_head)
        max_err = (prod - eye).abs().max().item()
        assert max_err < 1e-3, f"pinv identity max err {max_err}"


def test_numpy_svd_matches_torch(pythia_model):
    """The numpy SVD backend must agree with torch (up to SV signs)."""
    config_np = get_model_config(pythia_model, use_numpy_svd=True)
    config_t = get_model_config(pythia_model, use_numpy_svd=False)
    with torch.no_grad():
        U_n, S_n, VT_n = get_omega_decomposition(pythia_model, config_np, "cpu")
        U_t, S_t, VT_t = get_omega_decomposition(pythia_model, config_t, "cpu")

    assert torch.allclose(S_n, S_t, atol=1e-3)
    recon_n = (U_n * S_n.unsqueeze(-2)) @ VT_n
    recon_t = (U_t * S_t.unsqueeze(-2)) @ VT_t
    assert (recon_n - recon_t).abs().max().item() < 1e-3


def test_cache_roundtrip(tmp_path, gpt2_model):
    config = get_model_config(gpt2_model)
    U, S, VT = get_omega_decomposition(gpt2_model, config, "cpu")
    W_Q_pinv, W_K_pinv = compute_weight_pseudoinverses(gpt2_model, config, "cpu")

    path = save_decomposition_cache(
        tmp_path, U, S, VT, W_Q_pinv, W_K_pinv,
        model_name=gpt2_model.cfg.model_name, use_numpy_svd=False,
    )
    assert path.exists()

    loaded = load_decomposition_cache(tmp_path, gpt2_model, False, "cpu")
    assert loaded is not None
    for key, tensor in (
        ("U", U), ("S", S), ("VT", VT),
        ("W_Q_pinv", W_Q_pinv), ("W_K_pinv", W_K_pinv),
    ):
        assert torch.equal(loaded[key], tensor), f"cache roundtrip changed {key}"


def test_robust_svd_single():
    """The fallback SVD path must exactly factor a rank-deficient matrix."""
    from accpp_tracer.decomposition import _robust_svd_single

    torch.manual_seed(0)
    rank = 32
    A = torch.randn(256, rank) @ torch.randn(rank, 256)
    U, S, VT = _robust_svd_single(A, rank)
    assert U.shape == (256, rank) and S.shape == (rank,) and VT.shape == (rank, 256)
    recon = (U * S) @ VT
    assert (recon - A).abs().max().item() < 1e-4 * A.abs().max().item()


def test_cache_version_invalidation(tmp_path, gpt2_model):
    """A cache written under an older format version must be a cache-miss.

    Guards the v1->v2 invalidation: v1 caches may contain silently corrupted
    SVD factors (LAPACK gesdd mis-convergence, pre-verification).
    """
    import h5py

    config = get_model_config(gpt2_model)
    with torch.no_grad():
        U, S, VT = get_omega_decomposition(gpt2_model, config, "cpu")
        W_Q_pinv, W_K_pinv = compute_weight_pseudoinverses(gpt2_model, config, "cpu")
    path = save_decomposition_cache(
        tmp_path, U, S, VT, W_Q_pinv, W_K_pinv,
        model_name=gpt2_model.cfg.model_name, use_numpy_svd=False,
    )
    with h5py.File(path, "r+") as f:
        f.attrs["cache_version"] = 1
    with pytest.warns(UserWarning, match="cache_version"):
        assert load_decomposition_cache(tmp_path, gpt2_model, False, "cpu") is None


def test_cache_miss_on_missing_file(tmp_path, gpt2_model):
    assert load_decomposition_cache(tmp_path, gpt2_model, False, "cpu") is None


def test_cache_miss_on_corrupt_file(tmp_path, gpt2_model):
    cache_file = tmp_path / "gpt2_torch.h5"
    cache_file.write_bytes(b"not an h5 file")
    with pytest.warns(UserWarning):
        assert load_decomposition_cache(tmp_path, gpt2_model, False, "cpu") is None


@pytest.mark.parametrize("model_name", ["gpt2", "pythia"])
def test_golden_decomposition(
    model_name, gpt2_tracer, pythia_tracer, update_golden
):
    """Decomposition tensors must match the pre-refactor golden fingerprint."""
    tracer = gpt2_tracer if model_name == "gpt2" else pythia_tracer
    sketch = decomposition_sketch(
        tracer.U, tracer.S, tracer.VT, tracer.W_Q_pinv, tracer.W_K_pinv
    )
    assert_decomposition_matches(
        sketch, GOLDEN_DIR / f"decomp_{model_name}.npz", update_golden
    )

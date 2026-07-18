"""Omega (QK^T) SVD decomposition for attention heads."""

import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from transformer_lens import HookedTransformer

from .models import ModelConfig
from ._typecheck import typechecked


# Disk-cache format version. Bump when the on-disk layout / dataset names
# change OR when the computation is fixed in a way that invalidates cached
# values (v2: per-head SVD/pinv verification with robust-driver fallback —
# torch's gesdd was observed to silently mis-converge on 2/672 Llama-3.2-3B
# heads, and v1 caches may hold the corrupted factors).
CACHE_VERSION = 2


def _robust_svd_single(
    omega_h: Tensor, rank: int
) -> tuple[Tensor, Tensor, Tensor]:
    """SVD of a single head's Omega with progressively more robust backends.

    Tries numpy's gesdd (a different LAPACK build than torch's, observed to
    succeed where torch's fails), then scipy's slow-but-robust gesvd driver.
    Returns rank-sliced fp32 (U, S, VT) CPU tensors.

    Raises RuntimeError if no backend produces an accurate factorization.
    """
    import scipy.linalg

    A = omega_h.detach().cpu().numpy()
    tol = 1e-4 * max(float(np.abs(A).max()), 1e-8)

    for backend, fn in (
        ("numpy gesdd", lambda: np.linalg.svd(A)),
        ("scipy gesvd", lambda: scipy.linalg.svd(A, lapack_driver="gesvd")),
    ):
        U_np, S_np, VT_np = fn()
        recon = (U_np[:, :rank] * S_np[:rank]) @ VT_np[:rank]
        err = float(np.abs(recon - A).max())
        if err < tol:
            return (
                torch.from_numpy(np.ascontiguousarray(U_np[:, :rank])).float(),
                torch.from_numpy(np.ascontiguousarray(S_np[:rank])).float(),
                torch.from_numpy(np.ascontiguousarray(VT_np[:rank, :])).float(),
            )
        warnings.warn(
            f"SVD fallback backend {backend} reconstruction error {err:.2e} "
            f"exceeds tolerance {tol:.2e}; trying next backend."
        )
    raise RuntimeError(
        "No SVD backend produced an accurate factorization for this head."
    )


@typechecked
def get_omega_decomposition(
    model: HookedTransformer,
    config: ModelConfig,
    device: str = "cpu",
) -> tuple[
    Float[Tensor, "n_layers n_heads d_model d_head"],
    Float[Tensor, "n_layers n_heads d_head"],
    Float[Tensor, "n_layers n_heads d_head d_model"],
]:
    """Compute SVD decomposition of Q@K^T (Omega) for all attention heads.

    Factorizes the attention weight matrix Omega = W_Q @ W_K^T into
    U @ diag(S) @ VT for each attention head, enabling decomposition of
    attention patterns into rank-1 components (singular vectors).

    The SVD is always computed in fp32 regardless of the model's dtype:
    ACC++ is numerically sensitive (same reason TF32 is disabled in
    ``Tracer.__init__``), and bf16/fp16 SVDs lose precision in the singular
    values. Returned tensors are fp32.

    Args:
        model: A HookedTransformer model instance.
        config: Model configuration (from get_model_config).
        device: Torch device for output tensors.

    The SVD is computed one layer at a time and sliced to the top
    ``rank = d_head`` singular vectors immediately: Omega has rank at most
    d_head, so nothing is lost, and the transient allocation is bounded by
    one layer's (n_heads, d_model, d_model) tensors instead of all layers'.
    (At Llama-3.1-8B scale that is ~6 GiB per layer instead of ~200 GiB
    total; the final stored factors are ~4.3 GiB.)

    Returns:
        Tuple of (U, S, VT) where:
            U: Left singular vectors, shape (n_layers, n_heads, d_model, d_head).
            S: Singular values, shape (n_layers, n_heads, d_head).
            VT: Right singular vectors, shape (n_layers, n_heads, d_head, d_model).
    """
    rank = model.cfg.d_head
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    # Cast to fp32 before SVD; output tensors are always fp32 (was: model
    # dtype). Detached — the SVD is a precomputation on frozen weights, and
    # the numpy path cannot handle grad-requiring tensors.
    W_Q = model.W_Q.detach().float()
    W_K = model.W_K.detach().float()
    if config.use_numpy_svd:
        W_Q = W_Q.cpu()
        W_K = W_K.cpu()

    U = torch.empty((n_layers, n_heads, d_model, rank), device=device)
    S = torch.empty((n_layers, n_heads, rank), device=device)
    VT = torch.empty((n_layers, n_heads, rank, d_model), device=device)

    for layer in range(n_layers):
        omega_layer = einsum(
            W_Q[layer],
            W_K[layer],
            "n_heads d_model d_head, n_heads d_model_out d_head "
            "-> n_heads d_model d_model_out",
        )
        if config.use_numpy_svd:
            U_l, S_l, VT_l = np.linalg.svd(omega_layer.numpy())
            U[layer] = torch.from_numpy(U_l[:, :, :rank]).to(device)
            S[layer] = torch.from_numpy(S_l[:, :rank]).to(device)
            VT[layer] = torch.from_numpy(VT_l[:, :rank, :]).to(device)
        else:
            U_l, S_l, VT_l = torch.linalg.svd(omega_layer)
            U[layer] = U_l[:, :, :rank].to(device)
            S[layer] = S_l[:, :rank].to(device)
            VT[layer] = VT_l[:, :rank, :].to(device)

        # Verify per-head reconstruction. LAPACK's gesdd can mis-converge
        # SILENTLY on specific matrices (observed with torch's bundled
        # LAPACK on Llama-3.2-3B, layers 15/21 head 18: reconstruction
        # error ~0.05-0.07 with no exception raised). Omega has rank
        # <= d_head, so the rank-sliced factors must reconstruct it
        # exactly up to fp32 rounding; any larger error means the
        # factorization itself is corrupt and a fallback backend is used.
        recon = (U[layer] * S[layer].unsqueeze(-2)) @ VT[layer]
        err = (recon - omega_layer.to(device)).abs().amax(dim=(1, 2))
        thresh = 1e-4 * omega_layer.abs().amax(dim=(1, 2)).to(device).clamp(
            min=1e-8
        )
        for h in torch.where(err > thresh)[0].tolist():
            warnings.warn(
                f"SVD verification failed for layer {layer} head {h} "
                f"(reconstruction error {err[h]:.2e}); recomputing with a "
                "fallback backend."
            )
            U_h, S_h, VT_h = _robust_svd_single(omega_layer[h], rank)
            U[layer, h] = U_h.to(device)
            S[layer, h] = S_h.to(device)
            VT[layer, h] = VT_h.to(device)

    return U, S, VT


@typechecked
def compute_weight_pseudoinverses(
    model: HookedTransformer,
    config: ModelConfig,
    device: str = "cpu",
) -> tuple[
    Float[Tensor, "n_layers n_heads d_head d_model"],
    Float[Tensor, "n_layers n_heads d_head d_model"],
]:
    """Compute pseudoinverses of W_Q and W_K weight matrices.

    Used for computing bias offsets in the trace_firing algorithm.
    Uses numpy for models that require it for numerical stability.

    The pseudoinverse is always computed in fp32 regardless of the model's
    dtype (same rationale as ``get_omega_decomposition``). Returned tensors
    are fp32.

    Args:
        model: A HookedTransformer model instance.
        config: Model configuration (from get_model_config).
        device: Torch device for output tensors.

    Returns:
        Tuple of (W_Q_pinv, W_K_pinv).
    """
    # Cast to fp32 before pinv; output tensors are always fp32 (was: model
    # dtype). Detached for the same reason as in get_omega_decomposition.
    W_Q = model.W_Q.detach().float()
    W_K = model.W_K.detach().float()

    if config.use_numpy_svd:
        W_Q_pinv = torch.from_numpy(np.linalg.pinv(W_Q.cpu().numpy())).to(device)
        W_K_pinv = torch.from_numpy(np.linalg.pinv(W_K.cpu().numpy())).to(device)
    else:
        W_Q_pinv = torch.linalg.pinv(W_Q).to(device)
        W_K_pinv = torch.linalg.pinv(W_K).to(device)

    # Verify the Moore-Penrose property W @ pinv @ W == W per head. Unlike
    # (pinv @ W == I), this holds for ANY rank — rank-deficient heads (e.g.
    # Llama-3.2-3B layer-0 heads 21/22) are not false positives. Guards
    # against the same silent LAPACK mis-convergence as in
    # get_omega_decomposition (pinv is SVD-based internally).
    for name, W, pinv in (("W_Q", W_Q, W_Q_pinv), ("W_K", W_K, W_K_pinv)):
        W_dev = W.to(device)
        err = (W_dev @ (pinv @ W_dev) - W_dev).abs().amax(dim=(2, 3))
        thresh = 1e-4 * W_dev.abs().amax(dim=(2, 3)).clamp(min=1e-8)
        bad = torch.nonzero(err > thresh, as_tuple=False)
        for layer, h in bad.tolist():
            warnings.warn(
                f"pinv verification failed for {name} layer {layer} head {h} "
                f"(Moore-Penrose error {err[layer, h]:.2e}); recomputing "
                "with numpy."
            )
            fixed = np.linalg.pinv(W[layer, h].cpu().numpy())
            if name == "W_Q":
                W_Q_pinv[layer, h] = torch.from_numpy(fixed).to(device)
            else:
                W_K_pinv[layer, h] = torch.from_numpy(fixed).to(device)

    return W_Q_pinv, W_K_pinv


def _cache_filename(model_name: str, use_numpy_svd: bool) -> str:
    """Build the cache filename for a (model_name, use_numpy_svd) pair.

    Slashes in HF-style model names (e.g. ``EleutherAI/pythia-160m``) are
    replaced with ``__`` for filesystem-safety. The SVD backend
    (``torch`` vs ``numpy``) is suffixed because the two produce numerically
    distinct results.
    """
    safe_name = model_name.replace("/", "__").replace("\\", "__")
    suffix = "numpy" if use_numpy_svd else "torch"
    return f"{safe_name}_{suffix}.h5"


def _expected_shapes(model: HookedTransformer) -> dict[str, tuple[int, ...]]:
    """Expected (n_layers, n_heads, ...) shapes for the five cached tensors."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    return {
        "U": (n_layers, n_heads, d_model, d_head),
        "S": (n_layers, n_heads, d_head),
        "VT": (n_layers, n_heads, d_head, d_model),
        "W_Q_pinv": (n_layers, n_heads, d_head, d_model),
        "W_K_pinv": (n_layers, n_heads, d_head, d_model),
    }


def load_decomposition_cache(
    cache_dir: str | Path,
    model: HookedTransformer,
    use_numpy_svd: bool,
    device: str = "cpu",
) -> dict[str, Tensor] | None:
    """Try to load cached ``U, S, VT, W_Q_pinv, W_K_pinv`` from disk.

    Returns ``None`` (cache-miss) on any failure: missing file, unreadable
    file, missing datasets, or shape mismatch with the current model.
    The caller is expected to fall back to recomputation and re-save.

    The cache file uses h5py with gzip compression. All tensors are stored
    in fp32 and loaded back as fp32 onto ``device``.

    Args:
        cache_dir: Directory containing cache files.
        model: HookedTransformer model — used to validate cached shapes and
            to derive the cache filename from ``model.cfg.model_name``.
        use_numpy_svd: Selects the cache file variant (``torch`` vs
            ``numpy`` SVD backend).
        device: Torch device to place loaded tensors on.

    Returns:
        Dict with keys ``"U", "S", "VT", "W_Q_pinv", "W_K_pinv"`` mapping to
        fp32 tensors on ``device``, or ``None`` on cache-miss.
    """
    cache_path = Path(cache_dir) / _cache_filename(model.cfg.model_name, use_numpy_svd)
    if not cache_path.exists():
        return None

    expected = _expected_shapes(model)
    try:
        with h5py.File(cache_path, "r") as f:
            version = int(f.attrs.get("cache_version", 0))
            if version != CACHE_VERSION:
                warnings.warn(
                    f"Cache file {cache_path} has cache_version={version} "
                    f"but the current format is {CACHE_VERSION}; falling "
                    "back to recomputation (the file will be overwritten)."
                )
                return None
            out: dict[str, Tensor] = {}
            for key, exp_shape in expected.items():
                if key not in f:
                    warnings.warn(
                        f"Cache file {cache_path} missing dataset '{key}'; "
                        "falling back to recomputation."
                    )
                    return None
                arr = f[key][:]
                if tuple(arr.shape) != exp_shape:
                    warnings.warn(
                        f"Cache file {cache_path} has shape {arr.shape} for '{key}' "
                        f"but model expects {exp_shape}; falling back to recomputation."
                    )
                    return None
                out[key] = torch.from_numpy(arr).to(device)
            return out
    except (OSError, KeyError, ValueError) as e:
        warnings.warn(
            f"Failed to load cache file {cache_path} ({type(e).__name__}: {e}); "
            "falling back to recomputation."
        )
        return None


def save_decomposition_cache(
    cache_dir: str | Path,
    U: Tensor,
    S: Tensor,
    VT: Tensor,
    W_Q_pinv: Tensor,
    W_K_pinv: Tensor,
    model_name: str,
    use_numpy_svd: bool,
    compression_level: int = 9,
) -> Path:
    """Save the decomposition tensors to a gzip-compressed h5 file.

    Tensors are written in fp32 to a file under ``cache_dir``. The directory
    is created if it does not exist.

    Args:
        cache_dir: Directory to write the cache file into.
        U, S, VT: Omega SVD tensors.
        W_Q_pinv, W_K_pinv: Weight pseudoinverses.
        model_name: ``model.cfg.model_name``; used to compose the filename.
        use_numpy_svd: Whether the SVD was computed with numpy (selects the
            ``_numpy.h5`` vs ``_torch.h5`` filename suffix).
        compression_level: gzip level 0–9 (default 9 — see CHANGELOG for
            rationale: this cache is written once and read many times, so
            the asymmetric workload makes the slower write irrelevant).

    Returns:
        The path of the written file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_filename(model_name, use_numpy_svd)

    tensors = {
        "U": U,
        "S": S,
        "VT": VT,
        "W_Q_pinv": W_Q_pinv,
        "W_K_pinv": W_K_pinv,
    }
    with h5py.File(cache_path, "w") as f:
        for key, tensor in tensors.items():
            f.create_dataset(
                key,
                data=tensor.detach().cpu().float().numpy(),
                compression="gzip",
                compression_opts=compression_level,
            )
        f.attrs["model_name"] = model_name
        f.attrs["use_numpy_svd"] = use_numpy_svd
        f.attrs["cache_version"] = CACHE_VERSION

    return cache_path

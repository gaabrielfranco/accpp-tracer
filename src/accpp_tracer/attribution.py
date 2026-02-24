"""Integrated Gradients attribution for ordering component contributions.
"""

from typing import Tuple

import numpy as np
from jaxtyping import Float
from torch import Tensor

from ._typecheck import typechecked


@typechecked
def ig_softmax_attributions(
    X: Float[Tensor, "n_components n_tokens"],
    j: int,
    T: int = 64,
    quadrature: str = "trapezoid",
) -> Tuple[np.ndarray, dict]:
    """Integrated Gradients attributions for softmax probability.

    Computes IG attributions for p_j = softmax(S)[j] where S = sum(X, axis=0).
    Uses the identity grad_S p_j(S) = p_j (e_j - p), so
    IG_i(p_j) = integral_0^1 p_j(alpha*S) * (x_{i,j} - x_i . p(alpha*S)) d_alpha.

    Args:
        X: Input tensor of shape (N, n) where N is the number of components.
        j: Target index for the softmax probability.
        T: Number of integration steps.
        quadrature: Integration method ("trapezoid", "riemann_left", "riemann_right").

    Returns:
        Tuple of (contributions array of shape (N,), info dict with S, p, p_j).
    """

    def stable_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        z_max = np.max(z, axis=axis, keepdims=True)
        e = np.exp(z - z_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    X = np.asarray(X.cpu(), dtype=float)
    N, n = X.shape
    if not (0 <= j < n):
        raise ValueError("j must be a valid index in [0, n).")
    if T < 1:
        raise ValueError("T must be >= 1.")
    if quadrature not in {"trapezoid", "riemann_left", "riemann_right"}:
        raise ValueError(
            "quadrature must be one of {'trapezoid', 'riemann_left', 'riemann_right'}."
        )

    S = X.sum(axis=0)
    alphas = np.linspace(0.0, 1.0, T + 1)
    Z = alphas[:, None] * S[None, :]
    P = stable_softmax(Z, axis=1)  # (T+1, n)
    pj_path = P[:, j]  # (T+1,)

    M = X @ P.T  # (N, T+1)
    diff = X[:, [j]] - M  # (N, T+1)
    weighted = diff * pj_path[None, :]  # (N, T+1)

    h = 1.0 / T
    if quadrature == "trapezoid":
        integral = h * (
            0.5 * weighted[:, 0]
            + weighted[:, 1:-1].sum(axis=1)
            + 0.5 * weighted[:, -1]
        )
    elif quadrature == "riemann_left":
        integral = h * weighted[:, :-1].sum(axis=1)
    else:  # riemann_right
        integral = h * weighted[:, 1:].sum(axis=1)

    contribs = integral
    info = {
        "S": S,
        "p": stable_softmax(S),
        "p_j": stable_softmax(S)[j],
    }
    return contribs, info

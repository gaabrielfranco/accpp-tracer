"""Golden-file helpers: serialization and sign-invariant comparison.

Golden files capture pre-refactor outputs so that memory refactors can be
validated as behavior-preserving. All comparisons are invariant to SVD sign
conventions (per-SV sign flips of U/V pairs cancel in every quantity we
compare) and allow small float tolerances for contraction-order changes.
"""

import json
from pathlib import Path

import numpy as np
import torch

SKETCH_COLS = 8
SKETCH_SEED = 0


def graph_to_jsonable(G) -> dict:
    """Serialize a traced circuit graph deterministically."""
    nodes = {}
    for n, data in G.nodes(data=True):
        entry = {}
        if "attn_weight" in data:
            entry["attn_weight"] = float(data["attn_weight"])
        nodes[repr(n)] = entry

    edges = []
    for u, v, k, data in G.edges(keys=True, data=True):
        edges.append(
            {
                "u": repr(u),
                "v": repr(v),
                "type": data.get("type"),
                "svs_used": data.get("svs_used"),
                "weight": float(data["weight"]),
            }
        )
    edges.sort(key=lambda e: (e["u"], e["v"], str(e["type"]), str(e["svs_used"])))
    return {"nodes": nodes, "edges": edges}


def assert_graph_matches(G, golden_path: Path, update: bool, atol: float = 5e-3):
    """Compare a traced graph against its golden serialization.

    Topology, edge types, and selected singular vectors must match EXACTLY;
    weights compare within ``atol``. The default 5e-3 accommodates fp32
    contraction-reorder noise (measured ~1e-4 on attention scores, amplified
    through softmax differences) while staying below the library's own 1e-2
    decomposition-correctness tolerance.
    """
    actual = graph_to_jsonable(G)
    if update:
        golden_path.write_text(json.dumps(actual, indent=1))
        return
    golden = json.loads(golden_path.read_text())

    assert set(actual["nodes"]) == set(golden["nodes"]), (
        "Node sets differ.\n"
        f"only in actual: {set(actual['nodes']) - set(golden['nodes'])}\n"
        f"only in golden: {set(golden['nodes']) - set(actual['nodes'])}"
    )
    for n, gdata in golden["nodes"].items():
        if "attn_weight" in gdata:
            assert abs(actual["nodes"][n]["attn_weight"] - gdata["attn_weight"]) < atol

    def key(e):
        return (e["u"], e["v"], str(e["type"]), str(e["svs_used"]))

    actual_edges = {key(e): e for e in actual["edges"]}
    golden_edges = {key(e): e for e in golden["edges"]}
    assert set(actual_edges) == set(golden_edges), (
        "Edge sets differ.\n"
        f"only in actual: {sorted(set(actual_edges) - set(golden_edges))}\n"
        f"only in golden: {sorted(set(golden_edges) - set(actual_edges))}"
    )
    for k, ge in golden_edges.items():
        assert abs(actual_edges[k]["weight"] - ge["weight"]) < atol, (
            f"weight mismatch on edge {k}: "
            f"{actual_edges[k]['weight']} vs golden {ge['weight']}"
        )


def trace_firing_to_jsonable(svs_dest, ew_dest, svs_src, ew_src) -> dict:
    def dictify(d, values_are_lists):
        out = {}
        for k, v in d.items():
            out[repr(tuple(int(x) for x in k))] = (
                [int(x) for x in v] if values_are_lists else float(v)
            )
        return out

    return {
        "svs_dest": dictify(svs_dest, True),
        "edge_weights_dest": dictify(ew_dest, False),
        "svs_src": dictify(svs_src, True),
        "edge_weights_src": dictify(ew_src, False),
    }


def assert_trace_firing_matches(
    result: dict, golden_path: Path, update: bool, atol: float = 5e-3
):
    if update:
        golden_path.write_text(json.dumps(result, indent=1))
        return
    golden = json.loads(golden_path.read_text())
    for side in ("svs_dest", "svs_src"):
        assert result[side] == golden[side], (
            f"{side} differs:\nactual: {result[side]}\ngolden: {golden[side]}"
        )
    for side in ("edge_weights_dest", "edge_weights_src"):
        assert set(result[side]) == set(golden[side]), f"{side} keys differ"
        for k, gv in golden[side].items():
            assert abs(result[side][k] - gv) < atol, (
                f"{side}[{k}]: {result[side][k]} vs golden {gv}"
            )


def decomposition_sketch(U, S, VT, W_Q_pinv, W_K_pinv) -> dict:
    """Small, SVD-sign-invariant fingerprint of the decomposition tensors.

    Omega is fingerprinted via (U * S) @ (VT @ Rm) with a fixed random matrix
    Rm — per-SV sign flips of (U_k, V_k) pairs cancel in the product. The
    pseudoinverses have no sign ambiguity and are sketched directly.
    """
    U, S, VT = U.detach(), S.detach(), VT.detach()
    W_Q_pinv, W_K_pinv = W_Q_pinv.detach(), W_K_pinv.detach()
    d_model = U.shape[2]
    g = torch.Generator().manual_seed(SKETCH_SEED)
    Rm = torch.randn(d_model, SKETCH_COLS, generator=g)
    omega_sketch = (U * S.unsqueeze(-2)) @ (VT @ Rm)
    return {
        "S": S.cpu().numpy(),
        "omega_sketch": omega_sketch.cpu().numpy(),
        "wq_pinv_sketch": (W_Q_pinv @ Rm).cpu().numpy(),
        "wk_pinv_sketch": (W_K_pinv @ Rm).cpu().numpy(),
    }


def assert_decomposition_matches(
    sketch: dict, golden_path: Path, update: bool, atol: float = 1e-3
):
    if update:
        np.savez_compressed(golden_path, **sketch)
        return
    golden = np.load(
        golden_path.with_suffix(".npz")
        if golden_path.suffix != ".npz"
        else golden_path
    )
    for key in ("S", "omega_sketch", "wq_pinv_sketch", "wk_pinv_sketch"):
        a, g = sketch[key], golden[key]
        assert a.shape == g.shape, f"{key} shape {a.shape} vs golden {g.shape}"
        max_err = np.max(np.abs(a - g))
        assert max_err < atol, f"{key} max abs err {max_err} >= {atol}"

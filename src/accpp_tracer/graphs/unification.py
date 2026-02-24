"""Combine per-prompt circuit graphs into a unified graph."""

from typing import Sequence

import networkx as nx
import numpy as np

from .pruning import remove_isolated_nodes


def combine_prompt_graphs(
    graphs: Sequence[nx.MultiDiGraph | None],
) -> nx.MultiDiGraph:
    """Combine individual per-prompt circuit graphs into a single unified graph.

    Merges edges across prompts, accumulating weights and tracking which
    prompts each edge appeared in.

    Args:
        graphs: Sequence of per-prompt graphs (None entries are skipped).
            Each graph should have edges with "weight" and "type" attributes.

    Returns:
        Combined MultiDiGraph with accumulated weights and "prompts_appeared"
        lists on each edge, and "n_appearences" counts on each node.
    """
    G_combined = nx.MultiDiGraph()

    for prompt_id, G_curr in enumerate(graphs):
        if G_curr is None:
            continue

        for edge in G_curr.edges(data=True):
            if (edge[0], edge[1]) in G_combined.edges:
                if len(G_combined[edge[0]][edge[1]]) > 2:
                    raise ValueError(
                        f"More than 2 parallel edges between {edge[0]} and {edge[1]}"
                    )
                elif len(G_combined[edge[0]][edge[1]]) == 2:
                    # Two edges (type=d and type=s). Find the matching one.
                    candidate_edge = G_combined[edge[0]][edge[1]][0]
                    if candidate_edge["type"] == edge[2]["type"]:
                        G_combined.edges[(edge[0], edge[1], 0)]["weight"] += edge[2][
                            "weight"
                        ]
                        G_combined.edges[(edge[0], edge[1], 0)][
                            "prompts_appeared"
                        ].append(prompt_id)
                    else:
                        G_combined.edges[(edge[0], edge[1], 1)]["weight"] += edge[2][
                            "weight"
                        ]
                        G_combined.edges[(edge[0], edge[1], 1)][
                            "prompts_appeared"
                        ].append(prompt_id)
                else:
                    # Single edge. Check if same type.
                    if (
                        G_combined.edges[(edge[0], edge[1], 0)]["type"]
                        == edge[2]["type"]
                    ):
                        G_combined.edges[(edge[0], edge[1], 0)]["weight"] += edge[2][
                            "weight"
                        ]
                        G_combined.edges[(edge[0], edge[1], 0)][
                            "prompts_appeared"
                        ].append(prompt_id)
                    else:
                        # Different type: create new parallel edge
                        G_combined.add_edge(
                            edge[0],
                            edge[1],
                            prompts_appeared=[prompt_id],
                            **edge[2],
                        )
            else:
                G_combined.add_edge(
                    edge[0],
                    edge[1],
                    prompts_appeared=[prompt_id],
                    **edge[2],
                )

        for node in G_curr.nodes:
            if "n_appearences" in G_combined.nodes[node]:
                G_combined.nodes[node]["n_appearences"] += 1
            else:
                G_combined.nodes[node]["n_appearences"] = 1

    # Normalize edge weights by number of non-None graphs
    n_graphs = sum(1 for g in graphs if g is not None)
    for edge in G_combined.edges:
        G_combined.edges[edge]["weight"] /= n_graphs

    return G_combined


def prune_by_frequency(
    G: nx.MultiDiGraph,
    threshold: float,
    n_graphs: int,
) -> nx.MultiDiGraph:
    """Prune a combined graph by node appearance frequency.

    Keeps only nodes that appeared in at least threshold * n_graphs prompts,
    then removes isolated nodes.

    Args:
        G: Combined graph from combine_prompt_graphs().
        threshold: Fraction of prompts a node must appear in (0.0 to 1.0).
        n_graphs: Total number of graphs used in combination.

    Returns:
        Pruned copy of the graph.
    """
    n_appearences_threshold = int(round(threshold * n_graphs, 0))
    G_pruned = nx.MultiDiGraph(
        nx.subgraph_view(
            G,
            filter_node=lambda node: G.nodes[node].get("n_appearences", 0)
            >= n_appearences_threshold,
        )
    )
    remove_isolated_nodes(G_pruned)
    return G_pruned

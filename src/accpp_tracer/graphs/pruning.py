"""General graph pruning utilities."""

import networkx as nx


def remove_isolated_nodes(G: nx.MultiDiGraph) -> None:
    """Remove degree-zero nodes from a graph in-place.

    Args:
        G: Graph to prune (modified in-place).
    """
    isolated = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(isolated)


def keep_connected_to_root(
    G: nx.MultiDiGraph,
    root_node: str,
) -> None:
    """Remove all connected components that don't contain the root node.

    Args:
        G: Graph to prune (modified in-place).
        root_node: The node that must be reachable.
    """
    for cc in list(nx.weakly_connected_components(G)):
        if root_node not in cc:
            for node in cc:
                G.remove_node(node)

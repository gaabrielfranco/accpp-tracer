"""Graph visualization utilities for Cytoscape-compatible layouts."""

import networkx as nx


def format_graph_cytoscape_by_token_pos(
    G: nx.Graph,
    root_node: str,
    n_layers: int,
    n_heads: int,
    sentence_tokens: list[str],
) -> nx.Graph:
    """Format a circuit graph for Cytoscape visualization, laid out by token position.

    Positions nodes in a grid where x-axis is token position and y-axis is layer.
    Nodes are color-coded by component type (MLP, AH bias, Embedding, AH offset).

    Args:
        G: NetworkX graph representing the circuit.
        root_node: The root/output node identifier.
        n_layers: Number of model layers.
        n_heads: Number of attention heads per layer.
        sentence_tokens: List of token strings for x-axis labels.

    Returns:
        Modified graph with position and style attributes for Cytoscape.
    """
    node_height = 40
    intra_group_gap = 30
    inter_layer_gap = 30
    token_spacing = 200

    pos: dict[str, tuple[float, float]] = {}

    bottom_y = 0

    for i, token in enumerate(sentence_tokens):
        bottom_node_id = token
        x = i * token_spacing
        pos[bottom_node_id] = (x, bottom_y)
        G.add_node(
            bottom_node_id,
            x=x,
            y=bottom_y,
            background_color="#FFFFFF",
            border_color="#FFFFFF",
            color="#FFFFFF",
            shape="square",
            label=token,
        )

    # Group internal nodes by layer
    layer_nodes: dict[int, list[str]] = {layer: [] for layer in range(n_layers)}
    for node in G.nodes:
        if node == root_node or node in sentence_tokens:
            continue
        layer, ah_idx, dest_token, src_token = eval(node)
        layer_nodes[layer].append(node)

    # Sort nodes in each layer by dest_token position
    for layer in range(n_layers):
        layer_nodes[layer].sort(
            key=lambda n: sentence_tokens.index(eval(n)[2])
            if eval(n)[2] in sentence_tokens
            else -1
        )

    cumulative_y_offset = node_height + inter_layer_gap

    def node_sort_func(node_str: str) -> int:
        try:
            tup = eval(node_str)
            second = tup[1]
            if second == "Embedding":
                return -1
            elif second == "AH bias":
                return 1000000 - 2
            elif second == "MLP":
                return 1000000 - 1
            elif second == "AH offset":
                return 1000000
            elif isinstance(second, int):
                return second
            else:
                return 999999
        except Exception:
            return 999999

    # Position nodes layer by layer
    for layer in range(n_layers):
        nodes = layer_nodes[layer]
        if not nodes:
            cumulative_y_offset += node_height + inter_layer_gap
            continue

        # Group by dest_token position
        token_groups: dict[int, list[str]] = {}
        for node in nodes:
            token = eval(node)[2]
            try:
                token_index = sentence_tokens.index(token)
            except ValueError:
                token_index = 0
            token_groups.setdefault(token_index, []).append(node)

        # Compute block height for this layer
        layer_block_height = 0
        for group in token_groups.values():
            m = len(group)
            group_height = m * node_height + (m - 1) * intra_group_gap
            layer_block_height = max(layer_block_height, group_height)
        if layer_block_height == 0:
            layer_block_height = node_height

        # Assign positions
        for token_index, group in token_groups.items():
            base_x = token_index * token_spacing
            m = len(group)
            group = sorted(group, key=node_sort_func)
            for i, node in enumerate(group):
                offset = (i - (m - 1) / 2) * (node_height + intra_group_gap)
                y = -(cumulative_y_offset + layer_block_height / 2 + offset)
                pos[node] = (base_x, y)

        cumulative_y_offset += layer_block_height + inter_layer_gap

    # Update bottom node positions
    for i, token in enumerate(sentence_tokens):
        bottom_node_id = token
        x = i * token_spacing
        pos[bottom_node_id] = (x, -10.0)

    # Position root node below everything
    pos[root_node] = (x, min(p[1] for p in pos.values()) - 50)

    # Apply positions to graph
    nx.set_node_attributes(
        G, {node: {"x": coord[0], "y": coord[1]} for node, coord in pos.items()}
    )

    # Style internal nodes by component type
    for node in G.nodes:
        if node == root_node or node in sentence_tokens:
            continue
        node_tuple = eval(node)
        if node_tuple[1] == "MLP":
            G.nodes[node]["shape"] = "ellipse"
            G.nodes[node]["color"] = "#32CD32"
        elif node_tuple[1] == "AH bias":
            G.nodes[node]["shape"] = "diamond"
            G.nodes[node]["color"] = "#FFD700"
        elif node_tuple[1] == "Embedding":
            G.nodes[node]["shape"] = "hexagon"
            G.nodes[node]["color"] = "#00FFFF"
        elif node_tuple[1] == "AH offset":
            G.nodes[node]["shape"] = "octagon"
            G.nodes[node]["color"] = "#8400FF"

    # Relabel nodes for readability
    _, dest_token_root = eval(root_node)
    node_names_mapping = {root_node: f"Logit direction\n{dest_token_root}"}
    for node in list(G.nodes):
        if node == root_node or node in sentence_tokens:
            continue
        node_tuple = eval(node)
        if node_tuple[1] in ["MLP", "AH bias", "Embedding", "AH offset"]:
            node_names_mapping[node] = f"{node_tuple[1]} {node_tuple[0]}\n{node_tuple[2]}"
        else:
            node_names_mapping[node] = (
                f"AH({node_tuple[0]},{node_tuple[1]})\n({node_tuple[2]},{node_tuple[3]})"
            )

    G = nx.relabel_nodes(G, node_names_mapping)

    return G

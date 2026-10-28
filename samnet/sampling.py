import random as rd
import networkx as nx
import numpy as np
from collections import deque
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import Union, Optional, Dict


def snowball(G, n, seeds=1, *args, **kwargs):
    """
    Snowball sampling method using BFS-like expansion from random seeds.
    """
    if n >= len(G):
        return G.copy()

    nodes = list(G.nodes())
    in_sampled_nodes = rd.sample(nodes, min(seeds, len(nodes)))

    visited = set(in_sampled_nodes)
    current_layer = in_sampled_nodes
    edgelist = []

    while len(visited) < n and current_layer:
        next_layer = []

        for node in current_layer:
            if len(visited) >= n:
                break

            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    edgelist.append((node, neighbor))
                    next_layer.append(neighbor)
                    if len(visited) >= n:
                        break
                else:
                    edgelist.append((node, neighbor))

        current_layer = next_layer

    return nx.Graph(edgelist)


def RWS(G, n, r=0, max_steps=1000000, *args, **kwargs):
    """
    Random Walk Sampling (RWS) with pre-cached neighbor lookup and edge deduplication.
    """
    if n >= len(G):
        return G.copy()

    valid_components = [c for c in nx.connected_components(G) if len(c) >= n]
    if not valid_components:
        valid_components = [max(nx.connected_components(G), key=len)]

    chosen_component = rd.choice(valid_components)
    start_node = rd.choice(list(chosen_component))

    #Pre-cache neighbor lists to avoid allocation overhead in loop
    adj = {node: list(G.neighbors(node)) for node in G.nodes()}

    current = start_node
    visited = {current}
    visited_edges = set()
    steps = 0

    while len(visited) < n and steps < max_steps:
        steps += 1
        neighbors = adj[current]
        if not neighbors:
            break

        next_node = rd.choice(neighbors)
        edge = (current, next_node) if current < next_node else (next_node, current)
        visited_edges.add(edge)
        visited.add(next_node)

        if rd.random() < r:
            current = start_node
        else:
            current = next_node

    sampled_graph = nx.Graph()
    sampled_graph.add_nodes_from(visited)
    sampled_graph.add_edges_from(visited_edges)
    return sampled_graph


def IRWS(G, n, r=0, max_steps=1000000, *args, **kwargs):
    """
    Induced Random Walk Sampling (IRWS).
    """
    if n >= len(G):
        return G.copy()

    #Filter components that can hold at least n nodes to prevent infinite loops
    valid_components = [c for c in nx.connected_components(G) if len(c) >= n]
    if not valid_components:
        valid_components = [max(nx.connected_components(G), key=len)]

    chosen_component = rd.choice(valid_components)
    start_node = rd.choice(list(chosen_component))

    #Pre-cache neighbor lists
    adj = {node: list(G.neighbors(node)) for node in G.nodes()}

    current = start_node
    visited = {current}
    steps = 0

    while len(visited) < n and steps < max_steps:
        steps += 1
        neighbors = adj[current]
        if not neighbors:
            break

        next_node = rd.choice(neighbors)
        visited.add(next_node)

        if rd.random() < r:
            current = start_node
        else:
            current = next_node

    return G.subgraph(visited).copy()


def traceroute(G, n, *args, **kwargs):
    """
    Traceroute sampling using shortest paths between random node pairs.
    """
    if n >= len(G):
        return G.copy()

    nodes = list(G.nodes())
    visited_nodes = set()
    edgelist = []

    while len(visited_nodes) < n:
        source, target = rd.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, source, target)
            for u, v in zip(path[:-1], path[1:]):
                edgelist.append((u, v))
                visited_nodes.add(u)
                visited_nodes.add(v)
                if len(visited_nodes) >= n:
                    break
        except nx.NetworkXNoPath:
            continue

    sampled_graph = nx.Graph(edgelist)

    #Ensure connectivity by retaining the largest connected component
    if not nx.is_connected(sampled_graph):
        largest_cc = max(nx.connected_components(sampled_graph), key=len)
        S = sampled_graph.subgraph(largest_cc).copy()

        s_nodes = set(S.nodes())
        not_in_s_nodes = list(set(nodes) - s_nodes)

        while len(S) < n and not_in_s_nodes:
            source = rd.choice(list(s_nodes))
            target = rd.choice(not_in_s_nodes)

            try:
                path = nx.shortest_path(G, source, target)
                for u, v in zip(path[:-1], path[1:]):
                    S.add_edge(u, v)
                    s_nodes.add(u)
                    s_nodes.add(v)
                    if len(S) >= n:
                        break
                not_in_s_nodes = [node for node in not_in_s_nodes if node not in s_nodes]
            except nx.NetworkXNoPath:
                continue

        sampled_graph = S

    return sampled_graph


def MHRW(G, n, tol=100000, *args, **kwargs):
    """
    Metropolis-Hastings Random Walk (MHRW) sampler with pre-cached neighbor lookup.
    """
    if n >= len(G):
        return G.copy()

    valid_components = [c for c in nx.connected_components(G) if len(c) >= n]
    if not valid_components:
        valid_components = [max(nx.connected_components(G), key=len)]

    chosen_component = rd.choice(valid_components)
    current = rd.choice(list(chosen_component))

    #Pre-cache neighbor lists and degrees
    adj = {node: list(G.neighbors(node)) for node in G.nodes()}
    degrees = dict(G.degree())

    visited = {current}
    visited_edges = set()
    control = 0

    while len(visited) < n and control < tol:
        control += 1
        neighbors = adj[current]
        if not neighbors:
            break

        next_node = rd.choice(neighbors)
        kc = degrees[current]
        kn = degrees[next_node]

        if rd.random() <= min(1.0, kc / kn):
            edge = (current, next_node) if current < next_node else (next_node, current)
            visited_edges.add(edge)
            visited.add(next_node)
            current = next_node

    sampled_graph = nx.Graph()
    sampled_graph.add_nodes_from(visited)
    sampled_graph.add_edges_from(visited_edges)
    return sampled_graph


def forest_fire_sampling(G, n, p=0.5, max_restarts=100, *args, **kwargs):
    """
    Forest Fire sampling for undirected graphs.
    Guarantees a single connected subgraph of size n.
    """
    num_nodes = len(G)
    if n >= num_nodes:
        return G.copy()

    valid_components = [
        c for c in nx.connected_components(G) if len(c) >= n
    ]

    if not valid_components:
        raise ValueError(f"No connected component in graph has at least {n} nodes.")

    geo_param = 1.0 - p

    for _ in range(max_restarts):
        chosen_component = rd.choice(valid_components)
        seed = rd.choice(list(chosen_component))

        visited = {seed}
        queue = deque([seed])

        while len(visited) < n:
            if not queue:
                boundary = set()
                for node in visited:
                    boundary.update(G.neighbors(node))
                boundary -= visited

                if not boundary:
                    break

                new_seed = rd.choice(list(boundary))
                visited.add(new_seed)
                queue.append(new_seed)

                if len(visited) == n:
                    break

            curr_node = queue.popleft()

            candidates = [nbr for nbr in G.neighbors(curr_node) if nbr not in visited]
            if not candidates:
                continue

            x = np.random.geometric(geo_param)
            remaining_needed = n - len(visited)
            num_to_sample = min(x, len(candidates), remaining_needed)

            if num_to_sample == 0:
                continue

            selected_neighbors = rd.sample(candidates, k=num_to_sample)

            for neighbor in selected_neighbors:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) == n:
                    break

        if len(visited) == n:
            return G.subgraph(visited).copy()

    raise RuntimeError("Failed to sample a connected component within max_restarts.")


def frontier_sampling(G, n, m=1, *args, **kwargs):
    """
    Frontier Sampling modified to return a connected subgraph with n nodes.
    """
    target_component = max(nx.connected_components(G), key=len)

    effective_N = min(n, len(target_component))
    if effective_N <= 1:
        L = rd.choice(list(target_component))
        return G.subgraph([L]).copy()

    root = rd.choice(list(target_component))
    L = [rd.choice(list(target_component)) for _ in range(m)]
    sampled_nodes = set(L)

    while len(sampled_nodes) < effective_N:
        degrees = [G.degree(node) for node in L]
        total_degree = sum(degrees)

        if total_degree == 0:
            L = rd.choices(list(sampled_nodes), k=m)
            continue

        u_idx = rd.choices(range(m), weights=degrees, k=1)[0]
        u = L[u_idx]

        neighbors = list(G.neighbors(u))
        if not neighbors:
            L[u_idx] = rd.choice(list(sampled_nodes))
            continue

        v = rd.choice(neighbors)
        L[u_idx] = v
        sampled_nodes.add(v)

    return G.subgraph(sampled_nodes).copy()

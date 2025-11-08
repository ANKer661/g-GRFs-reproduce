from typing import Callable
import numpy as np
import random


def get_U_matrix(W: np.ndarray, sigma: float) -> np.ndarray:
    """
    Normalise an adjacency matrix based on its degree and a regulariser sigma.

    Args:
        W (np.ndarray): Weighted adjacency matrix of shape (n_nodes, n_nodes).
        sigma (float): Regularisation parameter.
    """
    degrees = np.sum(W, axis=1)
    U = W / np.sqrt(degrees[:, None] @ degrees[None, :])

    return U * sigma**2


# ----- Functions to do with sampling random walks. -----


def adj_matrix_to_lists(W: np.ndarray) -> tuple[list[list[int]], list[list[float]]]:
    """Get adjacency lists and weight lists for a weighted adjacency matrix"""
    n = W.shape[0]
    adj_lists = []
    weight_lists = []

    for i in range(n):
        neighbor_idx = np.nonzero(W[i, :])[0]
        weights = W[i, neighbor_idx]
        adj_lists.append(neighbor_idx.tolist())
        weight_lists.append(weights.tolist())

    return adj_lists, weight_lists


def simulate_single_walk(adj_lists: list[list[int]], start_v: int, p_halt: float) -> list[int]:
    """Do a simple random walk and get vertex history."""
    path = [start_v]  # Set starting vertex
    cur_v = start_v
    # while np.random.random() > p_halt:  # Terminate with probability p
    #     cur_v = int(np.random.choice(adj_lists[cur_v]))  # Choose a new vertex
    #     path.append(cur_v)
    while random.uniform(0, 1) > p_halt:  # Terminate with probability p
        random_index = int(random.uniform(0, 1) * len(adj_lists[cur_v]))  # Choose a new vertex
        cur_v = adj_lists[cur_v][random_index]
        path.append(cur_v)
    return path


def simulate_walks_from_v(
    adj_lists: list[list[int]], v: int, p_halt: float, n_walks: int
) -> list[list[int]]:
    """Simulate multiple random walks from a starting vertex."""
    paths = []
    for _ in range(n_walks):
        paths.append(simulate_single_walk(adj_lists, v, p_halt))
    return paths


def simulate_walks_from_all(
    adj_lists: list[list[int]], p_halt: float, n_walks: int
) -> list[list[list[int]]]:
    """
    Simulate multiple random walks from all vertices.

    Args:
        adj_lists (list[list[int]]): Adjacency lists of the graph.
        p_halt (float): Probability of halting at each step.
        n_walks (int): Number of walks to simulate from each vertex.

    Returns:
        list[list[list[int]]]: [vertex i][walk j]
            gives the j-th walk starting from vertex i.
    """
    all_walk_paths = []

    for v in range(len(adj_lists)):
        all_walk_paths.append(simulate_walks_from_v(adj_lists, v, p_halt, n_walks))

    return all_walk_paths


# ----- Functions to actually construct GRFs to approximate graph kernels -----


def create_rf_vector_from_walk_paths(
    U: np.ndarray,
    adj_lists: list[list[int]],
    p_halt: float,
    v_walk_paths: list[list[int]],
    f: Callable,
) -> np.ndarray:
    """
    Create an RF vector for a node from a list of random walks.

    Args:
        U (np.ndarray): Normalised weighted adjacency matrix.
        adj_lists (list[list[int]]): Adjacency lists of the graph.
        p_halt (float): Probability of halting at each step.
        v_walk_paths (list[list[int]]): List of random walks starting from a vertex.
        f (Callable): Modulation function.
    """

    n_walks = len(v_walk_paths)
    n_nodes = len(adj_lists)
    rf_vector = np.zeros(n_nodes)

    # Find the longest walk.
    longest_walk = max(map(len, v_walk_paths))

    # Evaluate modulation function f up to longest walk length.
    f_vec = [float(f(length)) for length in range(longest_walk)]

    # Store product of weights and marginal probabilities.
    for walk in v_walk_paths:
        weights_product = 1.0
        marginal_prob = 1.0
        for step, node in enumerate(walk):
            rf_vector[node] += (weights_product / marginal_prob) * f_vec[step]
            if step < len(walk) - 1:
                weights_product *= U[walk[step]][walk[step + 1]]
                marginal_prob *= (1 - p_halt) / len(adj_lists[node])

    # Normalise by number of walks.
    rf_vector /= n_walks

    return rf_vector


def get_random_feature(
    U: np.ndarray,
    adj_lists: list[list[int]],
    p_halt: float,
    all_walk_paths: list[list[list[int]]],
    f: Callable,
) -> np.ndarray:
    """Combine the GRFs to get a kernel estimate."""
    rf_vectors = []

    # Stack up GRF vectors for each start node.
    for v_walks_paths in all_walk_paths:
        rf_v = create_rf_vector_from_walk_paths(U, adj_lists, p_halt, v_walks_paths, f)
        rf_vectors.append(rf_v)

    A = np.asarray(rf_vectors)

    return A


def frob_norm_error(K_true: np.ndarray, K_approx: np.ndarray) -> float:
    """Compute the Frobenius norm error between two matrices."""
    return float(np.linalg.norm(K_true - K_approx, ord="fro") / np.linalg.norm(K_true, ord="fro"))

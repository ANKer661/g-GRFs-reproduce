import random
from typing import Callable

import numpy as np
import scipy
import torch
import torch.nn as nn
from numba import njit
from scipy.special import gammaln


class NeuralModulationFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 1, bias=True),
            nn.ReLU(),
            nn.Linear(1, 1, bias=True),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GroundtruthKernels:
    """Compute the groundtruth kernel evaluations."""

    def __init__(self, sigma: float, alpha: float) -> None:
        self.functions = {
            1: self.d_regularised_1,
            2: self.d_regularised_2,
            3: self.p_step_rw_2,
            4: self.diffusion,
            5: self.cosine,
        }

        self.sigma = sigma
        self.alpha = alpha

    def d_regularised_1(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        d regularised Lapacian kernel with d=1. (I - W)^{-1}
        """
        E = np.eye(U.shape[0])
        U = self.sigma**2 / (1 + self.sigma**2) * U
        return np.linalg.inv((E - U)), U

    def d_regularised_2(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """d regularised Lapacian kernel with d=2. (I - W)^{-2}"""
        E = np.eye(U.shape[0])
        U = self.sigma**2 / (1 + self.sigma**2) * U
        I_minus_U = E - U
        return np.linalg.inv(I_minus_U @ I_minus_U), U

    def p_step_rw_2(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """p step RW kernel with d=2, (I + W)^2"""
        E = np.eye(U.shape[0])
        U = U / (self.alpha - 1)
        I_plus_U = E + U
        return I_plus_U @ I_plus_U, U

    def diffusion(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Diffusion kernel, exp(W)"""
        U = self.sigma**2 * U / 2
        return scipy.linalg.expm(U), U

    def cosine(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cosine kernel, sqrt{2}cos(pi/4-U) = cos(U) + sin(U)."""
        U = self.sigma**2 * U
        return scipy.linalg.sinm(U) + scipy.linalg.cosm(U), U

    def get_groundtruth_kernel(self, func_type: int, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the result of the modulation function of the given type.

        Args:
            func_type (int): The type of modulation (1 to 5).
            U (np.ndarray): The input matrix.
        """
        if func_type not in self.functions:
            raise ValueError(f"Invalid function type {func_type}. Must be between 1 and 5.")
        return self.functions[func_type](U)


class ModulationFunctions:
    """
    Modulation functions to generate GRFs, based on inverse convolution of Taylor expansion.
    """

    def __init__(self) -> None:
        self.functions = {
            1: self.d_regularised_1,
            2: self.d_regularised_2,
            3: self.p_step_rw_2,
            4: self.diffusion,
            5: self.cosine,
        }

    def d_regularised_1(self, x: int) -> float:
        if x == 0:
            return 1
        else:
            return scipy.special.factorial2(2 * x - 1) / (scipy.special.factorial2(2 * x))

    def d_regularised_2(self, x: int) -> float:
        return 1

    def p_step_rw_2(self, x: int) -> float:
        return scipy.special.binom(1, x)

    def diffusion(self, x: int) -> float:
        # 1 / (2**x * scipy.special.factorial(x))
        # Use log to avoid overflow
        log_val = -(x * np.log(2) + gammaln(x + 1))
        return np.exp(log_val)

    def alpha_func_cosine(self, k: int) -> float:
        return (-1) ** (k // 2) / scipy.special.factorial(k)

    def get_next_f_cosine(self, g_eval: float) -> float:
        """Helper function for computing next f function evaluation."""

        f0 = self.f_list[0]
        f1 = self.f_list[1:]
        f1_np = np.asarray(f1)
        f1r_np = f1_np[::-1]
        f1dot = np.dot(f1_np, f1r_np)
        fnext = (g_eval - f1dot) / (2 * f0)
        self.f_list.append(fnext)

        return fnext

    def cosine(self, x: int) -> float:
        """
        Here, there isn't a convenient closed form so we use the iterative formula in Eq. 6
        * Optimized with caching *
        """
        if not hasattr(self, "f_list"):
            self.f_list = [1.0]
        if x < len(self.f_list):
            return self.f_list[x]
        else:
            max_known = len(self.f_list) - 1
            for i in range(max_known, x):
                self.get_next_f_cosine(self.alpha_func_cosine(i + 1))
            return self.f_list[-1]


def get_U_matrix(W: np.ndarray) -> np.ndarray:
    """
    Normalise an adjacency matrix based on its degree and a regulariser sigma.

    Args:
        W (np.ndarray): Weighted adjacency matrix of shape (n_nodes, n_nodes).
    """
    degrees = np.sum(W, axis=1)
    U = W / np.sqrt(degrees[:, None] @ degrees[None, :])

    return U


# ----- Functions to do with sampling random walks. -----


def adj_matrix_to_lists(W: np.ndarray | torch.Tensor) -> tuple[list[list[int]], list[list[float]]]:
    """Get adjacency lists and weight lists for a weighted adjacency matrix"""

    assert isinstance(W, np.ndarray) or isinstance(W, torch.Tensor), (
        "W must be a numpy array or torch tensor."
    )
    n = W.shape[0]
    adj_lists = []
    weight_lists = []

    for i in range(n):
        if isinstance(W, torch.Tensor):
            neighbor_idx = torch.nonzero(W[i, :]).squeeze(-1).tolist()
            weights = W[i, neighbor_idx].tolist()
        else:
            neighbor_idx = np.nonzero(W[i, :])[0].tolist()
            weights = W[i, neighbor_idx].tolist()
        adj_lists.append(neighbor_idx)
        weight_lists.append(weights)

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


@njit(fastmath=True, cache=True)
def create_rf_vector(
    U: np.ndarray,
    degrees: np.ndarray,
    p_halt: float,
    walks_flat: np.ndarray,
    walk_starts: np.ndarray,
    walk_ends: np.ndarray,
    f_vec: np.ndarray,
) -> np.ndarray:
    """
    Create an RF vector for a node from a list of random walks.

    Args:
        U (np.ndarray): Normalised weighted adjacency matrix.
        degrees (np.ndarray): Degrees of the nodes.
        p_halt (float): Probability of halting at each step.
        walks_flat (np.ndarray): Flattened array of all walks.
        walk_starts (np.ndarray): Start indices of each walk in walks_flat.
        walk_ends (np.ndarray): End indices of each walk in walks_flat.
        f_vec (np.ndarray): Precomputed modulation function evaluations.
    """
    n_nodes = U.shape[0]
    n_walks = len(walk_starts)
    rf_vector = np.zeros(n_nodes, dtype=np.float64)
    p_continue = 1.0 - p_halt

    for walk_idx in range(n_walks):
        start = walk_starts[walk_idx]
        end = walk_ends[walk_idx]

        weights_product = 1.0
        marginal_prob = 1.0

        for pos in range(start, end):
            node = walks_flat[pos]
            step = pos - start

            rf_vector[node] += (weights_product / marginal_prob) * f_vec[step]

            if pos < end - 1:
                next_node = walks_flat[pos + 1]
                weights_product *= U[node, next_node]
                marginal_prob *= p_continue / degrees[node]

    rf_vector /= n_walks
    return rf_vector


def flatten_walks(v_walks_paths: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten random walk lists into 3 1-d arrays for numba.
    """
    walks_flat = []
    walk_starts = []
    walk_ends = []

    pos = 0
    for walk in v_walks_paths:
        walk_starts.append(pos)
        walks_flat.extend(walk)
        pos += len(walk)
        walk_ends.append(pos)

    return (
        np.array(walks_flat, dtype=np.int32),
        np.array(walk_starts, dtype=np.int32),
        np.array(walk_ends, dtype=np.int32),
    )


def get_random_feature(
    U: np.ndarray,
    adj_lists: list[list[int]],
    p_halt: float,
    all_walk_paths: list[list[list[int]]],
    f: Callable,
) -> np.ndarray:
    """Combine the GRFs to get a kernel estimate."""
    n_nodes = len(adj_lists)

    # precompute modulation func and degrees
    longest_walk = max(len(walk) for v_walks in all_walk_paths for walk in v_walks)
    f_vec = np.array([float(f(length)) for length in range(longest_walk)], dtype=np.float64)
    degrees = np.array([len(neighbors) for neighbors in adj_lists], dtype=np.float64)

    rf_matrix = np.empty((n_nodes, n_nodes), dtype=np.float64)
    # process each node
    for idx, v_walks_paths in enumerate(all_walk_paths):
        walks_flat, walk_starts, walk_ends = flatten_walks(v_walks_paths)
        rf_matrix[idx] = create_rf_vector(
            U, degrees, p_halt, walks_flat, walk_starts, walk_ends, f_vec
        )

    return rf_matrix


def create_rf_vector_t(
    log_U: torch.Tensor,
    log_degrees: torch.Tensor,
    log_p_continue: torch.Tensor,
    v_walk_paths: list[list[int]],
    f_vec: torch.Tensor,
) -> torch.Tensor:
    """
    Create an RF vector for a node from a list of random walks.

    Args:
        U (torch.Tensor): Normalised weighted adjacency matrix.
        adj_lists (list[list[int]]): Adjacency lists of the graph.
        p_halt (float): Probability of halting at each step.
        v_walk_paths (list[list[int]]): List of random walks starting from a vertex.
        f_vec (torch.Tensor): Precomputed modulation function evaluations.
    """
    dtype = log_U.dtype
    device = log_U.device
    n_walks = len(v_walk_paths)
    n_nodes = log_U.size(0)

    rf_vector = torch.zeros(n_nodes, device=device, dtype=dtype)
    # working in log space to avoid numerical issues
    for walk in v_walk_paths:
        log_weights_product = torch.tensor(0.0, dtype=dtype, device=device)
        log_marginal_prob = torch.tensor(0.0, dtype=dtype, device=device)

        for step, node in enumerate(walk):
            log_load = log_weights_product - log_marginal_prob
            load = torch.exp(log_load)

            rf_vector[node] += load * f_vec[step]

            if step < len(walk) - 1:
                next_node = walk[step + 1]
                log_weights_product += log_U[node, next_node]
                log_marginal_prob += log_p_continue - log_degrees[node]

    rf_vector = rf_vector / n_walks
    return rf_vector


def get_random_feature_t(
    U: torch.Tensor,
    adj_lists: list[list[int]],
    p_halt: float,
    all_walk_paths: list[list[list[int]]],
    f: torch.nn.Module,
) -> torch.Tensor:
    n_nodes = len(adj_lists)
    dtype = U.dtype
    device = U.device

    # evaluate modulation function f up to longest walk length.
    max_length = max(len(walk) for v_walks in all_walk_paths for walk in v_walks)
    steps_tensor = torch.arange(max_length, dtype=torch.float32, device=U.device).unsqueeze(-1)
    f_vec = f(steps_tensor).squeeze(-1)

    # precompute all constants
    degrees = [len(neighbors) for neighbors in adj_lists]
    log_degrees = torch.log(torch.tensor(degrees, dtype=dtype, device=device) + 1e-10)
    log_p_continue = torch.log(torch.tensor(1.0 - p_halt, dtype=dtype, device=device))
    log_U = torch.log(U + 1e-10)

    # preallocate output matrix
    A = torch.empty((n_nodes, n_nodes), dtype=dtype, device=device)
    for i, v_walks_paths in enumerate(all_walk_paths):
        A[i] = create_rf_vector_t(log_U, log_degrees, log_p_continue, v_walks_paths, f_vec)

    return A


def frob_norm_error(K_true: np.ndarray | torch.Tensor, K_approx: np.ndarray | torch.Tensor) -> float:
    """Compute the Frobenius norm error between two matrices."""
    if isinstance(K_true, torch.Tensor):
        K_true = K_true.cpu().numpy()
    if isinstance(K_approx, torch.Tensor):
        K_approx = K_approx.cpu().numpy()
    return float(np.linalg.norm(K_true - K_approx, ord="fro") / np.linalg.norm(K_true, ord="fro"))

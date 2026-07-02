"""NumPy reference implementations for the chunked decompositions.

Stream does not execute workloads; these references validate that each *chunked* decomposition (the
algorithm the rewritten subgraph mirrors) is numerically equal to the *direct* recurrence. The tests
in ``tests/rewrites`` assert ``chunked(...) ≈ direct(...)`` for random shapes and chunk sizes.

All arrays are float64. Decays are expected in ``(0, 1]``; the chunked scan divides by a cumulative
decay product, so very small decays with long chunks lose precision (as in any parallel-scan form).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Mamba1-style diagonal selective scan                                       #
# --------------------------------------------------------------------------- #
def direct_diagonal_scan(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """h[t] = a[t] * h[t-1] + b[t] * x[t], h[-1] = 0. All inputs shape (L, D); returns H (L, D)."""
    length, dim = x.shape
    h = np.zeros((length, dim))
    prev = np.zeros(dim)
    for t in range(length):
        prev = a[t] * prev + b[t] * x[t]
        h[t] = prev
    return h


def chunked_diagonal_scan(x: np.ndarray, a: np.ndarray, b: np.ndarray, chunk_size: int) -> np.ndarray:
    """Chunked parallel form of :func:`direct_diagonal_scan`: dense intra-chunk, state carried across chunks."""
    length, dim = x.shape
    h = np.zeros((length, dim))
    carry = np.zeros(dim)
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        a_c, b_c, x_c = a[start:end], b[start:end], x[start:end]
        cum_decay = np.cumprod(a_c, axis=0)  # A[i] = prod_{0..i} a_c
        contribution = b_c * x_c
        # h[i] = A[i]*carry + A[i] * cumsum(contribution / A)[i]   (segment-sum identity)
        local = cum_decay * carry + cum_decay * np.cumsum(contribution / cum_decay, axis=0)
        h[start:end] = local
        carry = local[-1]
    return h


# --------------------------------------------------------------------------- #
#  Mamba2 SSD (scalar decay) -- quadratic <-> chunked-linear duality          #
# --------------------------------------------------------------------------- #
def direct_ssd(x: np.ndarray, a: np.ndarray, b_mat: np.ndarray, c_mat: np.ndarray) -> np.ndarray:
    """Quadratic SSD: y[t] = sum_{s<=t} decay(t,s) * (C[t] . B[s]) * x[s].

    x (L,), a (L,) scalar decay per step, B/C (L, N). Returns y (L,).
    """
    length, _ = b_mat.shape
    log_decay = np.zeros(length)
    for t in range(1, length):
        log_decay[t] = log_decay[t - 1] + np.log(a[t])
    y = np.zeros(length)
    for t in range(length):
        for s in range(t + 1):
            y[t] += np.exp(log_decay[t] - log_decay[s]) * (c_mat[t] @ b_mat[s]) * x[s]
    return y


def chunked_ssd(x: np.ndarray, a: np.ndarray, b_mat: np.ndarray, c_mat: np.ndarray, chunk_size: int) -> np.ndarray:
    """Chunked-linear SSD: state h[t] = a[t] h[t-1] + B[t] x[t] via the chunked scan, then y = <C, h>."""
    length, state = b_mat.shape
    h = chunked_diagonal_scan(
        b_mat * x[:, None],
        np.repeat(a[:, None], state, axis=1),
        np.ones((length, state)),
        chunk_size,
    )
    return np.sum(c_mat * h, axis=1)


# --------------------------------------------------------------------------- #
#  Gated DeltaNet (matrix state)                                              #
# --------------------------------------------------------------------------- #
def direct_deltanet(q: np.ndarray, k: np.ndarray, v: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T ; y_t = S_t^T q_t.

    q/k (L, dk), v (L, dv), alpha/beta (L,). Returns y (L, dv).
    """
    length, dk = k.shape
    dv = v.shape[1]
    state = np.zeros((dk, dv))
    identity = np.eye(dk)
    y = np.zeros((length, dv))
    for t in range(length):
        state = alpha[t] * (identity - beta[t] * np.outer(k[t], k[t])) @ state + beta[t] * np.outer(k[t], v[t])
        y[t] = state.T @ q[t]
    return y


def chunked_deltanet(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, alpha: np.ndarray, beta: np.ndarray, chunk_size: int
) -> np.ndarray:
    """Chunked DeltaNet: the matrix state is carried across chunk boundaries (dense intra-chunk in the graph)."""
    length, dk = k.shape
    dv = v.shape[1]
    state = np.zeros((dk, dv))
    identity = np.eye(dk)
    y = np.zeros((length, dv))
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        for t in range(start, end):
            state = alpha[t] * (identity - beta[t] * np.outer(k[t], k[t])) @ state + beta[t] * np.outer(k[t], v[t])
            y[t] = state.T @ q[t]
    return y

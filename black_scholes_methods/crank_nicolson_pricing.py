from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrankNicolsonResult:
    price: float
    S_grid: np.ndarray
    t_grid: np.ndarray
    V: np.ndarray


def _thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Solve a tridiagonal linear system using the Thomas algorithm.
    lower: length n-1
    diag:  length n
    upper: length n-1
    rhs:   length n
    """
    n = len(diag)
    c_star = np.zeros(n - 1, dtype=float)
    d_star = np.zeros(n, dtype=float)

    c_star[0] = upper[0] / diag[0]
    d_star[0] = rhs[0] / diag[0]

    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c_star[i - 1]
        c_star[i] = upper[i] / denom
        d_star[i] = (rhs[i] - lower[i - 1] * d_star[i - 1]) / denom

    d_star[-1] = (rhs[-1] - lower[-1] * d_star[-2]) / (diag[-1] - lower[-1] * c_star[-2])

    x = np.zeros(n, dtype=float)
    x[-1] = d_star[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]

    return x


def crank_nicolson_european_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    S_max: float,
    M: int = 400,
    N: int = 400,
    option_type: str = "call",
) -> CrankNicolsonResult:
    """
    Crank-Nicolson solver for the Black-Scholes PDE.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0.0, S_max, M + 1)
    t_grid = np.linspace(0.0, T, N + 1)

    V = np.zeros((N + 1, M + 1), dtype=float)

    # Terminal condition
    if option_type == "call":
        V[-1, :] = np.maximum(S_grid - K, 0.0)
    else:
        V[-1, :] = np.maximum(K - S_grid, 0.0)

    i_vals = np.arange(1, M, dtype=float)

    alpha = 0.25 * dt * (sigma * sigma * i_vals * i_vals - r * i_vals)
    beta = -0.5 * dt * (sigma * sigma * i_vals * i_vals + r)
    gamma = 0.25 * dt * (sigma * sigma * i_vals * i_vals + r * i_vals)

    # Left-hand matrix coefficients
    lower_A = -alpha[1:]
    diag_A = 1.0 - beta
    upper_A = -gamma[:-1]

    # Right-hand matrix coefficients
    lower_B = alpha[1:]
    diag_B = 1.0 + beta
    upper_B = gamma[:-1]

    for n in range(N - 1, -1, -1):
        tau_now = T - t_grid[n]
        tau_next = T - t_grid[n + 1]

        if option_type == "call":
            left_now = 0.0
            right_now = S_max - K * np.exp(-r * tau_now)
            left_next = 0.0
            right_next = S_max - K * np.exp(-r * tau_next)
        else:
            left_now = K * np.exp(-r * tau_now)
            right_now = 0.0
            left_next = K * np.exp(-r * tau_next)
            right_next = 0.0

        V[n, 0] = left_now
        V[n, -1] = right_now

        rhs = np.zeros(M - 1, dtype=float)

        interior_next = V[n + 1, 1:M]

        rhs[:] = diag_B * interior_next
        rhs[1:] += lower_B * interior_next[:-1]
        rhs[:-1] += upper_B * interior_next[1:]

        # Boundary adjustments
        rhs[0] += alpha[0] * left_next + alpha[0] * left_now
        rhs[-1] += gamma[-1] * right_next + gamma[-1] * right_now

        V[n, 1:M] = _thomas_solve(lower_A, diag_A, upper_A, rhs)

    price = float(np.interp(S0, S_grid, V[0, :]))

    return CrankNicolsonResult(
        price=price,
        S_grid=S_grid,
        t_grid=t_grid,
        V=V,
    )
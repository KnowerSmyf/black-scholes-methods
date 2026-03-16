from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiniteDifferenceResult:
    price: float
    S_grid: np.ndarray
    t_grid: np.ndarray
    V: np.ndarray


def explicit_fd_european_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    S_max: float,
    M: int = 400,
    N: int = 4000,
    option_type: str = "call",
) -> FiniteDifferenceResult:
    """
    Explicit finite-difference solver for the Black-Scholes PDE.

    Space grid:
        i = 0, ..., M
    Time grid:
        n = 0, ..., N
    We march backward from maturity T to 0.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0.0, S_max, M + 1)
    t_grid = np.linspace(0.0, T, N + 1)

    V = np.zeros((N + 1, M + 1), dtype=float)

    # Terminal payoff at maturity
    if option_type == "call":
        V[-1, :] = np.maximum(S_grid - K, 0.0)
    else:
        V[-1, :] = np.maximum(K - S_grid, 0.0)

    # Backward induction
    for n in range(N - 1, -1, -1):
        tau = T - t_grid[n]

        # Boundary conditions
        if option_type == "call":
            V[n, 0] = 0.0
            V[n, -1] = S_max - K * np.exp(-r * tau)
        else:
            V[n, 0] = K * np.exp(-r * tau)
            V[n, -1] = 0.0

        for i in range(1, M):
            a = 0.5 * dt * (sigma * sigma * i * i - r * i)
            b = 1.0 - dt * (sigma * sigma * i * i + r)
            c = 0.5 * dt * (sigma * sigma * i * i + r * i)

            V[n, i] = a * V[n + 1, i - 1] + b * V[n + 1, i] + c * V[n + 1, i + 1]

    price = float(np.interp(S0, S_grid, V[0, :]))

    return FiniteDifferenceResult(
        price=price,
        S_grid=S_grid,
        t_grid=t_grid,
        V=V,
    )
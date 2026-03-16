from __future__ import annotations

import numpy as np


def simulate_gbm_terminal(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Exact risk-neutral sampling of terminal GBM prices:
        S_T = S_0 * exp((r - 0.5 sigma^2) T + sigma sqrt(T) Z)
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    drift = (r - 0.5 * sigma * sigma) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def simulate_gbm_paths(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_sims: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact stepwise GBM path simulation.
    Returns:
        times: shape (n_steps + 1,)
        paths: shape (n_sims, n_steps + 1)
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if T < 0:
        raise ValueError("T must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive.")

    dt = T / n_steps
    rng = np.random.default_rng(seed)

    z = rng.standard_normal((n_sims, n_steps))
    increments = (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z

    log_paths = np.empty((n_sims, n_steps + 1), dtype=float)
    log_paths[:, 0] = np.log(S0)
    log_paths[:, 1:] = log_paths[:, [0]] + np.cumsum(increments, axis=1)

    paths = np.exp(log_paths)
    times = np.linspace(0.0, T, n_steps + 1)
    return times, paths
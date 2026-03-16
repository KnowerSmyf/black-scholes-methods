from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .gbm_simulation import simulate_gbm_terminal


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    std_error: float
    discounted_payoffs: np.ndarray


def monte_carlo_european_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int = 100_000,
    option_type: str = "call",
    seed: int | None = None,
) -> MonteCarloResult:
    option_type = option_type.lower()

    ST = simulate_gbm_terminal(
        S0=S0,
        T=T,
        r=r,
        sigma=sigma,
        n_sims=n_sims,
        seed=seed,
    )

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    discounted_payoffs = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted_payoffs))
    std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_sims))

    return MonteCarloResult(
        price=price,
        std_error=std_error,
        discounted_payoffs=discounted_payoffs,
    )
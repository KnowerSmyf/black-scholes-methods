from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from black_scholes_methods import black_scholes_price, monte_carlo_european_option

def main() -> None:
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    true_price = black_scholes_price(S0, K, T, r, sigma, option_type="call")

    sim_counts = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    errors = []
    std_errors = []

    for n_sims in sim_counts:
        result = monte_carlo_european_option(
            S0=S0,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            n_sims=n_sims,
            option_type="call",
            seed=42,
        )
        errors.append(abs(result.price - true_price))
        std_errors.append(result.std_error)

    plt.figure(figsize=(8, 5))
    plt.loglog(sim_counts, errors, marker="o", label="Absolute pricing error")
    plt.loglog(sim_counts, std_errors, marker="s", label="MC standard error")
    plt.xlabel("Number of simulations")
    plt.ylabel("Error")
    plt.title("Monte Carlo convergence for Black-Scholes European call")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/monte_carlo_convergence.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

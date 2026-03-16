from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from black_scholes_methods import (
    black_scholes_price,
    crank_nicolson_european_option,
    monte_carlo_european_option,
)


def main() -> None:
    S0 = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    S_max = 400.0

    strikes = np.arange(60, 141, 10, dtype=float)

    analytic_prices = []
    mc_prices = []
    cn_prices = []

    for K in strikes:
        analytic_prices.append(
            black_scholes_price(S0, K, T, r, sigma, option_type="call")
        )
        mc_prices.append(
            monte_carlo_european_option(
                S0, K, T, r, sigma, n_sims=100_000, option_type="call", seed=42
            ).price
        )
        cn_prices.append(
            crank_nicolson_european_option(
                S0, K, T, r, sigma, S_max=S_max, M=400, N=400, option_type="call"
            ).price
        )

    plt.figure(figsize=(9, 5))
    plt.plot(strikes, analytic_prices, marker="o", label="Analytic")
    plt.plot(strikes, mc_prices, marker="s", label="Monte Carlo")
    plt.plot(strikes, cn_prices, marker="^", label="Crank-Nicolson")

    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("European Call Price vs Strike Under Black-Scholes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/strike_sweep.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
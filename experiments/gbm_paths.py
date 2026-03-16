from __future__ import annotations

import matplotlib.pyplot as plt

from black_scholes_methods.gbm_simulation import simulate_gbm_paths


def main() -> None:
    S0 = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    n_steps = 252
    n_sims = 50

    times, paths = simulate_gbm_paths(
        S0=S0,
        T=T,
        r=r,
        sigma=sigma,
        n_steps=n_steps,
        n_sims=n_sims,
        seed=42,
    )

    plt.figure(figsize=(9, 5))
    for i in range(n_sims):
        plt.plot(times, paths[i], alpha=0.8)

    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Simulated GBM Paths Under the Risk-Neutral Measure")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/gbm_paths.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
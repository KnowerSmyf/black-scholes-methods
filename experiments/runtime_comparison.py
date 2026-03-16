from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt

from black_scholes_methods import (
    black_scholes_price,
    crank_nicolson_european_option,
    monte_carlo_european_option,
    explicit_fd_european_option,
)

def timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def main() -> None:
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    S_max = 4 * K

    methods = []
    runtimes = []

    _, t_analytic = timed_call(black_scholes_price, S0, K, T, r, sigma, "call")
    methods.append("Analytic")
    runtimes.append(t_analytic)

    _, t_mc = timed_call(
        monte_carlo_european_option, S0, K, T, r, sigma, 200_000, "call", 42
    )
    methods.append("Monte Carlo")
    runtimes.append(t_mc)

    _, t_fd = timed_call(
        explicit_fd_european_option, S0, K, T, r, sigma, S_max, 400, 4000, "call"
    )
    methods.append("Explicit FD")
    runtimes.append(t_fd)

    _, t_cn = timed_call(
        crank_nicolson_european_option, S0, K, T, r, sigma, S_max, 400, 400, "call"
    )
    methods.append("Crank-Nicolson")
    runtimes.append(t_cn)

    plt.figure(figsize=(8, 5))
    plt.bar(methods, runtimes)
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime comparison of Black-Scholes pricing methods")
    plt.tight_layout()
    plt.savefig("plots/runtime_comparison.png", dpi=200)
    plt.show()

    for method, runtime in zip(methods, runtimes):
        print(f"{method:20s}: {runtime:.6f}s")


if __name__ == "__main__":
    main()
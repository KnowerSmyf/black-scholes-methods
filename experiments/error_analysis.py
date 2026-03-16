from __future__ import annotations
import pandas as pd

from black_scholes_methods import (
    black_scholes_price,
    crank_nicolson_european_option,
    monte_carlo_european_option,
    explicit_fd_european_option,
)

def main() -> None:
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    S_max = 4 * K

    analytic = black_scholes_price(S0, K, T, r, sigma, option_type="call")

    mc = monte_carlo_european_option(S0, K, T, r, sigma, n_sims=200_000, seed=42)
    fd = explicit_fd_european_option(S0, K, T, r, sigma, S_max=S_max, M=400, N=4000)
    cn = crank_nicolson_european_option(S0, K, T, r, sigma, S_max=S_max, M=400, N=400)

    rows = [
        ("Analytic Black-Scholes", analytic, 0.0),
        ("Monte Carlo", mc.price, abs(mc.price - analytic)),
        ("Explicit Finite Difference", fd.price, abs(fd.price - analytic)),
        ("Crank-Nicolson", cn.price, abs(cn.price - analytic)),
    ]

    df = pd.DataFrame(rows, columns=["Method", "Price", "Absolute Error"])
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
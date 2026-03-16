from black_scholes_methods import (
    black_scholes_price,
    monte_carlo_european_option,
    explicit_fd_european_option,
    crank_nicolson_european_option,
)

def main():
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2

    analytic = black_scholes_price(S0, K, T, r, sigma)

    mc = monte_carlo_european_option(S0, K, T, r, sigma, 100_000)
    fd = explicit_fd_european_option(S0, K, T, r, sigma, S_max=400)
    cn = crank_nicolson_european_option(S0, K, T, r, sigma, S_max=400)

    print("Analytic:", analytic)
    print("Monte Carlo:", mc.price)
    print("Finite Difference:", fd.price)
    print("Crank-Nicolson:", cn.price)


if __name__ == "__main__":
    main()
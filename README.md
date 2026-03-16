# Black-Scholes Methods

A small quantitative finance project comparing multiple approaches to pricing European options under the Black-Scholes framework.

## Methods implemented

- Analytical Black-Scholes formula
- Monte Carlo pricing under risk-neutral GBM
- Explicit finite-difference solution of the Black-Scholes PDE
- Crank-Nicolson finite-difference scheme

## Mathematical background

Under the risk-neutral measure, the stock price follows:

$$dS_t = r S_t dt + \sigma S_t dW_t$$

This implies the discounted stock price is a martingale, and European derivative prices can be written as discounted conditional expectations of their terminal payoff.

The same pricing problem can also be expressed as the Black-Scholes PDE, giving a bridge between:

- stochastic differential equations
- martingale pricing
- numerical PDE methods

## Repo goals

- Compare pricing accuracy across methods
- Compare runtime and convergence
- Demonstrate the link between Monte Carlo and PDE-based pricing
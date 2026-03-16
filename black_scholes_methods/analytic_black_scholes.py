from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt


def norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@dataclass(frozen=True)
class BlackScholesResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    d1: float
    d2: float


def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError("Spot price S must be positive.")
    if K <= 0:
        raise ValueError("Strike K must be positive.")
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative.")
    if sigma < 0:
        raise ValueError("Volatility sigma must be non-negative.")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T == 0 or sigma == 0:
        # Degenerate edge case; d1/d2 not meaningful in the usual way.
        intrinsic = log(S / K) if S != K else 0.0
        huge = 1e12
        sign = 1.0 if intrinsic >= 0 else -1.0
        return sign * huge, sign * huge

    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    _validate_inputs(S, K, T, r, sigma)
    option_type = option_type.lower()

    if T == 0:
        if option_type == "call":
            return max(S - K, 0.0)
        if option_type == "put":
            return max(K - S, 0.0)
        raise ValueError("option_type must be 'call' or 'put'.")

    if sigma == 0:
        discounted_strike = K * exp(-r * T)
        if option_type == "call":
            return max(S - discounted_strike, 0.0)
        if option_type == "put":
            return max(discounted_strike - S, 0.0)
        raise ValueError("option_type must be 'call' or 'put'.")

    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    if option_type == "put":
        return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    raise ValueError("option_type must be 'call' or 'put'.")


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> BlackScholesResult:
    _validate_inputs(S, K, T, r, sigma)
    option_type = option_type.lower()

    price = black_scholes_price(S, K, T, r, sigma, option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if T == 0 or sigma == 0:
        # Keep edge-case handling simple.
        return BlackScholesResult(
            price=price,
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            rho=0.0,
            d1=d1,
            d2=d2,
        )

    pdf_d1 = norm_pdf(d1)
    sqrt_T = sqrt(T)

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T

    if option_type == "call":
        delta = norm_cdf(d1)
        theta = (
            -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * exp(-r * T) * norm_cdf(d2)
        )
        rho = K * T * exp(-r * T) * norm_cdf(d2)
    elif option_type == "put":
        delta = norm_cdf(d1) - 1.0
        theta = (
            -(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * exp(-r * T) * norm_cdf(-d2)
        )
        rho = -K * T * exp(-r * T) * norm_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return BlackScholesResult(
        price=price,
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        d1=d1,
        d2=d2,
    )
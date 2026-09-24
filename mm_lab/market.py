"""Synthetic option prices, customer arrivals, and lagged public features."""

from dataclasses import dataclass
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd


FEATURES = [
    "abs_imbalance",
    "public_volume",
    "public_trade_rate",
    "premium_volatility",
    "mean_public_markout",
    "mean_abs_change",
]


@dataclass(frozen=True)
class Config:
    steps: int = 390
    initial_cash: float = 100_000.0

    inventory_limit: int = 8
    quote_size: int = 3
    multiplier: int = 100

    half_spread: float = 0.03
    tick: float = 0.01

    # Dollars per contract per execution.
    fee: float = 0.65

    hedge_cost_per_share: float = 0.005
    external_half_spread: float = 0.04

    # Error in the informed customer's estimate of the next premium.
    signal_noise: float = 0.04

    low_informed: float = 0.05
    high_informed: float = 0.75

    # Hidden-regime transition probabilities.
    enter_high: float = 0.025
    exit_high: float = 0.10

    jump_scale: float = 1.0
    kind: str = "call"
    hedge: bool = True
    window: int = 20

    def __post_init__(self):
        if type(self.steps) is not int or not 20 <= self.steps <= 5000:
            raise ValueError("steps must be an integer from 20 to 5000")

        if self.kind not in ("call", "put"):
            raise ValueError("kind must be call or put")

        for name in (
            "low_informed",
            "high_informed",
            "enter_high",
            "exit_high",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        for name in (
            "initial_cash",
            "half_spread",
            "tick",
            "external_half_spread",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

        for name in (
            "fee",
            "hedge_cost_per_share",
            "signal_noise",
            "jump_scale",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")

        for name in (
            "inventory_limit",
            "quote_size",
            "multiplier",
            "window",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def option_value(spot, strike, years, sigma, kind="call"):
    """European Black-Scholes price and delta with zero rates/dividends."""
    if min(spot, strike, years, sigma) <= 0:
        raise ValueError("Positive inputs required")

    if kind not in ("call", "put"):
        raise ValueError("Unknown option kind")

    def normal_cdf(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    d1 = (
        log(spot / strike) + 0.5 * sigma * sigma * years
    ) / (sigma * sqrt(years))

    d2 = d1 - sigma * sqrt(years)

    call_price = (
        spot * normal_cdf(d1)
        - strike * normal_cdf(d2)
    )

    if kind == "call":
        return max(call_price, 0.0), normal_cdf(d1)

    put_price = call_price - spot + strike
    put_delta = normal_cdf(d1) - 1

    return max(put_price, 0.0), put_delta


def past_features(flow, sizes, moves, markouts, window):
    """Build features using completed previous rounds only."""
    recent_flow = np.asarray(flow[-window:], dtype=float)

    if len(recent_flow) == 0:
        return np.zeros(len(FEATURES))

    recent_sizes = np.asarray(sizes[-window:], dtype=float)
    recent_moves = np.asarray(moves[-window:], dtype=float)

    imbalance = abs(
        np.sum(recent_flow * recent_sizes)
    ) / max(recent_sizes.sum(), 1)

    return np.array([
        imbalance,
        recent_sizes.mean(),
        np.mean(recent_flow != 0),
        recent_moves.std(),
        np.mean(markouts[-window:]),
        np.abs(recent_moves).mean(),
    ])


def generate_tape(seed, cfg):
    """Generate an exogenous market session.

    Future prices, private signals, and customer types belong to the
    simulator. The market-maker's quoting function never receives them.

    Public flow comes from a synthetic external reference venue. Our
    dealer does not affect that venue or the underlying price path.
    """
    streams = [
        np.random.default_rng(child)
        for child in np.random.SeedSequence(seed).spawn(4)
    ]

    price_rng, regime_rng, customer_rng, signal_rng = streams

    spot = 100.0
    iv = 0.22
    regime = 0
    dt = 1 / (252 * 390)

    flow = []
    sizes = []
    moves = []
    markouts = []
    rows = []

    for step in range(cfg.steps):
        # Compute these before observing the current customer or price move.
        features = past_features(
            flow, sizes, moves, markouts, cfg.window
        )

        if regime == 0:
            regime = int(regime_rng.random() < cfg.enter_high)
        else:
            regime = int(
                not (regime_rng.random() < cfg.exit_high)
            )

        years = 30 / 365 - step * dt

        fair, delta = option_value(
            spot, 100, years, iv, cfg.kind
        )

        realized_vol = 0.22 * (3 if regime else 1)
        shock = price_rng.normal()

        jump_probability = 0.12 if regime else 0.002

        if price_rng.random() < jump_probability:
            jump = price_rng.normal(0, 0.004 * cfg.jump_scale)
        else:
            jump = 0.0

        next_spot = spot * exp(
            -0.5 * realized_vol**2 * dt
            + realized_vol * sqrt(dt) * shock
            + jump
        )

        target_iv = 0.40 if regime else 0.22

        next_iv = float(np.clip(
            iv
            + 0.08 * (target_iv - iv)
            + price_rng.normal(0, 0.002),
            0.08,
            0.90,
        ))

        future, next_delta = option_value(
            next_spot, 100, years - dt, next_iv, cfg.kind
        )

        informed_probability = (
            cfg.high_informed if regime else cfg.low_informed
        )

        informed = (
            customer_rng.random() < informed_probability
        )

        noise_side = 1 if customer_rng.random() < 0.5 else -1
        willingness = float(customer_rng.exponential(0.10))
        quantity = int(customer_rng.integers(1, 5))

        # Noisy information about the next premium, visible only to
        # an informed customer.
        signal = future + signal_rng.normal(
            0, cfg.signal_noise
        )

        # External reference-venue activity.
        if informed:
            if signal > fair + cfg.half_spread:
                public_side = 1
            elif signal < fair - cfg.half_spread:
                public_side = -1
            else:
                public_side = 0
        else:
            public_side = (
                noise_side
                if willingness >= cfg.half_spread
                else 0
            )

        public_quantity = quantity if public_side else 0

        row = {
            "step": step,
            "spot": spot,
            "next_spot": next_spot,
            "iv": iv,
            "fair": fair,
            "future": future,
            "delta": delta,
            "next_delta": next_delta,
            "informed": int(informed),
            "high_regime": regime,
            "signal": signal,
            "noise_side": noise_side,
            "willingness": willingness,
            "quantity": quantity,
            "public_side": public_side,
            "public_quantity": public_quantity,
        }

        row.update(dict(zip(FEATURES, features)))
        rows.append(row)

        # These observations become available to the next round.
        flow.append(public_side)
        sizes.append(public_quantity)
        moves.append(future - fair)
        markouts.append(public_side * (future - fair))

        spot, iv = next_spot, next_iv

    return pd.DataFrame(rows)

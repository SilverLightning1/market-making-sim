"""Continuous dealer sessions and reconciled PnL attribution."""

from math import ceil, floor

import numpy as np
import pandas as pd

from .execution import (
    Account,
    Quote,
    execute,
    hedge,
    liquidate,
    wealth,
)


STRATEGIES = (
    "fixed",
    "inventory",
    "volatility",
    "always_wide",
    "toxicity",
    "shuffled",
    "no_trade",
)


def quote_policy(
    fair,
    inventory,
    lag_vol,
    probability,
    strategy,
    cfg,
    threshold,
    strength,
):
    """Only current public value, inventory, and past data are allowed."""
    half_spread = cfg.half_spread

    if strategy == "always_wide":
        half_spread *= 1 + strength

    elif strategy == "volatility":
        half_spread += min(0.35, lag_vol)

    elif (
        strategy in ("toxicity", "shuffled")
        and probability >= threshold
    ):
        half_spread *= 1 + strength

    # Long inventory lowers both quotes, encouraging sales.
    skew = 0 if strategy == "fixed" else 0.003 * inventory
    center = fair - skew

    bid = max(
        cfg.tick,
        floor(
            (center - half_spread) / cfg.tick + 1e-9
        ) * cfg.tick,
    )

    ask = max(
        bid + cfg.tick,
        ceil(
            (center + half_spread) / cfg.tick - 1e-9
        ) * cfg.tick,
    )

    bid_size = min(
        cfg.quote_size,
        max(0, cfg.inventory_limit - inventory),
    )

    ask_size = min(
        cfg.quote_size,
        max(0, cfg.inventory_limit + inventory),
    )

    if strategy == "no_trade":
        bid_size = 0
        ask_size = 0

    return Quote(bid, ask, bid_size, ask_size)


def run_session(
    tape,
    cfg,
    strategy="fixed",
    probabilities=None,
    threshold=0.35,
    strength=3.0,
):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")

    if len(tape) == 0:
        raise ValueError("Empty tape")

    probabilities = (
        np.zeros(len(tape))
        if probabilities is None
        else np.asarray(probabilities)
    )

    if (
        len(probabilities) != len(tape)
        or not np.isfinite(probabilities).all()
        or (
            (probabilities < 0)
            | (probabilities > 1)
        ).any()
    ):
        raise ValueError("Invalid probabilities")

    account = Account(cfg.initial_cash)
    records = []

    peak = cfg.initial_cash
    max_drawdown = 0.0

    for index, row in enumerate(
        tape.itertuples(index=False)
    ):
        wealth_before = wealth(
            account, row.fair, row.spot, cfg
        )

        previous_inventory = account.inventory

        quote = quote_policy(
            fair=row.fair,
            inventory=previous_inventory,
            lag_vol=row.premium_volatility,
            probability=probabilities[index],
            strategy=strategy,
            cfg=cfg,
            threshold=threshold,
            strength=strength,
        )

        bid, ask = quote.bid, quote.ask

        # The customer acts after the dealer has posted its quotes.
        if row.informed:
            if row.signal > ask:
                side = "buy"
            elif row.signal < bid:
                side = "sell"
            else:
                side = None

            limit = ask if side == "buy" else bid

        else:
            side = (
                "buy" if row.noise_side == 1 else "sell"
            )

            limit = max(
                cfg.tick,
                row.fair
                + row.noise_side * row.willingness,
            )

        fees_before = account.fees
        hedge_costs_before = account.hedge_costs

        if side:
            report = execute(
                account,
                quote,
                side,
                int(row.quantity),
                limit,
                cfg,
            )
        else:
            report = {
                "filled": 0,
                "remaining": 0,
                "price": None,
            }

        filled = report["filled"]
        inventory_change = (
            account.inventory - previous_inventory
        )

        # Hedge at the current spot before the next price is revealed.
        hedge(account, row.delta, row.spot, cfg)
        hedge_shares = account.shares

        execution_price = (
            report["price"]
            if report["price"] is not None
            else row.fair
        )

        spread_capture = (
            -inventory_change
            * (execution_price - row.fair)
            * cfg.multiplier
        )

        premium_change = row.future - row.fair

        new_position_move = (
            inventory_change
            * premium_change
            * cfg.multiplier
        )

        old_position_move = (
            previous_inventory
            * premium_change
            * cfg.multiplier
        )

        hedge_move = (
            hedge_shares * (row.next_spot - row.spot)
        )

        costs = (
            account.fees
            - fees_before
            + account.hedge_costs
            - hedge_costs_before
        )

        wealth_after = wealth(
            account, row.future, row.next_spot, cfg
        )

        attributed_pnl = (
            spread_capture
            + new_position_move
            + old_position_move
            + hedge_move
            - costs
        )

        if not np.isclose(
            wealth_after - wealth_before,
            attributed_pnl,
            atol=1e-7,
        ):
            raise AssertionError("PnL identity failed")

        peak = max(peak, wealth_after)
        max_drawdown = max(
            max_drawdown, peak - wealth_after
        )

        records.append({
            "step": index,
            "strategy": strategy,
            "pnl": wealth_after - cfg.initial_cash,
            "cash": account.cash,
            "inventory": account.inventory,
            "hedge_shares": hedge_shares,
            "net_delta": (
                account.inventory
                * cfg.multiplier
                * row.next_delta
                + hedge_shares
            ),
            "fair": row.fair,
            "next_fair": row.future,
            "spot": row.spot,
            "bid": bid,
            "ask": ask,
            "toxicity_probability": probabilities[index],
            "informed": row.informed,
            "high_regime": row.high_regime,
            "side": side or "none",
            "filled": filled,
            "requested": row.quantity if side else 0,
            "spread_capture": spread_capture,
            "new_position_move": new_position_move,
            "old_position_move": old_position_move,
            "hedge_move": hedge_move,
            "costs": costs,
            "round_pnl": wealth_after - wealth_before,
            "markout": spread_capture + new_position_move,
        })

    history = pd.DataFrame(records)

    liquidation_cost = liquidate(
        account, row.future, row.next_spot, cfg
    )

    max_drawdown = max(
        max_drawdown, peak - account.cash
    )

    contracts = int(history["filled"].sum())
    total_markout = history["markout"].sum()

    summary = {
        "strategy": strategy,
        "net_pnl": account.cash - cfg.initial_cash,
        "marked_pnl": float(history["pnl"].iloc[-1]),
        "liquidation_cost": liquidation_cost,
        "max_drawdown": max_drawdown,
        "contracts": contracts,
        "fill_rate": float(
            (history["filled"] > 0).mean()
        ),
        "volume_fill_rate": (
            contracts
            / max(int(history["requested"].sum()), 1)
        ),
        "mean_abs_inventory": float(
            history["inventory"].abs().mean()
        ),
        "max_abs_inventory": int(
            history["inventory"].abs().max()
        ),
        "inventory_limit_fraction": float(
            (
                history["inventory"].abs()
                == cfg.inventory_limit
            ).mean()
        ),
        "fees": account.fees,
        "hedge_costs": account.hedge_costs,
        "mean_markout": (
            total_markout / contracts
            if contracts
            else np.nan
        ),
        "terminal_inventory": account.inventory,
        "terminal_shares": account.shares,
    }

    if not np.isclose(
        history["round_pnl"].sum() - liquidation_cost,
        summary["net_pnl"],
        atol=1e-6,
    ):
        raise AssertionError("Session reconciliation failed")

    return history, summary

"""Dealer accounting, limit-order execution, hedging, and liquidation."""

from dataclasses import dataclass
from math import floor, isfinite


@dataclass
class Account:
    cash: float
    inventory: int = 0
    shares: float = 0.0
    fees: float = 0.0
    hedge_costs: float = 0.0


@dataclass
class Quote:
    bid: float
    ask: float
    bid_size: int
    ask_size: int


def execute(
    account,
    quote,
    side,
    quantity,
    limit,
    cfg,
    tif="IOC",
):
    """Execute from the CUSTOMER's perspective.

    Customer buys -> dealer sells at its ask.
    Customer sells -> dealer buys at its bid.

    DAY returns a pending status for unfilled quantity. A caller would
    need to maintain and retry that remainder. The experiment uses IOC.
    """
    if side not in ("buy", "sell"):
        raise ValueError("side must be buy or sell")

    if tif not in ("IOC", "DAY"):
        raise ValueError("tif must be IOC or DAY")

    if type(quantity) is not int or quantity <= 0:
        raise ValueError("Whole positive contracts required")

    if not isfinite(limit) or limit <= 0:
        raise ValueError("Positive finite limit required")

    if not 0 < quote.bid < quote.ask:
        raise ValueError("Invalid quote")

    if any(
        type(size) is not int or size < 0
        for size in (quote.bid_size, quote.ask_size)
    ):
        raise ValueError("Quote sizes must be nonnegative integers")

    customer_buys = side == "buy"

    price = quote.ask if customer_buys else quote.bid

    compatible = (
        limit >= price if customer_buys else limit <= price
    )

    if customer_buys:
        capacity = cfg.inventory_limit + account.inventory
        available = quote.ask_size
    else:
        capacity = cfg.inventory_limit - account.inventory
        available = quote.bid_size

        cash_capacity = max(
            0,
            floor(
                account.cash
                / (price * cfg.multiplier + cfg.fee)
            ),
        )

        capacity = min(capacity, cash_capacity)

    filled = (
        min(quantity, available, max(0, capacity))
        if compatible
        else 0
    )

    # Signed change in the dealer's option inventory.
    inventory_change = -filled if customer_buys else filled
    fee = filled * cfg.fee

    account.cash -= (
        inventory_change * price * cfg.multiplier + fee
    )

    account.inventory += inventory_change
    account.fees += fee

    if customer_buys:
        quote.ask_size -= filled
    else:
        quote.bid_size -= filled

    remaining = quantity - filled

    if remaining == 0:
        status = "filled"
    elif tif == "DAY":
        status = "pending"
    else:
        status = "canceled"

    return {
        "filled": filled,
        "remaining": remaining,
        "price": price if filled else None,
        "status": status,
    }


def wealth(account, premium, spot, cfg):
    """Cash plus marked option and stock positions."""
    return (
        account.cash
        + account.inventory * premium * cfg.multiplier
        + account.shares * spot
    )


def hedge(account, delta, spot, cfg):
    """Offset current option delta with stock.

    This model allows fractional hedge shares and stock financing.
    """
    target = (
        -account.inventory * cfg.multiplier * delta
        if cfg.hedge
        else 0.0
    )

    trade = target - account.shares
    cost = abs(trade) * cfg.hedge_cost_per_share

    account.cash -= trade * spot + cost
    account.shares = target
    account.hedge_costs += cost


def liquidate(account, premium, spot, cfg):
    """Close positions against an external venue with assumed depth."""
    before = wealth(account, premium, spot, cfg)

    if account.inventory:
        if account.inventory > 0:
            price = max(
                0, premium - cfg.external_half_spread
            )
        else:
            price = premium + cfg.external_half_spread

        fee = abs(account.inventory) * cfg.fee

        account.cash += (
            account.inventory * price * cfg.multiplier - fee
        )

        account.fees += fee
        account.inventory = 0

    stock_cost = (
        abs(account.shares) * cfg.hedge_cost_per_share
    )

    account.cash += account.shares * spot - stock_cost
    account.hedge_costs += stock_cost
    account.shares = 0.0

    return before - account.cash

# Adverse-selection lab

A Python experiment on adverse selection in options market making.

The dealer posts a bid and ask for one synthetic option. Noise customers
trade for liquidity reasons. Informed customers receive a noisy estimate
of the next option premium and trade when the dealer's quotes look favorable.

The question is whether lagged public order flow can help the dealer
recognize dangerous periods and adjust its quotes.

## What is implemented

- European call and put valuation using Black-Scholes.
- Informed and noise customer arrivals.
- Partial fills, contract inventory limits, and cash accounting.
- Delta hedging with transaction costs.
- A logistic flow-toxicity detector.
- Separate training, validation, and test sessions.
- Matched strategy comparisons and session-level confidence intervals.
- Stress tests, PnL attribution, tables, and plots.

This is a single-dealer quote simulator, not a full exchange matching engine.

## Setup

Use Python 3.12 or 3.13.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

Run the accounting and timing checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Run a smaller experiment first:

```bash
.venv/bin/python run_lab.py --quick
```

Run the full experiment:

```bash
.venv/bin/python run_lab.py
```

Run puts in a separate output directory:

```bash
.venv/bin/python run_lab.py --kind put --output results_put
```

`simulator.py` is an alternative entry point with the same arguments.

## Strategies

| Strategy | Behavior |
| --- | --- |
| fixed | Constant-width spread around current fair value |
| inventory | Shifts quotes to reduce inventory |
| volatility | Widens using lagged premium volatility |
| always_wide | Uses a wider spread throughout the session |
| toxicity | Widens when predicted informed-arrival probability is high |
| shuffled | Uses independent scores drawn from validation predictions |
| no_trade | Holds cash and submits no executable size |

The fixed strategy has a fixed spread, not permanently frozen price levels.
Every strategy observes the same current synthetic fair value.

## Timing and information

Each round follows this order:

1. Observe current fair value and completed public history.
2. Post dealer quotes.
3. Generate the customer's trading decision.
4. Execute compatible volume.
5. Hedge using current delta and stock price.
6. Reveal the next price and revalue the account.

The simulator knows the generated future. The dealer does not.

Informed customers receive the next premium plus Gaussian error.
This deliberately creates an information advantage so its effects can
be measured.

Public features come from an independent synthetic reference venue.
They are observable even when our dealer receives no fill.

## Evaluation

The detector is fitted on training sessions. Spread parameters are chosen
on validation sessions. Test sessions are evaluated after those choices.

Strategies share potential customers and price paths within each seed.
Different quotes can produce different executions.

Reported net PnL includes:

- Option execution fees.
- Stock hedge costs.
- Terminal option liquidation across an external spread.
- Closing the stock hedge.

The final account has no remaining option or stock position.

Confidence intervals use independent session observations rather than
treating correlated trades as independent samples.

## Outputs

The program creates these files in `results/`:

- `REPORT.md`: generated findings and assumptions.
- `summary.csv`: strategy-level performance.
- `session_metrics.csv`: individual session outcomes.
- `paired_differences.csv`: toxicity versus each baseline.
- `classifier_metrics.json`: classification diagnostics.
- `validation_tuning.csv`: toxicity parameter search.
- `wide_validation_tuning.csv`: always-wide parameter search.
- `stress_summary.csv`: performance under changed assumptions.
- `sample_*.csv`: round-by-round sample-session records.
- `detector.json`: fitted classifier parameters.
- `manifest.json`: configuration, seeds, versions, and source hashes.
- PNG charts for strategies, calibration, a session, and stress tests.

Quick-run results go into `results_quick/`.

Generated results are ignored by Git by default.

## Reading the code

- `market.py`: synthetic prices, customer types, and lagged features.
- `execution.py`: cash, inventory, fills, hedging, and liquidation.
- `simulation.py`: quote policies and continuous-session accounting.
- `detector.py`: classifier fitting and prediction.
- `experiment.py`: evaluation and generated outputs.
- `tests/test_lab.py`: accounting and information-timing checks.

## Limitations

The generator creates persistent regimes that make informed flow partly
predictable. That does not establish predictability in real markets.

Current fair prices and deltas are known exactly within the synthetic model.
There is no queue competition, latency, market impact, assignment, early
exercise, volatility smile, or broker margin model.

Stock hedges may be fractional and financed. Option purchases face a cash
constraint. Terminal external liquidity is assumed sufficient.

Sessions use IOC requests. The execution engine can report a DAY remainder,
but the experiment does not maintain resting customer DAY orders.

A useful result can be that a simple wider-spread baseline beats the
detector. Classification accuracy and trading profitability are different
questions.

This project studies a mechanism under explicit assumptions; it does not
demonstrate a live trading edge.

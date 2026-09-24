"""Training, validation, held-out evaluation, and generated results."""

from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import platform

import matplotlib

# Save plots to files without opening blocking GUI windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn

from .detector import Detector, metrics
from .market import Config, FEATURES, generate_tape
from .simulation import STRATEGIES, run_session


def summarize(results):
    rows = []

    for strategy, group in results.groupby(
        "strategy", sort=False
    ):
        pnl = group["net_pnl"].to_numpy()

        standard_error = (
            pnl.std(ddof=1) / np.sqrt(len(pnl))
            if len(pnl) > 1
            else 0.0
        )

        rows.append({
            "strategy": strategy,
            "sessions": len(group),
            "mean_net_pnl": pnl.mean(),
            "ci95_low": pnl.mean() - 1.96 * standard_error,
            "ci95_high": pnl.mean() + 1.96 * standard_error,
            "loss_rate": float(np.mean(pnl < 0)),
            "worst_net_pnl": pnl.min(),
            "mean_drawdown": group["max_drawdown"].mean(),
            "mean_contracts": group["contracts"].mean(),
            "mean_fill_rate": group["fill_rate"].mean(),
            "mean_abs_inventory": (
                group["mean_abs_inventory"].mean()
            ),
            "mean_markout": group["mean_markout"].mean(),
            "mean_fees": group["fees"].mean(),
            "mean_hedge_costs": group["hedge_costs"].mean(),
        })

    return pd.DataFrame(rows)


def paired_differences(results):
    wide = results.pivot(
        index="seed",
        columns="strategy",
        values="net_pnl",
    )

    rows = []

    for baseline in STRATEGIES:
        if baseline == "toxicity":
            continue

        differences = (
            wide["toxicity"] - wide[baseline]
        )

        standard_error = (
            differences.std(ddof=1)
            / np.sqrt(len(differences))
        )

        rows.append({
            "comparison": f"toxicity minus {baseline}",
            "mean_difference": differences.mean(),
            "ci95_low": (
                differences.mean()
                - 1.96 * standard_error
            ),
            "ci95_high": (
                differences.mean()
                + 1.96 * standard_error
            ),
        })

    return pd.DataFrame(rows)


def evaluate(
    tapes,
    seeds,
    cfg,
    detector,
    selected,
    save_sample=False,
):
    rows = []
    traces = {}

    for tape, seed in zip(tapes, seeds):
        probabilities = detector.predict(tape)

        # Negative control: sample validation scores independently.
        # Do not shuffle future scores from the current test session.
        shuffled = np.random.default_rng(
            seed + 900_000
        ).choice(
            selected["score_pool"],
            len(tape),
        )

        for strategy in STRATEGIES:
            strength = (
                selected["wide_strength"]
                if strategy == "always_wide"
                else selected["strength"]
            )

            scores = (
                shuffled
                if strategy == "shuffled"
                else probabilities
            )

            trace, result = run_session(
                tape,
                cfg,
                strategy,
                scores,
                selected["threshold"],
                strength,
            )

            result["seed"] = seed
            rows.append(result)

            if save_sample and seed == seeds[0]:
                traces[strategy] = trace

    return pd.DataFrame(rows), traces


def save_charts(out, summary, traces, detector, tapes):
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Strategy comparison.
    fig, ax = plt.subplots(figsize=(10, 5))

    errors = (
        summary["ci95_high"] - summary["mean_net_pnl"]
    ).to_numpy()

    ax.bar(
        summary["strategy"],
        summary["mean_net_pnl"],
        yerr=errors,
        capsize=4,
        color=[
            "#8094a8",
            "#537c9b",
            "#3890a0",
            "#637959",
            "#de8c38",
            "#9c77a4",
            "#c6c9cc",
        ],
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net dollars per session, after liquidation")
    ax.set_title(
        "Held-out mean PnL and approximate 95% confidence intervals"
    )
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out / "strategy_comparison.png", dpi=150)
    plt.close(fig)

    # First held-out session, selected before seeing performance.
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True
    )

    for strategy in (
        "fixed",
        "inventory",
        "volatility",
        "always_wide",
        "toxicity",
    ):
        trace = traces[strategy]

        axes[0].plot(
            trace["step"],
            trace["pnl"],
            label=strategy,
            linewidth=1.1,
        )

        axes[1].plot(
            trace["step"],
            trace["inventory"],
            label=strategy,
            linewidth=0.8,
            alpha=0.8,
        )

    trace = traces["toxicity"]

    axes[2].plot(
        trace["step"],
        trace["toxicity_probability"],
        label="Predicted informed-arrival probability",
    )

    axes[2].fill_between(
        trace["step"],
        0,
        1,
        where=trace["high_regime"].astype(bool),
        color="gray",
        alpha=0.2,
        label="Hidden regime: diagnostic only",
    )

    axes[0].set_title("First held-out session")
    axes[0].set_ylabel("Marked PnL ($)")
    axes[1].set_ylabel("Option contracts")
    axes[2].set_ylabel("Probability")
    axes[2].set_xlabel("One-minute round")

    axes[0].legend(ncol=3, fontsize=8)
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out / "sample_session.png", dpi=150)
    plt.close(fig)

    # Classifier calibration.
    labels = np.concatenate([
        tape["informed"].to_numpy()
        for tape in tapes
    ])

    probabilities = np.concatenate([
        detector.predict(tape)
        for tape in tapes
    ])

    table = pd.DataFrame({
        "label": labels,
        "probability": probabilities,
    })

    table["bin"] = pd.cut(
        table["probability"],
        np.linspace(0, 1, 11),
        include_lowest=True,
    )

    calibration = table.groupby(
        "bin", observed=True
    ).agg(
        predicted=("probability", "mean"),
        observed=("label", "mean"),
        count=("label", "size"),
    )

    calibration.to_csv(out / "calibration.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(
        [0, 1], [0, 1], "--", color="gray"
    )

    axes[0].plot(
        calibration["predicted"],
        calibration["observed"],
        "o-",
    )

    axes[0].set(
        xlabel="Mean prediction",
        ylabel="Observed informed fraction",
        title="Held-out calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    axes[1].barh(
        FEATURES, detector.coefficients
    )
    axes[1].set_title("Standardized logistic coefficients")

    fig.tight_layout()
    fig.savefig(out / "detector.png", dpi=150)
    plt.close(fig)

    # Diagnostic markouts for the first held-out session.
    rows = []

    for strategy, trace in traces.items():
        filled = trace[trace["filled"] > 0]

        for informed, group in filled.groupby("informed"):
            contracts = int(group["filled"].sum())

            rows.append({
                "strategy": strategy,
                "informed": int(informed),
                "contracts": contracts,
                "mean_markout": (
                    group["markout"].sum() / contracts
                ),
            })

    pd.DataFrame(rows).to_csv(
        out / "sample_markouts.csv", index=False
    )


def markdown_table(frame):
    """Render a Markdown table without a tabulate dependency."""
    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(
        ["---"] * len(frame.columns)
    ) + " |"

    rows = [
        "| " + " | ".join(
            format_value(value) for value in row
        ) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]

    return "\n".join([header, separator, *rows])


def write_report(
    out,
    cfg,
    counts,
    selected,
    summary,
    differences,
    classifier,
    stress_summary,
):
    ntrain, nval, ntest = counts

    best_strategy = summary.sort_values(
        "mean_net_pnl", ascending=False
    ).iloc[0]["strategy"]

    worst_stress = stress_summary[
        stress_summary["strategy"] == "toxicity"
    ].sort_values("mean_net_pnl").iloc[0]

    summary_columns = [
        "strategy",
        "mean_net_pnl",
        "ci95_low",
        "ci95_high",
        "mean_contracts",
        "mean_drawdown",
        "loss_rate",
    ]

    stress_columns = [
        "scenario",
        "strategy",
        "mean_net_pnl",
        "mean_contracts",
        "loss_rate",
    ]

    report = f"""# Adverse-selection lab: results

## Experiment

Synthetic European {cfg.kind}, strike 100, initially 30 calendar days
to expiry. Black-Scholes valuation uses zero rates and dividends.
Each contract has multiplier {cfg.multiplier}.

Each session contains {cfg.steps} one-minute rounds.
Training sessions: {ntrain}. Validation: {nval}. Test: {ntest}.

All strategies face the same potential customers and price paths
for each seed. Accounts persist throughout each session.

## Strategy comparison

Highest held-out mean: **{best_strategy}**.

{markdown_table(summary[summary_columns])}

![Strategy comparison](strategy_comparison.png)

Final net PnL includes option fees, hedge costs, and terminal liquidation.

## Does the detector help?

The classifier predicts the next customer's informed type using only
completed past observations. Labels are available for synthetic training
and evaluation, but never to the quoting policy.

ROC AUC: {classifier['roc_auc']:.3f}
Average precision: {classifier['average_precision']:.3f}
Informed prevalence: {classifier['informed_rate']:.3f}
Brier score: {classifier['brier']:.3f}
Constant training-prior Brier: {classifier['constant_train_prior_brier']:.3f}

{markdown_table(differences)}

Intervals use independent session-level paired PnL differences.
They are approximate 95% normal intervals, with no multiple-comparison
correction. They quantify simulation sampling uncertainty, not model risk.

A good classifier does not automatically produce a better trading policy.
Compare toxicity with volatility and always-wide as well as fixed quotes.

![Detector diagnostics](detector.png)

## Selected policy

Validation selected toxicity threshold {selected['threshold']}
and widening strength {selected['strength']}.

The always-wide baseline independently selected strength
{selected['wide_strength']}.

The base half-spread is ${cfg.half_spread:.2f}.
Inventory-aware strategies shift the quote center by
-0.003 times current option inventory.

The shuffled control independently samples the validation score
distribution. It does not use future test-session predictions.

## Session timing

1. Observe current fair value and completed public history.
2. Post bid, ask, and sizes.
3. Customer acts using its private signal or liquidity need.
4. Execute compatible volume subject to capacity.
5. Hedge at current stock price using current delta.
6. Reveal the next stock price and implied volatility.
7. Revalue the account.

After the last round, close all positions at an external venue.

![Sample session](sample_session.png)

The sample session is the first held-out seed, not a selected winner.
Its plotted PnL is marked PnL before terminal liquidation.

## PnL attribution

Every round checks this identity:

Change in wealth =
spread capture
+ new-position value change
+ old-position value change
+ hedge-share PnL
- option fees
- hedge costs.

Trade markout equals spread capture plus new-position value change.
It measures the option trade against the next reference premium,
before fees and hedging.

Negative markouts identify adverse fills under this generator.
Total PnL also includes accumulated inventory and hedge performance.

## Stress cases

Frozen model and selected parameters are evaluated under each scenario.

The lowest toxicity-policy mean occurred in
**{worst_stress['scenario']}**:
${worst_stress['mean_net_pnl']:.2f} per session.

![Stress tests](stress_tests.png)

{markdown_table(stress_summary[stress_columns])}

Stress means are descriptive. Results are retained when the detector
loses money or a simpler policy performs better.

## Limits

This is a single-dealer quote simulator, not a full exchange order book.

- Hidden persistent regimes jointly affect volatility and informed flow.
  Predictable clues therefore exist by construction.
- The dealer knows the current synthetic fair premium and delta exactly.
- Informed customers observe the next premium plus Gaussian error.
  This is a modeling assumption, not an actual forecast implementation.
- Public history comes from an independent synthetic reference venue.
  The dealer's actions have no market impact.
- Noise customers have random direction, size, and price tolerance.
  Wider spreads can lose benign volume.
- Sessions use IOC orders. The execution engine reports DAY remainders
  but does not maintain a persistent DAY-order queue.
- Stock hedges can be fractional and financed.
  Option purchases face a cash constraint.
- Terminal external liquidity is assumed sufficient.
- No queue competition, latency, assignment, early exercise, margin,
  funding interest, stock borrow limits, or volatility smile is modeled.

These results do not establish a real trading edge.
"""

    (out / "REPORT.md").write_text(
        report, encoding="utf-8"
    )


def run_experiment(out, quick=False, kind="call"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        steps=160 if quick else 390,
        kind=kind,
    )

    counts = (8, 4, 6) if quick else (32, 12, 40)
    ntrain, nval, ntest = counts

    seeds = {
        "train": list(range(1000, 1000 + ntrain)),
        "validation": list(range(2000, 2000 + nval)),
        "test": list(range(5000, 5000 + ntest)),
    }

    print(
        "Generating training and validation sessions...",
        flush=True,
    )

    train = [
        generate_tape(seed, cfg)
        for seed in seeds["train"]
    ]

    validation = [
        generate_tape(seed, cfg)
        for seed in seeds["validation"]
    ]

    detector = Detector.fit(train)
    detector.save(out / "detector.json")

    validation_scores = [
        detector.predict(tape)
        for tape in validation
    ]

    print(
        "Selecting policy parameters on validation sessions...",
        flush=True,
    )

    candidates = []

    for threshold in (0.15, 0.30, 0.45):
        for strength in (1.0, 3.0, 6.0):
            scores = []

            for tape, probabilities in zip(
                validation, validation_scores
            ):
                _, result = run_session(
                    tape,
                    cfg,
                    "toxicity",
                    probabilities,
                    threshold,
                    strength,
                )
                scores.append(result["net_pnl"])

            candidates.append({
                "threshold": threshold,
                "strength": strength,
                "mean_validation_pnl": float(np.mean(scores)),
            })

    pd.DataFrame(candidates).to_csv(
        out / "validation_tuning.csv",
        index=False,
    )

    best = max(
        candidates,
        key=lambda row: row["mean_validation_pnl"],
    )

    wide_candidates = []

    for strength in (1.0, 3.0, 6.0):
        scores = [
            run_session(
                tape,
                cfg,
                "always_wide",
                strength=strength,
            )[1]["net_pnl"]
            for tape in validation
        ]

        wide_candidates.append({
            "strength": strength,
            "mean_validation_pnl": float(np.mean(scores)),
        })

    pd.DataFrame(wide_candidates).to_csv(
        out / "wide_validation_tuning.csv",
        index=False,
    )

    best_wide = max(
        wide_candidates,
        key=lambda row: row["mean_validation_pnl"],
    )

    selected = {
        "threshold": best["threshold"],
        "strength": best["strength"],
        "wide_strength": best_wide["strength"],
        "score_pool": np.concatenate(validation_scores),
    }

    test = [
        generate_tape(seed, cfg)
        for seed in seeds["test"]
    ]

    print(
        "Evaluating seven strategies on held-out sessions...",
        flush=True,
    )

    results, traces = evaluate(
        test,
        seeds["test"],
        cfg,
        detector,
        selected,
        save_sample=True,
    )

    results.to_csv(
        out / "session_metrics.csv",
        index=False,
    )

    summary = summarize(results)
    summary.to_csv(out / "summary.csv", index=False)

    differences = paired_differences(results)
    differences.to_csv(
        out / "paired_differences.csv",
        index=False,
    )

    for strategy, trace in traces.items():
        trace.to_csv(
            out / f"sample_{strategy}.csv",
            index=False,
        )

    classifier = metrics(test, detector)

    training_prior = float(np.concatenate([
        tape["informed"].to_numpy()
        for tape in train
    ]).mean())

    test_labels = np.concatenate([
        tape["informed"].to_numpy()
        for tape in test
    ])

    classifier["training_prior"] = training_prior
    classifier["constant_train_prior_brier"] = float(
        np.mean((test_labels - training_prior) ** 2)
    )

    (out / "classifier_metrics.json").write_text(
        json.dumps(classifier, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame({
        "feature": FEATURES,
        "coefficient": detector.coefficients,
    }).to_csv(
        out / "feature_coefficients.csv",
        index=False,
    )

    save_charts(
        out, summary, traces, detector, test
    )

    print(
        "Running stress cases with frozen parameters...",
        flush=True,
    )

    opposite_kind = (
        "put" if kind == "call" else "call"
    )

    stress_configs = {
        "no_informed": replace(
            cfg,
            low_informed=0,
            high_informed=0,
        ),
        "higher_costs": replace(
            cfg,
            fee=2.0,
            hedge_cost_per_share=0.02,
        ),
        "weak_private_signal": replace(
            cfg,
            signal_noise=0.25,
        ),
        "large_jumps": replace(
            cfg,
            jump_scale=2,
        ),
        "no_regime_persistence": replace(
            cfg,
            enter_high=0.2,
            exit_high=0.8,
        ),
        f"{opposite_kind}_instead_of_{kind}": replace(
            cfg,
            kind=opposite_kind,
        ),
        "unhedged": replace(
            cfg,
            hedge=False,
        ),
    }

    stress_count = 4 if quick else 16
    stress_seeds = list(
        range(6000, 6000 + stress_count)
    )

    stress_rows = []

    for scenario, stress_cfg in stress_configs.items():
        tapes = [
            generate_tape(seed, stress_cfg)
            for seed in stress_seeds
        ]

        scenario_results, _ = evaluate(
            tapes,
            stress_seeds,
            stress_cfg,
            detector,
            selected,
        )

        scenario_results["scenario"] = scenario
        stress_rows.append(scenario_results)

    stress_results = pd.concat(
        stress_rows, ignore_index=True
    )

    stress_results.to_csv(
        out / "stress_session_metrics.csv",
        index=False,
    )

    stress_summaries = []

    for scenario, group in stress_results.groupby("scenario"):
        scenario_summary = summarize(group)
        scenario_summary.insert(0, "scenario", scenario)
        stress_summaries.append(scenario_summary)

    stress_summary = pd.concat(
        stress_summaries, ignore_index=True
    )

    stress_summary.to_csv(
        out / "stress_summary.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    stress_summary.pivot(
        index="scenario",
        columns="strategy",
        values="mean_net_pnl",
    )[[
        "inventory",
        "always_wide",
        "toxicity",
    ]].plot.bar(ax=ax)

    ax.set_ylabel("Mean net PnL ($)")
    ax.set_xlabel("")
    ax.set_title("Frozen-policy stress tests")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(out / "stress_tests.png", dpi=150)
    plt.close(fig)

    source_dir = Path(__file__).parent

    manifest = {
        "config": asdict(cfg),
        "seeds": seeds,
        "stress_seeds": stress_seeds,
        "selected": {
            key: value
            for key, value in selected.items()
            if key != "score_pool"
        },
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "code_sha256": {
            path.name: hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_dir.glob("*.py")
        },
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    write_report(
        out,
        cfg,
        counts,
        selected,
        summary,
        differences,
        classifier,
        stress_summary,
    )

    print(
        summary[[
            "strategy",
            "mean_net_pnl",
            "mean_contracts",
        ]].round(2).to_string(index=False),
        flush=True,
    )

    print(
        f"\nReport and plots saved to: {out.resolve()}",
        flush=True,
    )

    return summary

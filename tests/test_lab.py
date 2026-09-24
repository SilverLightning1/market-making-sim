"""Accounting, timing, reproducibility, and model checks."""

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from mm_lab.detector import Detector
from mm_lab.execution import (
    Account,
    Quote,
    execute,
    liquidate,
    wealth,
)
from mm_lab.market import (
    Config,
    FEATURES,
    generate_tape,
    option_value,
)
from mm_lab.simulation import run_session


class LabTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(steps=40, fee=0)

    def test_partial_fill_accounting(self):
        account = Account(1000)
        quote = Quote(1.90, 2.00, 2, 2)

        report = execute(
            account, quote, "buy", 3, 2.00, self.cfg
        )

        self.assertEqual(
            (
                report["filled"],
                report["remaining"],
                report["status"],
            ),
            (2, 1, "canceled"),
        )

        self.assertEqual(
            (
                account.cash,
                account.inventory,
                quote.ask_size,
            ),
            (1400, -2, 0),
        )

        execute(
            account, quote, "sell", 1, 1.90, self.cfg
        )

        self.assertEqual(
            (account.cash, account.inventory),
            (1210, -1),
        )

    def test_price_and_day(self):
        account = Account(1000)
        quote = Quote(1.90, 2.00, 2, 2)

        report = execute(
            account,
            quote,
            "buy",
            1,
            1.95,
            self.cfg,
            "DAY",
        )

        self.assertEqual(report["status"], "pending")
        self.assertEqual(account.cash, 1000)

        with self.assertRaises(ValueError):
            execute(
                account,
                quote,
                "buy",
                1.5,
                2.00,
                self.cfg,
            )

        with self.assertRaises(ValueError):
            execute(
                account,
                quote,
                "buy",
                1,
                float("nan"),
                self.cfg,
            )

    def test_inventory_limits_both_sides(self):
        cfg = replace(self.cfg, inventory_limit=2)

        cases = [
            ("buy", -2, 2.00),
            ("sell", 2, 1.90),
        ]

        for side, expected_inventory, limit in cases:
            account = Account(10000)
            quote = Quote(1.90, 2.00, 10, 10)

            report = execute(
                account, quote, side, 5, limit, cfg
            )

            self.assertEqual(report["filled"], 2)
            self.assertEqual(
                account.inventory, expected_inventory
            )

            second = execute(
                account, quote, side, 1, limit, cfg
            )

            self.assertEqual(second["filled"], 0)

    def test_cash_limit(self):
        account = Account(100)
        quote = Quote(1.90, 2.00, 2, 2)

        report = execute(
            account, quote, "sell", 2, 1.90, self.cfg
        )

        self.assertEqual(report["filled"], 0)

    def test_liquidation(self):
        account = Account(
            1000, inventory=-2, shares=3
        )

        initial = wealth(
            account, 2.00, 100, self.cfg
        )

        cost = liquidate(
            account, 2.00, 100, self.cfg
        )

        self.assertAlmostEqual(cost, 8 + 3 * 0.005)
        self.assertAlmostEqual(
            account.cash, initial - cost
        )

        self.assertEqual(
            (account.inventory, account.shares),
            (0, 0),
        )

    def test_put_call_parity_and_delta(self):
        call, call_delta = option_value(
            101, 100, 0.1, 0.2, "call"
        )

        put, put_delta = option_value(
            101, 100, 0.1, 0.2, "put"
        )

        self.assertAlmostEqual(call - put, 1)
        self.assertAlmostEqual(
            call_delta - put_delta, 1
        )

        h = 0.0001

        numerical_delta = (
            option_value(
                101 + h, 100, 0.1, 0.2
            )[0]
            - option_value(
                101 - h, 100, 0.1, 0.2
            )[0]
        ) / (2 * h)

        self.assertAlmostEqual(
            call_delta,
            numerical_delta,
            places=6,
        )

    def test_reproducible_tape(self):
        first = generate_tape(1, self.cfg)
        second = generate_tape(1, self.cfg)

        self.assertTrue(first.equals(second))
        self.assertTrue((first["fair"] >= 0).all())

    def test_future_cannot_change_prefix_features(self):
        short = generate_tape(
            4, replace(self.cfg, steps=40)
        )

        long = generate_tape(
            4, replace(self.cfg, steps=80)
        )

        np.testing.assert_allclose(
            short[FEATURES],
            long[FEATURES].iloc[:40],
        )

        self.assertTrue(
            (short[FEATURES].iloc[0] == 0).all()
        )

    def test_lagged_feature_timing(self):
        tape = generate_tape(4, self.cfg)

        for index in range(1, len(tape)):
            previous = tape.iloc[
                max(0, index - self.cfg.window):index
            ]

            expected = (
                previous["public_side"]
                * (
                    previous["future"]
                    - previous["fair"]
                )
            ).mean()

            self.assertAlmostEqual(
                tape["mean_public_markout"].iloc[index],
                expected,
            )

    def test_default_funding_does_not_trap_short_cover(self):
        cfg = Config(steps=390)
        tape = generate_tape(3000, cfg)

        trace, _ = run_session(tape, cfg, "fixed")

        self.assertGreater(trace["cash"].min(), 0)

    def test_detector_ignores_hidden_fields(self):
        tapes = [
            generate_tape(seed, self.cfg)
            for seed in range(12)
        ]

        detector = Detector.fit(tapes)
        tape = tapes[0]
        altered = tape.copy()

        for column in (
            "future",
            "signal",
            "informed",
            "high_regime",
            "next_spot",
        ):
            altered[column] = 9999

        np.testing.assert_array_equal(
            detector.predict(tape),
            detector.predict(altered),
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            detector.save(path)
            restored = Detector.load(path)

            np.testing.assert_allclose(
                detector.predict(tape),
                restored.predict(tape),
            )

    def test_continuous_account_and_attribution(self):
        tape = generate_tape(7, self.cfg)

        for strategy in (
            "fixed",
            "inventory",
            "volatility",
            "always_wide",
            "toxicity",
            "no_trade",
        ):
            trace, summary = run_session(
                tape,
                self.cfg,
                strategy,
                np.full(len(tape), 0.5),
            )

            self.assertLessEqual(
                trace["inventory"].abs().max(),
                self.cfg.inventory_limit,
            )

            self.assertAlmostEqual(
                trace["round_pnl"].sum()
                - summary["liquidation_cost"],
                summary["net_pnl"],
                places=6,
            )

            self.assertEqual(
                (
                    summary["terminal_inventory"],
                    summary["terminal_shares"],
                ),
                (0, 0),
            )

            if strategy == "no_trade":
                self.assertEqual(summary["net_pnl"], 0)

    def test_private_signal_does_not_change_current_quote(self):
        tape = generate_tape(11, self.cfg)
        altered = tape.copy()
        altered.loc[0, "signal"] = 1000

        first, _ = run_session(tape, self.cfg)
        second, _ = run_session(altered, self.cfg)

        self.assertEqual(
            first["bid"].iloc[0],
            second["bid"].iloc[0],
        )

        self.assertEqual(
            first["ask"].iloc[0],
            second["ask"].iloc[0],
        )

    def test_expensive_fees_reduce_same_fills(self):
        tape = generate_tape(9, self.cfg)

        cheap_trace, cheap_summary = run_session(
            tape, self.cfg
        )

        expensive_trace, expensive_summary = run_session(
            tape, replace(self.cfg, fee=2)
        )

        np.testing.assert_array_equal(
            cheap_trace["filled"],
            expensive_trace["filled"],
        )

        self.assertLess(
            expensive_summary["net_pnl"],
            cheap_summary["net_pnl"],
        )


if __name__ == "__main__":
    unittest.main()

"""Regression and property tests for montecarlocli (standard library only)."""

import io
import os
import subprocess
import sys
import unittest

import montecarlocli as mc

VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
GOLDEN_SEED = 123

# Golden values captured from a known-good run with GOLDEN_SEED.
GOLDEN_WHEN = {
    "count": 1000,
    "minimum": 2,
    "maximum": 8,
    "mean": 4.243999999999992,
    "stdev": 1.0934622351319037,
    "p85": 5,
    "p50": 4,
    "p20": 3,
}
GOLDEN_HOWMANY = {
    "count": 1000,
    "minimum": 10,
    "maximum": 45,
    "mean": 27.276999999999997,
    "stdev": 6.33438200579941,
    "p85": 34,
    "p50": 27,
    "p20": 22,
}

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

GOLDEN_CLI_WHEN = """\
Reading data/example.csv.
* Forecast for 25 pbi on 500 trials *
* Number of iterations forecasted with 85.0% confidence: 5
* Number of iterations forecasted with 50.0% confidence: 4
* Number of iterations forecasted with 20.0% confidence: 3
* Min iterations 2.00 *
* Max iterations 9.00 *
* Avg iterations 3.88 *
* Standard deviations  1.23 *
"""

GOLDEN_CLI_HOWMANY = """\
Reading data/example.csv.
* Forecast for 4 with 500 trials *
* Number of pbi forecasted with 85.0% confidence: 21
* Number of pbi forecasted with 50.0% confidence: 31
* Number of pbi forecasted with 20.0% confidence: 38
* Min pbi 7 *
* Max pbi 53 *
* Avg pbi 30 *
* Standard deviations  9 *
"""


def run_when(seed=GOLDEN_SEED, trials=1000, backlog=20, batches=10):
    stream = mc.RandomStream(VALUES, seed)
    return mc.run_simulation(
        trials, lambda: mc.when_trial(stream, backlog), batch_count=batches
    )


def run_howmany(seed=GOLDEN_SEED, trials=1000, iterations=5, batches=10):
    stream = mc.RandomStream(VALUES, seed)
    return mc.run_simulation(
        trials, lambda: mc.howmany_trial(stream, iterations), batch_count=batches
    )


def aggregator_stats(agg):
    return {
        "count": agg.count,
        "minimum": agg.minimum,
        "maximum": agg.maximum,
        "mean": agg.mean,
        "stdev": agg.stdev,
        "p85": agg.percentile(0.85),
        "p50": agg.percentile(0.50),
        "p20": agg.percentile(0.20),
    }


class GoldenRegressionTest(unittest.TestCase):
    def test_when_golden(self):
        self.assertEqual(aggregator_stats(run_when()), GOLDEN_WHEN)

    def test_howmany_golden(self):
        self.assertEqual(aggregator_stats(run_howmany()), GOLDEN_HOWMANY)

    def test_cli_golden_output(self):
        env = dict(os.environ, MONTECARLO_SEED="7")
        for argv, golden in (
            (["-t", "500", "-f", "data/example.csv", "when", "-b", "25"],
             GOLDEN_CLI_WHEN),
            (["-t", "500", "-f", "data/example.csv", "howmany", "-i", "4"],
             GOLDEN_CLI_HOWMANY),
        ):
            result = subprocess.run(
                [sys.executable, "montecarlocli.py"] + argv,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, golden)


class ReproducibilityTest(unittest.TestCase):
    def test_same_seed_same_result(self):
        self.assertEqual(aggregator_stats(run_when()), aggregator_stats(run_when()))
        self.assertEqual(
            aggregator_stats(run_howmany()), aggregator_stats(run_howmany())
        )

    def test_different_seed_different_result(self):
        self.assertNotEqual(run_when(seed=1).mean, run_when(seed=2).mean)


class BatchInvarianceTest(unittest.TestCase):
    """Batch count may only affect progress, never the final statistics."""

    def test_when_stats_do_not_depend_on_batches(self):
        reference = aggregator_stats(run_when(batches=1))
        for batches in (2, 3, 7, 100, 1000):
            self.assertEqual(aggregator_stats(run_when(batches=batches)), reference)

    def test_howmany_stats_do_not_depend_on_batches(self):
        reference = aggregator_stats(run_howmany(batches=1))
        for batches in (2, 3, 7, 100, 1000):
            self.assertEqual(aggregator_stats(run_howmany(batches=batches)), reference)

    def test_batch_sizes_cover_all_trials(self):
        for trials in (1, 2, 5, 1000):
            for batches in (1, 3, 100, 5000):
                sizes = mc.batch_sizes(trials, batches)
                self.assertEqual(sum(sizes), trials)
                self.assertTrue(all(size > 0 for size in sizes))


class UnbiasednessTest(unittest.TestCase):
    """The sample mean and the ddof=1 sample variance are unbiased."""

    def test_sample_mean_is_unbiased(self):
        # Uniform draws from 0..30 have theoretical mean 15.
        values = list(range(31))
        stream = mc.RandomStream(values, seed=2024)
        agg = mc.run_simulation(200_000, stream.sample, batch_count=200)
        self.assertAlmostEqual(agg.mean, 15.0, delta=0.05)

    def test_sample_variance_is_unbiased(self):
        # Uniform draws from 0..30 have theoretical variance (31^2 - 1) / 12 = 80.
        values = list(range(31))
        true_variance = 80.0
        unbiased_estimates = []
        biased_estimates = []
        for rep in range(4000):
            stream = mc.RandomStream(values, seed=rep)
            draws = [stream.sample() for _ in range(10)]
            mean = sum(draws) / len(draws)
            m2 = sum((draw - mean) ** 2 for draw in draws)
            unbiased_estimates.append(m2 / (len(draws) - 1))
            biased_estimates.append(m2 / len(draws))
        avg_unbiased = sum(unbiased_estimates) / len(unbiased_estimates)
        avg_biased = sum(biased_estimates) / len(biased_estimates)
        # E[s^2] with ddof=1 equals the true variance...
        self.assertAlmostEqual(avg_unbiased, true_variance, delta=2.0)
        # ...while the ddof=0 estimator is systematically too low (factor 9/10).
        self.assertAlmostEqual(avg_biased, true_variance * 0.9, delta=2.0)
        self.assertLess(
            abs(avg_unbiased - true_variance), abs(avg_biased - true_variance)
        )

    def test_variance_of_sample_mean(self):
        # Var(mean of n draws) = sigma^2 / n; here sigma^2 = 80, n = 25.
        values = list(range(31))
        sample_size = 25
        means = []
        for rep in range(4000):
            stream = mc.RandomStream(values, seed=10_000 + rep)
            draws = [stream.sample() for _ in range(sample_size)]
            means.append(sum(draws) / sample_size)
        agg = mc.Aggregator()
        agg.update(means)
        self.assertAlmostEqual(agg.variance, 80.0 / sample_size, delta=0.3)

    def test_aggregator_variance_matches_naive_computation(self):
        draws = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        agg = mc.Aggregator()
        agg.update(draws)
        mean = sum(draws) / len(draws)
        expected = sum((d - mean) ** 2 for d in draws) / (len(draws) - 1)
        self.assertAlmostEqual(agg.mean, mean)
        self.assertAlmostEqual(agg.variance, expected)


class InputValidationTest(unittest.TestCase):
    def test_zero_trials_fails_cleanly(self):
        stderr = io.StringIO()
        old = sys.stderr
        sys.stderr = stderr
        try:
            exit_code = mc.main(["-t", "0", "when"])
        finally:
            sys.stderr = old
        self.assertNotEqual(exit_code, 0)
        self.assertIn("positive integer", stderr.getvalue())

    def test_negative_trials_fails_cleanly(self):
        stderr = io.StringIO()
        old = sys.stderr
        sys.stderr = stderr
        try:
            exit_code = mc.main(["-t", "-5", "howmany"])
        finally:
            sys.stderr = old
        self.assertNotEqual(exit_code, 0)
        self.assertIn("positive integer", stderr.getvalue())

    def test_batch_sizes_reject_non_positive_trials(self):
        for trials in (0, -1, -100):
            with self.assertRaises(ValueError):
                mc.batch_sizes(trials, 10)

    def test_invalid_percentile_fails(self):
        stdout = io.StringIO()
        old = sys.stdout
        sys.stdout = stdout
        try:
            exit_code = mc.main(["-p", "1.5", "when"])
        finally:
            sys.stdout = old
        self.assertEqual(exit_code, 1)
        self.assertIn("Percentiles", stdout.getvalue())


class ProgressTest(unittest.TestCase):
    def test_progress_reports_every_batch(self):
        stream = io.StringIO()
        progress = mc.Progress(10, stream=stream)
        mc.run_simulation(
            10, lambda: 1, batch_count=5, progress=progress
        )
        progress.close()
        output = stream.getvalue()
        self.assertEqual(output.count("Processing..."), 5)
        self.assertTrue(output.endswith("\n"))

    def test_progress_line_is_closed_on_exception(self):
        stream = io.StringIO()
        progress = mc.Progress(10, stream=stream)

        def failing_trial():
            raise RuntimeError("boom")

        try:
            try:
                mc.run_simulation(10, failing_trial, batch_count=5, progress=progress)
            finally:
                progress.close()
        except RuntimeError:
            pass
        else:
            self.fail("expected RuntimeError")
        # The progress line must be terminated so the terminal is not garbled.
        self.assertTrue(stream.getvalue().endswith("\n"))


if __name__ == "__main__":
    unittest.main()

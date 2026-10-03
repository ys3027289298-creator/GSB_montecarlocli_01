"""Regression and property tests for montecarlocli (stdlib unittest only)."""

import io
import math
import os
import statistics
import subprocess
import sys
import unittest
from contextlib import redirect_stderr

import montecarlocli as mc

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(REPO_ROOT, "data", "example.csv")
VALUES = mc.load_values(DATA_FILE)

GOLDEN_SEED = 42
GOLDEN_TRIALS = 10000

# Golden values recorded for seed 42. Any change to the random stream,
# the batching or the aggregation that alters these numbers is a
# regression.
GOLDEN_WHEN = {
    "count": 10000,
    "total": 73019,
    "total_sq": 557437,
    "minimum": 4,
    "maximum": 14,
    "mean": 7.3019,
    "variance": 2.426199009900985,
    "std": 1.557626081542353,
    "p85": 9,
    "p50": 7,
    "p20": 6,
}
GOLDEN_HOWMANY = {
    "count": 10000,
    "total": 229468,
    "total_sq": 5855074,
    "minimum": 0,
    "maximum": 45,
    "mean": 22.9468,
    "variance": 58.95766552655262,
    "std": 7.678389513859831,
    "p85": 31,
    "p50": 23,
    "p20": 16,
}


def run_when(seed=GOLDEN_SEED, trials=GOLDEN_TRIALS, backlog=50, **kwargs):
    stream = mc.RandomStream(VALUES, seed=seed)
    return mc.run_simulation(mc.when_trials(stream, backlog), trials, **kwargs)


def run_howmany(seed=GOLDEN_SEED, trials=GOLDEN_TRIALS, iterations=3, **kwargs):
    stream = mc.RandomStream(VALUES, seed=seed)
    return mc.run_simulation(mc.howmany_trials(stream, iterations), trials, **kwargs)


def stats_dict(result):
    stats = result.stats
    return {
        "count": stats.count,
        "total": stats.total,
        "total_sq": stats.total_sq,
        "minimum": stats.minimum,
        "maximum": stats.maximum,
        "mean": stats.mean,
        "variance": stats.variance,
        "std": stats.std,
        "p85": result.percentile(85),
        "p50": result.percentile(50),
        "p20": result.percentile(20),
    }


class GoldenRegressionTest(unittest.TestCase):
    """Same seed must always reproduce the recorded golden values."""

    def test_when_golden(self):
        self.assertEqual(stats_dict(run_when()), GOLDEN_WHEN)

    def test_howmany_golden(self):
        self.assertEqual(stats_dict(run_howmany()), GOLDEN_HOWMANY)

    def test_same_seed_same_samples(self):
        first = run_when(seed=123, trials=500)
        second = run_when(seed=123, trials=500)
        self.assertEqual(first.samples, second.samples)

    def test_different_seed_different_samples(self):
        first = run_when(seed=1, trials=500)
        second = run_when(seed=2, trials=500)
        self.assertNotEqual(first.samples, second.samples)


class RecordingProgress:
    def __init__(self):
        self.updates = []

    def update(self, done):
        self.updates.append(done)


class BatchInvarianceTest(unittest.TestCase):
    """The number of batches may only affect progress reporting."""

    def test_batch_size_does_not_change_statistics(self):
        reference = run_when(batch_size=1)
        for batch_size in (2, 3, 7, 333, 1000, 10 ** 9):
            with self.subTest(batch_size=batch_size):
                result = run_when(batch_size=batch_size)
                self.assertEqual(result.samples, reference.samples)
                self.assertEqual(stats_dict(result), stats_dict(reference))

    def test_batch_size_only_changes_progress(self):
        trials = 1000
        for batch_size, expected_calls in ((1, 1000), (7, 143), (1000, 1), (10 ** 9, 1)):
            with self.subTest(batch_size=batch_size):
                progress = RecordingProgress()
                run_when(trials=trials, batch_size=batch_size, progress=progress)
                self.assertEqual(len(progress.updates), expected_calls)
                self.assertEqual(progress.updates[-1], trials)


class StatisticsCorrectnessTest(unittest.TestCase):
    """Aggregation must match the stdlib statistics module."""

    def test_running_stats_against_statistics_module(self):
        rng = __import__("random").Random(7)
        data = [rng.gauss(10.0, 3.0) for _ in range(1000)]
        stats = mc.RunningStats()
        for value in data:
            stats.add(value)
        self.assertEqual(stats.count, len(data))
        self.assertEqual(stats.minimum, min(data))
        self.assertEqual(stats.maximum, max(data))
        self.assertAlmostEqual(stats.mean, statistics.mean(data), places=12)
        self.assertAlmostEqual(stats.variance, statistics.variance(data), places=9)
        self.assertAlmostEqual(stats.std, statistics.stdev(data), places=9)

    def test_variance_is_bessel_corrected(self):
        data = (2, 4, 4, 4, 5, 5, 7, 9)
        stats = mc.RunningStats()
        for value in data:
            stats.add(value)
        self.assertAlmostEqual(stats.variance, 32 / 7)
        self.assertEqual(stats.variance, statistics.variance(data))
        self.assertAlmostEqual(stats.std, math.sqrt(32 / 7))

    def test_merge_matches_single_pass(self):
        one_pass = mc.RunningStats()
        for value in VALUES:
            one_pass.add(value)
        merged = mc.RunningStats()
        for batch in mc.batched(VALUES, 7):
            batch_stats = mc.RunningStats()
            for value in batch:
                batch_stats.add(value)
            merged.merge(batch_stats)
        self.assertEqual(merged.count, one_pass.count)
        self.assertEqual(merged.total, one_pass.total)
        self.assertEqual(merged.total_sq, one_pass.total_sq)
        self.assertEqual(merged.minimum, one_pass.minimum)
        self.assertEqual(merged.maximum, one_pass.maximum)

    def test_unbiased_mean_and_variance_estimates(self):
        # Trials are sums of 3 i.i.d. draws from the historical values,
        # so the true mean is 3*mean and the true variance is
        # 3*pvariance of the data. Averaging many independent estimates
        # must converge to those values (unbiasedness).
        true_mean = 3 * statistics.mean(VALUES)
        true_variance = 3 * statistics.pvariance(VALUES)
        means, variances = [], []
        for rep in range(300):
            result = run_howmany(seed=rep, trials=40)
            means.append(result.stats.mean)
            variances.append(result.stats.variance)
        self.assertLess(abs(statistics.mean(means) - true_mean) / true_mean, 0.01)
        self.assertLess(
            abs(statistics.mean(variances) - true_variance) / true_variance, 0.05
        )


class PercentileTest(unittest.TestCase):
    def test_closest_observation_semantics(self):
        data = [4, 1, 3, 2]
        self.assertEqual(mc.percentile_closest_observation(data, 0), 1)
        self.assertEqual(mc.percentile_closest_observation(data, 25), 1)
        self.assertEqual(mc.percentile_closest_observation(data, 50), 3)
        self.assertEqual(mc.percentile_closest_observation(data, 75), 3)
        self.assertEqual(mc.percentile_closest_observation(data, 100), 4)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            mc.percentile_closest_observation([], 50)


class TrialsValidationTest(unittest.TestCase):
    def test_zero_trials_rejected(self):
        with self.assertRaises(ValueError):
            run_when(trials=0)

    def test_negative_trials_rejected(self):
        with self.assertRaises(ValueError):
            run_when(trials=-100)

    def test_zero_batch_size_rejected(self):
        with self.assertRaises(ValueError):
            run_when(trials=10, batch_size=0)


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class ProgressBarTest(unittest.TestCase):
    def test_bar_line_is_terminated_on_exception(self):
        stream = FakeTTY()
        bar = mc.ProgressBar(10, stream=stream)
        with self.assertRaises(RuntimeError):
            with bar:
                bar.update(3)
                raise RuntimeError("boom")
        output = stream.getvalue()
        self.assertIn("3/10", output)
        self.assertTrue(output.endswith("\n"))

    def test_bar_reaches_total_on_success(self):
        stream = FakeTTY()
        with mc.ProgressBar(4, stream=stream) as bar:
            for done in range(1, 5):
                bar.update(done)
        output = stream.getvalue()
        self.assertIn("4/4", output)
        self.assertTrue(output.endswith("\n"))

    def test_non_tty_stays_silent(self):
        stream = io.StringIO()  # not a TTY
        with mc.ProgressBar(4, stream=stream) as bar:
            bar.update(4)
        self.assertEqual(stream.getvalue(), "")


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "montecarlocli.py", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_when_golden_output(self):
        proc = self.run_cli("-s", "42", "-t", "10000", "when", "-b", "50")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout,
            "Reading ./data/example.csv.\n"
            "* Forecast for 50 pbi on 10000 trials *\n"
            "* Number of iterations forecasted with 85.0% confidence: 9\n"
            "* Number of iterations forecasted with 50.0% confidence: 7\n"
            "* Number of iterations forecasted with 20.0% confidence: 6\n"
            "* Min iterations 4.00 *\n"
            "* Max iterations 14.00 *\n"
            "* Avg iterations 7.30 *\n"
            "* Standard deviations  1.56 *\n",
        )

    def test_howmany_golden_output(self):
        proc = self.run_cli("-s", "42", "-t", "10000", "howmany", "-i", "3")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            proc.stdout,
            "Reading ./data/example.csv.\n"
            "* Forecast for 3 with 10000 trials *\n"
            "* Number of pbi forecasted with 85.0% confidence: 15\n"
            "* Number of pbi forecasted with 50.0% confidence: 23\n"
            "* Number of pbi forecasted with 20.0% confidence: 30\n"
            "* Min pbi 0 *\n"
            "* Max pbi 45 *\n"
            "* Avg pbi 23 *\n"
            "* Standard deviations  8 *\n",
        )

    def test_same_seed_same_cli_output(self):
        first = self.run_cli("-s", "9", "-t", "2000", "when")
        second = self.run_cli("-s", "9", "-t", "2000", "when")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)

    def test_zero_trials_exit_code(self):
        proc = self.run_cli("-t", "0", "when")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("positive integer", proc.stderr)

    def test_negative_trials_exit_code(self):
        proc = self.run_cli("-t", "-5", "howmany")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("positive integer", proc.stderr)

    def test_invalid_percentile_exit_code(self):
        proc = self.run_cli("-p", "1.5", "when")
        self.assertEqual(proc.returncode, 1)
        self.assertIn("Percentiles should be between 0 excluded and 1 included",
                      proc.stdout)


if __name__ == "__main__":
    unittest.main()

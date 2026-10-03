"""Monte Carlo forecasting CLI.

Pure standard-library implementation. The simulation is built from three
decoupled components:

* ``RandomStream`` -- the random number stream. A single seeded stream of
  samples drawn with replacement from the historical data. Given the same
  seed it always produces the same sequence, no matter how the work is
  split into batches.
* ``batch_sizes`` / ``run_simulation`` -- the batching layer. It only
  decides how many trials are executed between progress updates; it never
  influences which random values are drawn.
* ``Aggregator`` -- the aggregation layer. It consumes trial results and
  produces the final statistics (min/max/mean, unbiased sample standard
  deviation and closest-observation percentiles).
"""

import argparse
import csv
import math
import os
import random
import sys

DEFAULT_TRIALS = 10000
DEFAULT_PERCENTILES = (0.20, 0.50, 0.85)
DEFAULT_FILE = "./data/example.csv"
DEFAULT_SEED = 20241007
SEED_ENV_VAR = "MONTECARLO_SEED"
PROGRESS_BATCHES = 100


class RandomStream:
    """Deterministic stream of samples drawn with replacement from values."""

    def __init__(self, values, seed):
        if not values:
            raise ValueError("cannot sample from an empty data set")
        self._values = list(values)
        self._rng = random.Random(seed)

    def sample(self):
        return self._values[self._rng.randrange(len(self._values))]


def when_trial(stream, backlog_size):
    """Iterations needed to burn down ``backlog_size`` pbi."""
    iterations = 0
    remaining = backlog_size
    while remaining > 0:
        remaining -= stream.sample()
        iterations += 1
    return iterations


def howmany_trial(stream, iterations):
    """Pbi completed in ``iterations`` iterations."""
    total = 0
    for _ in range(iterations):
        total += stream.sample()
    return total


def batch_sizes(trials, batch_count):
    """Split ``trials`` into consecutive, roughly equal batch sizes.

    Batching only controls how often progress is reported. Because every
    batch consumes the same shared random stream in order, the number of
    batches never changes the final statistics.
    """
    if trials < 1:
        raise ValueError("trials must be a positive integer")
    batch_count = max(1, min(batch_count, trials))
    base, extra = divmod(trials, batch_count)
    return [base + (1 if index < extra else 0) for index in range(batch_count)]


class Aggregator:
    """Incremental statistics over trial results (Welford's algorithm)."""

    def __init__(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = None
        self._max = None
        self._values = []

    def update(self, results):
        for value in results:
            self._count += 1
            delta = value - self._mean
            self._mean += delta / self._count
            self._m2 += delta * (value - self._mean)
            self._min = value if self._min is None else min(self._min, value)
            self._max = value if self._max is None else max(self._max, value)
            self._values.append(value)

    @property
    def count(self):
        return self._count

    @property
    def minimum(self):
        return self._min

    @property
    def maximum(self):
        return self._max

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        """Unbiased sample variance (ddof=1)."""
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def stdev(self):
        return math.sqrt(self.variance)

    def percentile(self, p):
        """Closest-observation percentile, ``p`` in (0, 1]."""
        if not self._values:
            raise ValueError("no samples to aggregate")
        ordered = sorted(self._values)
        index = int(round(p * (len(ordered) - 1)))
        return ordered[index]


class Progress:
    """Minimal stderr progress display that always leaves a clean line."""

    def __init__(self, total, stream=None):
        self._total = total
        self._done = 0
        self._stream = stream if stream is not None else sys.stderr

    def advance(self, amount):
        self._done += amount
        percent = 100.0 * self._done / self._total
        self._stream.write(
            "\rProcessing... {}/{} ({:5.1f}%)".format(self._done, self._total, percent)
        )
        self._stream.flush()

    def close(self):
        self._stream.write("\n")
        self._stream.flush()


def run_simulation(trials, trial_fn, batch_count=PROGRESS_BATCHES, progress=None):
    """Run ``trials`` trials in batches and aggregate their results."""
    aggregator = Aggregator()
    for size in batch_sizes(trials, batch_count):
        aggregator.update(trial_fn() for _ in range(size))
        if progress is not None:
            progress.advance(size)
    return aggregator


def load_values(path):
    """Read the ``values`` column of the CSV data file."""
    values = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "values" not in reader.fieldnames:
            raise ValueError("{} has no 'values' column".format(path))
        for row in reader:
            number = float(row["values"])
            values.append(int(number) if number.is_integer() else number)
    if not values:
        raise ValueError("{} contains no data rows".format(path))
    return values


def resolve_seed(environ=None):
    """Seed from MONTECARLO_SEED, falling back to a fixed default."""
    environ = os.environ if environ is None else environ
    raw = environ.get(SEED_ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_SEED
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            "{} must be an integer, got {!r}".format(SEED_ENV_VAR, raw)
        )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="montecarlocli",
        description="This software will help you to forecast when a backlog "
        "of a certain size will be done or how many PBI you can do in a "
        "certain amount of iterations",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of trials for the montecarlo simulation (Default: 10000).",
    )
    parser.add_argument(
        "-f",
        "--file",
        default=DEFAULT_FILE,
        help="The CSV file containing the the input data (look at the github page).",
    )
    parser.add_argument(
        "-p",
        "--percentiles",
        type=float,
        action="append",
        default=None,
        help="The percentiles you want to look at default (0.20, 0.50, 0.85) "
        "all should be smaller or equal to 1",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    when_parser = subparsers.add_parser(
        "when", help="Command to forecast when a certain amount of pbi can be done."
    )
    when_parser.add_argument(
        "-b",
        "--backlog_size",
        type=int,
        default=50,
        help="Size of the backlog to forecast (Default: 50).",
    )

    howmany_parser = subparsers.add_parser(
        "howmany",
        help="Command to forecast how many pbi can be don ein a certain amount of time.",
    )
    howmany_parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=3,
        help="The number of iterations for which you are forecasting.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    percentiles = sorted(args.percentiles or DEFAULT_PERCENTILES, reverse=True)
    if any(p <= 0 or p > 1 for p in percentiles):
        print("Percentiles should be between 0 excluded and 1 included")
        return 1

    if args.trials < 1:
        print(
            "Number of trials must be a positive integer, got {}".format(args.trials),
            file=sys.stderr,
        )
        return 2

    print("Reading {}.".format(args.file))
    try:
        values = load_values(args.file)
        seed = resolve_seed()
    except (OSError, ValueError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 2

    stream = RandomStream(values, seed)
    progress = Progress(args.trials)
    try:
        if args.command == "when":
            if not any(value > 0 for value in values):
                print(
                    "Error: the data must contain at least one positive value",
                    file=sys.stderr,
                )
                return 2
            aggregator = run_simulation(
                args.trials,
                lambda: when_trial(stream, args.backlog_size),
                progress=progress,
            )
            report_when(args, percentiles, aggregator)
        else:
            aggregator = run_simulation(
                args.trials,
                lambda: howmany_trial(stream, args.iterations),
                progress=progress,
            )
            report_howmany(args, percentiles, aggregator)
    finally:
        progress.close()
    return 0


def report_when(args, percentiles, aggregator):
    print("* Forecast for {} pbi on {} trials *".format(args.backlog_size, args.trials))
    for p in percentiles:
        print(
            "* Number of iterations forecasted with {}% confidence: {}".format(
                p * 100, aggregator.percentile(p)
            )
        )
    print("* Min iterations {:.2f} *".format(aggregator.minimum))
    print("* Max iterations {:.2f} *".format(aggregator.maximum))
    print("* Avg iterations {:.2f} *".format(aggregator.mean))
    print("* Standard deviations  {:.2f} *".format(aggregator.stdev))


def report_howmany(args, percentiles, aggregator):
    print("* Forecast for {} with {} trials *".format(args.iterations, args.trials))
    for p in percentiles:
        print(
            "* Number of pbi forecasted with {}% confidence: {}".format(
                p * 100, aggregator.percentile(1 - p)
            )
        )
    print("* Min pbi {:.0f} *".format(aggregator.minimum))
    print("* Max pbi {:.0f} *".format(aggregator.maximum))
    print("* Avg pbi {:.0f} *".format(aggregator.mean))
    print("* Standard deviations  {:.0f} *".format(aggregator.stdev))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)

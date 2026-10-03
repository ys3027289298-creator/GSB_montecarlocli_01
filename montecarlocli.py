"""Monte Carlo forecasting command line tool.

The simulation pipeline is split into three decoupled pieces:

- :class:`RandomStream` -- the seedable random number stream. The same
  seed always produces the same sequence of draws.
- :func:`batched` -- the batch layer. It only groups the sequential
  stream of trials, so the batch size only affects how often progress
  is reported, never the resulting statistics.
- :class:`RunningStats` -- the aggregation layer. Per-batch statistics
  are merged into the final result and expose min/max/mean plus the
  unbiased sample variance.

Only the Python standard library is used.
"""

import argparse
import csv
import math
import random
import sys
from itertools import islice

DEFAULT_PERCENTILES = (0.20, 0.50, 0.85)
DEFAULT_BATCH_SIZE = 1000


class RandomStream:
    """An endless, seedable stream of draws from the historical values."""

    def __init__(self, values, seed=None):
        values = list(values)
        if not values:
            raise ValueError("cannot draw from an empty data set")
        self._values = values
        self._rng = random.Random(seed)

    def draw(self):
        return self._values[self._rng.randrange(len(self._values))]


def when_trials(stream, backlog_size):
    """Yield the number of iterations needed to burn down each backlog."""
    while True:
        remaining = backlog_size
        iterations = 0
        while remaining >= 0:
            remaining -= stream.draw()
            iterations += 1
        yield iterations


def howmany_trials(stream, iterations):
    """Yield the number of pbi completed in a fixed number of iterations."""
    while True:
        completed = 0
        for _ in range(iterations):
            completed += stream.draw()
        yield completed


def batched(items, size):
    """Group ``items`` into lists of at most ``size`` elements."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class RunningStats:
    """Streaming min/max/mean and unbiased sample variance.

    Partial statistics collected on any batch can be merged with
    :meth:`merge`. For integer samples (as shipped in the CSV data) the
    running sums are exact Python integers, so results are identical no
    matter how the samples were split into batches.
    """

    def __init__(self):
        self.count = 0
        self.total = 0
        self.total_sq = 0
        self.minimum = None
        self.maximum = None

    def add(self, value):
        self.count += 1
        self.total += value
        self.total_sq += value * value
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value

    def merge(self, other):
        if other.count == 0:
            return self
        if self.minimum is None or other.minimum < self.minimum:
            self.minimum = other.minimum
        if self.maximum is None or other.maximum > self.maximum:
            self.maximum = other.maximum
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq
        return self

    @property
    def mean(self):
        if self.count == 0:
            raise ValueError("cannot compute the mean of zero samples")
        return self.total / self.count

    @property
    def variance(self):
        """Unbiased sample variance (Bessel-corrected, ddof = 1)."""
        if self.count < 2:
            return 0.0
        variance = (self.total_sq - self.total * self.total / self.count) / (
            self.count - 1
        )
        return variance if variance > 0.0 else 0.0

    @property
    def std(self):
        return math.sqrt(self.variance)


def percentile_closest_observation(values, q):
    """Return the percentile ``q`` (0..100) using numpy's
    ``closest_observation`` method.

    The 0-based virtual index is ``q/100 * n - 0.5``; the closest
    integer (round half to even, like ``numpy.around``) selects the
    observation. The index is clipped to the valid range.
    """
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("cannot compute a percentile of zero samples")
    index = round(q / 100.0 * count - 0.5)
    index = min(max(index, 0), count - 1)
    return ordered[index]


class SimulationResult:
    """Final aggregated result of a simulation run."""

    def __init__(self, stats, samples):
        self.stats = stats
        self.samples = samples

    def percentile(self, q):
        return percentile_closest_observation(self.samples, q)


def run_simulation(trial_stream, trials, batch_size=DEFAULT_BATCH_SIZE, progress=None):
    """Consume exactly ``trials`` samples from the (infinite) trial stream.

    The random stream is consumed sequentially, independent of how
    samples are split into batches: ``batch_size`` only controls how
    often ``progress.update(done)`` is called.
    """
    if not isinstance(trials, int) or trials <= 0:
        raise ValueError("number of trials must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    stats = RunningStats()
    samples = []
    done = 0
    for batch in batched(islice(trial_stream, trials), batch_size):
        batch_stats = RunningStats()
        for value in batch:
            batch_stats.add(value)
        stats.merge(batch_stats)
        samples.extend(batch)
        done += len(batch)
        if progress is not None:
            progress.update(done)
    return SimulationResult(stats, samples)


class ProgressBar:
    """A tiny progress bar written to a stream.

    Nothing is emitted when the stream is not a TTY. The bar always
    terminates its own line on close (including after an exception, via
    the context manager protocol), so tracebacks never overwrite the
    bar or start on a half-written line.
    """

    def __init__(self, total, description="Processing...", stream=None):
        self.total = total
        self.description = description
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = self.stream.isatty()
        self._closed = False

    def update(self, done):
        if not self.enabled or self._closed:
            return
        width = 40
        ratio = min(done / self.total, 1.0) if self.total else 1.0
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        self.stream.write(f"\r{self.description} |{bar}| {done}/{self.total}")
        self.stream.flush()

    def close(self):
        if self.enabled and not self._closed:
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def parse_number(text):
    """Parse a CSV cell as int when possible, otherwise as float."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        return float(text)


def load_values(path):
    """Read the ``values`` column of the input CSV file."""
    values = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "values" not in reader.fieldnames:
            raise ValueError(f"{path}: missing 'values' column")
        for row in reader:
            values.append(parse_number(row["values"]))
    if not values:
        raise ValueError(f"{path}: no data rows")
    return values


def build_parser():
    parser = argparse.ArgumentParser(
        prog="montecarlocli",
        description=(
            "This software will help you to forecast when a backlog of a "
            "certain size will be done or how many PBI you can do in a "
            "certain amount of iterations"
        ),
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        default=10000,
        help="Number of trials for the montecarlo simulation (Default: 10000).",
    )
    parser.add_argument(
        "-f",
        "--file",
        default="./data/example.csv",
        type=str,
        help="The CSV file containing the the input data (look at the github page).",
    )
    parser.add_argument(
        "-p",
        "--percentiles",
        help=(
            "The percentiles you want to look at default (0.20, 0.50, 0.85) "
            "all should be smaller or equal to 1"
        ),
        default=None,
        type=float,
        action="append",
        metavar="FLOAT",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Seed of the random number generator for reproducible runs (Default: random).",
    )
    subparsers = parser.add_subparsers(dest="command")

    when = subparsers.add_parser(
        "when", help="Command to forecast when a certain amount of pbi can be done."
    )
    when.add_argument(
        "-b",
        "--backlog_size",
        type=int,
        default=50,
        help="Size of the backlog to forecast (Default: 50).",
    )

    howmany = subparsers.add_parser(
        "howmany",
        help="Command to forecast how many pbi can be don ein a certain amount of time.",
    )
    howmany.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=3,
        help="The number of iterations for which you are forecasting.",
    )
    return parser


def command_when(stream, trials, backlog_size, percentiles):
    with ProgressBar(trials) as progress:
        result = run_simulation(
            when_trials(stream, backlog_size), trials, progress=progress
        )
    stats = result.stats
    print(f"* Forecast for {backlog_size} pbi on {trials} trials *")
    for percentile in percentiles:
        value = result.percentile(percentile * 100)
        print(
            f"* Number of iterations forecasted with {percentile * 100}% "
            f"confidence: {value}"
        )
    print(f"* Min iterations {stats.minimum:.2f} *")
    print(f"* Max iterations {stats.maximum:.2f} *")
    print(f"* Avg iterations {stats.mean:.2f} *")
    print(f"* Standard deviations  {stats.std:.2f} *")


def command_howmany(stream, trials, iterations, percentiles):
    with ProgressBar(trials) as progress:
        result = run_simulation(
            howmany_trials(stream, iterations), trials, progress=progress
        )
    stats = result.stats
    print(f"* Forecast for {iterations} with {trials} trials *")
    for percentile in percentiles:
        value = result.percentile((1 - percentile) * 100)
        print(
            f"* Number of pbi forecasted with {percentile * 100}% "
            f"confidence: {value}"
        )
    print(f"* Min pbi {stats.minimum:.0f} *")
    print(f"* Max pbi {stats.maximum:.0f} *")
    print(f"* Avg pbi {stats.mean:.0f} *")
    print(f"* Standard deviations  {stats.std:.0f} *")


def cli(argv=None):
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        sys.exit(0)

    percentiles = (
        args.percentiles if args.percentiles is not None else list(DEFAULT_PERCENTILES)
    )
    if any(p <= 0 or p > 1 for p in percentiles):
        print("Percentiles should be between 0 excluded and 1 included")
        sys.exit(1)
    percentiles = sorted(percentiles, reverse=True)

    if args.trials <= 0:
        print(
            f"Number of trials should be a positive integer, got {args.trials}.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Reading {args.file}.")
    try:
        values = load_values(args.file)
    except (OSError, ValueError) as exc:
        print(f"Could not read input data: {exc}", file=sys.stderr)
        sys.exit(2)

    stream = RandomStream(values, seed=args.seed)
    if args.command == "when":
        command_when(stream, args.trials, args.backlog_size, percentiles)
    else:
        command_howmany(stream, args.trials, args.iterations, percentiles)


if __name__ == "__main__":
    cli()

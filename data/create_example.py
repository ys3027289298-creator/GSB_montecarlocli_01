"""Generate an example CSV file for montecarlocli (stdlib only)."""

import argparse
import csv
import datetime
import random


def parse_date(text):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"cannot parse date: {text}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--start-date", default="1/1/2024", help="starting date")
    parser.add_argument("-s", "--size", default=30, type=int, help="number of line to generate")
    parser.add_argument(
        "-m",
        "--max-value",
        default=30,
        type=int,
        help="max random number to generate (inclusive) min is 0",
    )
    parser.add_argument(
        "-f",
        "--file-name",
        default="example.csv",
        help="output file name, file with the same name are overwritten",
    )
    args = parser.parse_args(argv)

    start = parse_date(args.start_date)
    rng = random.Random()
    with open(args.file_name, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dates", "values"])
        for offset in range(args.size):
            date = start + datetime.timedelta(days=offset)
            writer.writerow([date.isoformat(), rng.randint(0, args.max_value)])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Utility to report empty files under the specified roots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List


def find_empty_files(paths: Iterable[Path], recursive: bool) -> Iterable[Path]:
    """Yield files that have zero bytes."""
    for root in paths:
        if not root.exists():
            continue

        if root.is_file():
            if root.stat().st_size == 0:
                yield root
            continue

        if recursive:
            iterator = root.rglob("*")
        else:
            iterator = root.iterdir()

        for candidate in iterator:
            if candidate.is_file() and candidate.stat().st_size == 0:
                yield candidate


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Look for zero-byte files so you can re-generate corrupted data."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to scan.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Do not descend into directories.",
    )

    args = parser.parse_args(argv)
    empty_files = list(find_empty_files(args.paths, args.recursive))

    if not empty_files:
        print("No empty files found.")
        return 0

    print("Empty files:")
    for empty in empty_files:
        print(f"  {empty}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ihepc_forecast.data import prepare_train_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare daily UCI household-power data")
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "household_power_consumption.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--split-date",
        default=None,
        help="Optional YYYY-MM-DD override; default uses the final 365 complete days",
    )
    parser.add_argument(
        "--skip-minute-splits",
        action="store_true",
        help="Do not write the large raw minute-level train/test CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_train_test(
        args.raw,
        args.output_dir,
        split_date=args.split_date,
        test_days=365,
        save_raw_minute_splits=not args.skip_minute_splits,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

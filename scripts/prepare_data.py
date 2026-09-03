#!/usr/bin/env python3
"""Run Phase 0 data preparation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import run_preprocess_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Tiki product data (Phase 0)")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "product_tiki_data.json",
        help="Path to raw product JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory for products_clean.jsonl and metadata",
    )
    parser.add_argument(
        "--boilerplate-threshold",
        type=float,
        default=0.05,
        help="Sentence frequency threshold for boilerplate removal",
    )
    args = parser.parse_args()

    stats = run_preprocess_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        boilerplate_threshold=args.boilerplate_threshold,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

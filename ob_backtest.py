"""Supported offline entry point for the synthetic market-making comparison.

This script keeps the historical filename but now delegates to the reproducible
synthetic workflow described in README.md.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("pipeline").joinpath("4_backtest.py")), run_name="__main__")

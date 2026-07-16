from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_ALGOS = ("baseline", "obi")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("configuration must be a mapping")
    if cfg.get("delete_inputs_after", False):
        raise ValueError("delete_inputs_after must remain disabled for the public synthetic workflow")
    return cfg


def build_command(repo_root: Path, cfg: dict[str, Any], algo: str) -> list[str]:
    fixture = repo_root / cfg["fixture"]
    output_dir = repo_root / cfg["output_dir"]
    return [
        "cargo",
        "run",
        "--quiet",
        "--bin",
        "statmm-bin",
        "--",
        "--fixture",
        str(fixture),
        "--output-dir",
        str(output_dir),
        "--algo",
        algo,
        "--tick-size",
        str(cfg["tick_size"]),
        "--order-qty",
        str(cfg["order_qty"]),
        "--max-inventory",
        str(cfg["max_inventory"]),
        "--half-spread",
        str(cfg["half_spread"]),
        "--inventory-skew",
        str(cfg["inventory_skew"]),
        "--entry-latency-ns",
        str(cfg["entry_latency_ns"]),
        f"--maker-fee={cfg['maker_fee']}",
        "--signal-levels",
        str(cfg["signal_levels"]),
        "--signal-window",
        str(cfg["signal_window"]),
        "--alpha-scale",
        str(cfg["alpha_scale"]),
    ]


def run_case(repo_root: Path, cfg: dict[str, Any], algo: str) -> dict[str, Any]:
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"unsupported algo: {algo}")
    command = build_command(repo_root, cfg, algo)
    subprocess.run(command, cwd=repo_root, check=True)
    summary_path = repo_root / cfg["output_dir"] / f"{algo}_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the supported synthetic baseline and OBI comparison")
    parser.add_argument("-c", "--config", default="pipeline/backtest_config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / args.config)

    summaries = {algo: run_case(repo_root, cfg, algo) for algo in SUPPORTED_ALGOS}
    output_dir = repo_root / cfg["output_dir"]
    print("Synthetic market-making comparison complete:")
    for algo, summary in summaries.items():
        print(
            f"  {algo}: final_mtm={summary['final_mark_to_market']:.6f}, "
            f"inventory={summary['final_inventory']:.6f}, fills={summary['fills']}"
        )
    print(f"Artifacts written to {output_dir}")


if __name__ == "__main__":
    main()

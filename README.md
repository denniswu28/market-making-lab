# statmm

Event-driven market-making research lab for testing order-book signals, latency, queue assumptions, inventory controls, and execution costs with reproducible synthetic examples.

## Research question

How can order-book and short-horizon statistical signals be incorporated into an event-driven market-making strategy, and under what assumptions do they improve quote placement and inventory control relative to a signal-free baseline?

## What is supported from a fresh clone

- Linux Rust build pinned to `hftbacktest` commit `6557e564ac984c46405a0ddfd08272f5009abc2e` with only the upstream `backtest` feature enabled.
- A deterministic offline synthetic level-2 fixture at `fixtures/synthetic_l2.csv`.
- One public comparison between:
  - a signal-free market-making baseline; and
  - an order-book-imbalance (OBI) variant with trailing-window warm-up.
- A minimal Python wrapper that runs both variants without exchange credentials, vendor data, network access, or a sibling checkout.

## Architecture

- `src/synthetic.rs` contains the repository-owned synthetic event-driven lab used for the public example and invariant tests.
- `src/algo.rs` and the Rust examples remain the adaptation boundary to upstream `hftbacktest` concepts and APIs.
- `pipeline/4_backtest.py` is the supported Python entry point for the synthetic public workflow.
- `pipeline/*.yaml` now use portable placeholders for optional user-supplied real-data research only.

## Dennis Wu's demonstrated contribution in this repository

Dennis Wu designed and validated the market-making strategy application, the surrounding research pipeline, and the repository-specific synthetic validation workflow in this repository. The upstream simulation engine, queue/fill models, connectors, data tooling, and tutorial base come from `nkaz001/hftbacktest` and are not claimed here as independently created work.

See `PROVENANCE.md` for the file-by-file attribution map.

## Synthetic quick start

Rust 1.89 and Python 3.11 are the supported toolchain versions.

```bash
python -m pip install -r requirements.txt
python pipeline/4_backtest.py -c pipeline/backtest_config.yaml
```

That command writes deterministic baseline and OBI artifacts to `out/synthetic/`:

- `baseline.csv`
- `baseline_summary.json`
- `obi.csv`
- `obi_summary.json`

The output is a reproducible simulation artifact, not investment performance and not evidence of live deployability.

## Optional user-supplied real-data path

The numbered `pipeline/` conversion and latency scripts are kept only as optional, user-run research scaffolding for self-supplied data under the user's own vendor terms. Current repository evidence demonstrates Tardis conversion and Binance Futures metadata/data paths. Do **not** treat the repository as verified general Binance or OKX production integration.

## Tested assumptions and limitations

The public synthetic workflow explicitly assumes:

- event processing in local-timestamp order, with exchange timestamp as a deterministic tie-breaker;
- quote updates that become active only after the configured entry latency;
- no lookahead in OBI warm-up or signal standardization;
- full-fill/no-partial-fill behavior once the synthetic book crosses a resting quote;
- maker-fee accounting with signed fees or rebates;
- inventory caps enforced by suppressing any same-side quote that would exceed the cap;
- cancellations and replacements applied when a new desired quote differs from the currently active quote.

The supported public example is intentionally narrow:

- **Implemented and validated:** signal-free baseline, OBI variant, deterministic synthetic fixture, invariant tests, portable configs, CI.
- **Experimental / user-supplied only:** historical data ingestion, latency generation from vendor feeds, grid search, alternate alpha families, and notebooks beyond the synthetic walkthrough.

## Validation commands

```bash
cargo fmt --all -- --check
cargo clippy --lib --bins --tests -- -D warnings
cargo test --all-targets
python -m py_compile pipeline/4_backtest.py ob_backtest.py
python -m unittest discover -s tests -p 'test_*.py'
python pipeline/4_backtest.py -c pipeline/backtest_config.yaml
```

## Repository hygiene notes

- Public fixtures are synthetic only.
- `delete_inputs_after` remains disabled by default everywhere it appears in tracked configuration.
- Tracked notebooks are intentionally unexecuted and contain no local data paths or public performance claims.
- Live trading is outside scope and is not part of the supported public example.

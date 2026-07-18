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
- After dependencies are installed, a minimal Python wrapper runs both variants without runtime network or exchange calls, exchange credentials, vendor data, or a sibling checkout.
- An optional, offline-only HftBacktest `.npz` grid-search adapter for user-supplied research data.

## Architecture

- `src/synthetic.rs` contains the repository-owned synthetic event-driven lab used for the public example and invariant tests.
- `src/algo.rs` and the Rust examples remain the adaptation boundary to upstream `hftbacktest` concepts and APIs.
- `pipeline/4_backtest.py` is the supported Python entry point for the synthetic public workflow.
- `pipeline/5_gridsearch.py` validates and consumes externally produced level-2 HftBacktest NPZ files for the optional research workflow.
- `pipeline/snapshot_manifest.py` creates the required integrity sidecar for an externally produced initial-snapshot NPZ file.
- `pipeline/*.yaml` now use portable placeholders for optional user-supplied real-data research only.

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

The only workflow supported from a fresh clone is the synthetic quick start above. The optional
grid-search workflow starts from externally produced, user-supplied HftBacktest NPZ files. This
repository validates and consumes those files; it does not claim that vendor download, conversion,
latency generation, or snapshot generation is reproducible from a fresh clone. The older numbered
conversion, latency, and snapshot scripts remain legacy research scaffolding and require a separately
managed upstream HftBacktest environment and data obtained under the user's own vendor terms.
Current repository evidence demonstrates Tardis conversion and Binance Futures metadata/data paths.
Do **not** treat the repository as verified general Binance or OKX production integration.

The optional grid search uses the pinned upstream HftBacktest APIs and requires exact level-2 `.npz`
market-data and latency files, plus an optional initial-snapshot NPZ file, in the pinned upstream
schema. It accepts only level-2 depth and trade event kinds; depth and trade events require valid side
semantics. Before launching Rust, it rejects empty arrays, unsupported event kinds, invalid sides,
non-finite or non-positive market prices, negative depth quantities, non-positive trade quantities,
`local_ts < exch_ts`, processor-specific exchange/local ordering defects, negative per-row order
latencies, and unordered request/exchange interpolation axes. Zero depth quantity is retained as the
pinned engine's level-deletion operation, and concurrent requests may complete out of response order.
It does not accept the CSV fixture; use the synthetic quick start for that workflow. Its Python
analysis dependencies are isolated from the default installation:

```bash
python -m pip install -r requirements-research.txt
cargo build --release --example gridtrading_backtest_args
python pipeline/5_gridsearch.py --phase explore -c pipeline/gridsearch_config.yaml
```

When `initial_snapshot` is configured, the externally produced snapshot must have a JSON sidecar
named `<snapshot>.manifest.json`. Create it only after independently verifying the snapshot's
as-of timestamp and data provenance:

```bash
python pipeline/snapshot_manifest.py \
  --snapshot /path/to/snapshot.npz \
  --as-of-ns 1735689599999999999 \
  --source external-research-snapshot
```

The `source` value is a logical provenance label, not a machine-specific path. The command writes:

```json
{"as_of_ns": 1735689599999999999, "schema_version": 1, "snapshot_sha256": "<sha256>", "source": "external-research-snapshot"}
```

The grid search verifies the sidecar hash, requires `as_of_ns` to cover every event stored in the
snapshot, and requires it to be strictly earlier than the first replay event. It also requires only
snapshot-depth rows, positive tick/lot-aligned values, both book sides, and a strictly non-crossed
BBO. These checks prevent a same-period, future, or structurally invalid snapshot from initializing
the replay.

The explore phase runs train and validation only and writes
`gridsearch_validation_summary.csv`; it never reads or executes the test partition. After reviewing
that file, set `gridsearch.locked_candidate` to exactly one successful validation `candidate_id`,
then run the held-out evaluation separately:

```bash
python pipeline/5_gridsearch.py --phase test -c pipeline/gridsearch_config.yaml
```

The test command refuses to run without the persisted validation summary and a successful matching
candidate. The template deliberately leaves the lock as `TODO(Dennis)`; the pipeline does not
select a candidate or run validation and test concurrently. Phase summaries remain in stable
`symbol`, `candidate_id` order and never rank or recommend candidates by return. Explore mode also
refuses a stale `gridsearch_test_summary.csv` in the output directory, so held-out output cannot be
silently mixed into a later exploration run. Each successful research CLI run also writes a Rust
execution manifest and a grid-search artifact manifest. Result artifacts are accepted only when
they contain observations with the required numeric columns, monotonic timestamps, finite equity,
and positive prices. The held-out lock verifies
the candidate's strategy, fees, queue model, timing, engine source and binary identity, declared
partition plan, validation inputs, and exact validation artifacts before test execution. Retain the
validation inputs and artifacts until the held-out test is complete. `--skip-existing` resumes only
artifacts whose manifests and exact result files still match the current run specification. These
manifests use content identities and logical partitions for reproducibility across checkout and data
roots. Resolved paths are retained only as non-locking provenance. Keep the ignored `out/` directory
private and do not commit user-data research artifacts.

For VAMP, effective VAMP, and weighted-depth signals, `alpha_scale` is executed only with the
`zscore` transform. Supplying it with a non-z-score transform is rejected instead of silently
creating duplicate candidates. The Rust adapter treats `max_position` as a hard inventory boundary:
it limits new grid levels by remaining capacity, defers replacement submissions while same-side
operations are pending, and fails the run when a required cancel or submission fails.

## Tested assumptions and limitations

The public synthetic workflow explicitly assumes:

- event processing in local-timestamp order, with exchange timestamp as a deterministic tie-breaker;
- quote updates that become active only after the configured entry latency;
- no lookahead in OBI warm-up or signal standardization;
- full-fill/no-partial-fill behavior once the synthetic book crosses a resting quote;
- maker-fee accounting with signed fees or rebates;
- inventory caps enforced as a hard boundary by sizing the active grid to remaining capacity;
- cancellations and replacements applied when a new desired quote differs from the currently active quote.

The supported public example is intentionally narrow:

- **Implemented and validated:** signal-free baseline, OBI variant, deterministic synthetic fixture, invariant tests, portable configs, CI.
- **Experimental / user-supplied only:** historical data ingestion, latency generation from vendor feeds, grid search, alternate alpha families, and notebooks beyond the synthetic walkthrough.

## Validation commands

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --all-features
python -m py_compile pipeline/2_latency.py pipeline/3_snapshot.py pipeline/4_backtest.py pipeline/5_gridsearch.py pipeline/snapshot_manifest.py pipeline/snapshot_validator.py ob_backtest.py
python -m unittest discover -s tests -p 'test_*.py'
python pipeline/4_backtest.py -c pipeline/backtest_config.yaml
```

## Repository hygiene notes

- Public fixtures are synthetic only.
- `delete_inputs_after` remains disabled by default everywhere it appears in tracked configuration.
- Tracked notebooks are intentionally unexecuted and contain no local data paths or public performance claims.
- Live trading is outside scope and is not part of the supported public example.
- Upstream license notices for materially adapted `hftbacktest` code are preserved in
  `THIRD_PARTY_NOTICES.md`.

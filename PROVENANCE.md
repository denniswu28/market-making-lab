# Provenance and attribution map

This repository depends on and adapts the upstream MIT-licensed `nkaz001/hftbacktest` project. The canonical dependency is pinned in `Cargo.toml` to commit `6557e564ac984c46405a0ddfd08272f5009abc2e` with only the `backtest` feature enabled for the public build.

## File-by-file map

| Local path | Closest upstream reference | Boundary | Current status |
| --- | --- | --- | --- |
| `examples/gridtrading_backtest.rs` | `hftbacktest/examples/gridtrading_backtest.rs` | Upstream example name retained; local file now wraps the repository-owned synthetic fixture instead of vendor market data paths. | Supported as a synthetic baseline wrapper. |
| `examples/gridtrading_backtest_args.rs` | Upstream `hftbacktest/examples/gridtrading_backtest_args.rs` plus local strategy adaptation | Offline adapter over pinned upstream `Backtest`, L2 asset, latency, queue, fee, snapshot, and recorder APIs for user-supplied `.npz` data. | Experimental user-data research only; no live execution. |
| `src/algo.rs` | Upstream grid-trading and market-making tutorial patterns | Repository adaptation layer for OBI, VAMP, weighted-depth, and GLFT-style quote logic against upstream APIs. | Experimental / user-data-facing. |
| `ob_backtest.ipynb` | Upstream tutorial lineage, especially "Market Making with Alpha - Order Book Imbalance" | Historical notebook replaced with an unexecuted synthetic walkthrough to avoid redistributing unsupported claims or local paths. | Documentation-only. |
| `ob_backtest.py` | Historical notebook export / local orchestration | Retained filename now delegates to the supported synthetic workflow. | Supported offline entry point. |
| `pipeline/0_ticker.py` through `pipeline/6_*.py` | Upstream Python examples and data-preparation utilities | Repository-specific orchestration around user-supplied data conversion, latency generation, backtest execution, and summarization. | Experimental unless run with user-supplied data under vendor terms. |
| `pipeline/*.yaml` and `pipeline/templates/*.yaml` | Local configuration layer | Sanitized to portable placeholders or the synthetic public path; destructive cleanup remains opt-in only. | Public-safe configuration examples. |
| `src/synthetic.rs` | Repository-owned | Deterministic synthetic research harness for validating quote placement, latency, signal warm-up, fees, and inventory behavior without vendor data. | Supported public baseline. |

## Upstream scope that is not claimed as Dennis-originated here

The following remain upstream `hftbacktest` contributions and should not be described as independently created in this repository:

- the replay/backtest engine itself;
- queue and fill models;
- live connectors and execution support;
- data normalization utilities and tutorial foundations;
- the broader examples that this repository adapts.

## Repository-owned additions demonstrated here

This repository demonstrates Dennis Wu's strategy application and validation work around:

- selecting the market-making research question and public synthetic benchmark;
- shaping portable configuration and pipeline boundaries for reproducibility;
- adding repository-owned invariant tests for latency, inventory, fee, and signal behavior;
- defining the supported/offline-versus-experimental boundary for public review.

## License notes

- `statmm` is declared as MIT in `Cargo.toml`.
- Upstream `hftbacktest` is MIT-licensed at the pinned revision.
- The upstream copyright and permission notice is preserved verbatim in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the materially adapted files listed above.
- This file records engineering provenance only and does not make a broader legal conclusion beyond the visible repository evidence.

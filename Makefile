# ------------------------------------------------------------
# Makefile for statmm (matches your current file names)
# ------------------------------------------------------------

SHELL := /bin/bash

# -------- Settings (override with `make VAR=...`) -----------
CONDA       ?= conda
CONDA_ENV   ?= hftbacktest
ACTIVATE    = eval "$$($(CONDA) shell.bash hook)"; $(CONDA) activate $(CONDA_ENV);

# Configs
CFG_DIR                ?= pipeline
CONVERT_CFG            ?= $(CFG_DIR)/convert_config.yaml
LATENCY_CFG            ?= $(CFG_DIR)/latency_config.yaml
CONVERT_LATENCY_CFG    ?= $(CFG_DIR)/convert_latency_config.yaml
BACKTEST_CFG           ?= $(CFG_DIR)/backtest_config.yaml
GRIDSEARCH_CFG         ?= $(CFG_DIR)/gridsearch_config.yaml
SNAPSHOT_CFG           ?= $(CFG_DIR)/snapshot_config.yaml

# Python entry points (match your files)
PY_TICKER          ?= pipeline/0_ticker.py
PY_CONVERT         ?= pipeline/1_convert.py
PY_LATENCY         ?= pipeline/2_latency.py
PY_SNAPSHOT        ?= pipeline/3_snapshot.py
PY_BACKTEST        ?= pipeline/4_backtest.py
PY_GRIDSEARCH      ?= pipeline/5_gridsearch.py
PY_STATS           ?= pipeline/6_stats.py
PY_CONVERT_LATENCY ?= pipeline/1_2_convert_latency.py

# Rust
CARGO ?= cargo
BACKTEST_EXAMPLE ?= target/release/examples/gridtrading_backtest_args

# Output dir
OUT_DIR ?= out

# -------- Phonies -------------------------------------------
.PHONY: help fmt fmt-rs fmt-py fmt-check lint lint-rs lint-py \
        check build build-backtest ticker convert latency convert-latency \
        snapshot backtest gridsearch stats pipeline clean clean-out clean-all

# -------- Help ----------------------------------------------
help:
	@echo "Targets:"
	@echo "  fmt / fmt-check  - format/check Rust & Python"
	@echo "  lint             - clippy + ruff/flake8"
	@echo "  check            - cargo check"
	@echo "  build            - cargo build --release"
	@echo "  build-backtest   - build example: $(BACKTEST_EXAMPLE)"
	@echo "  ticker           - fetch Binance futures tickers.json"
	@echo "  convert          - 1_convert.py -c $(CONVERT_CFG) (strict)"
	@echo "  latency          - 2_latency.py -c $(LATENCY_CFG)"
	@echo "  convert-latency  - 1_2_convert_latency.py -c $(CONVERT_LATENCY_CFG) (strict)"
	@echo "  backtest         - 4_backtest.py -c $(BACKTEST_CFG)"
	@echo "  gridsearch       - 5_gridsearch.py -c $(GRIDSEARCH_CFG)"
	@echo "  snapshot         - 3_snapshot.py -c $(SNAPSHOT_CFG)"
	@echo "  pipeline         - convert -> latency -> build-backtest -> backtest"
	@echo "  clean / clean-out/ clean-all"

# -------- Formatting ----------------------------------------
fmt: fmt-rs fmt-py
fmt-rs:
	$(CARGO) fmt --all
fmt-py:
	@$(ACTIVATE) \
	if command -v black >/dev/null 2>&1; then black pipeline; else echo "[skip] black not found"; fi; \
	if command -v isort >/dev/null 2>&1; then isort pipeline; else echo "[skip] isort not found"; fi
fmt-check:
	$(CARGO) fmt --all -- --check
	@$(ACTIVATE) \
	if command -v black >/dev/null 2>&1; then black --check pipeline; else echo "[skip] black --check not found"; fi; \
	if command -v isort >/dev/null 2>&1; then isort --check-only pipeline; else echo "[skip] isort --check not found"; fi

# -------- Linting -------------------------------------------
lint: lint-rs lint-py
lint-rs:
	$(CARGO) clippy --all-targets --all-features -- -D warnings
lint-py:
	@$(ACTIVATE) \
	if command -v ruff >/dev/null 2>&1; then ruff check pipeline --fix; \
	elif command -v flake8 >/dev/null 2>&1; then flake8 pipeline; \
	else echo "[skip] ruff/flake8 not found"; fi

# -------- Build / Check -------------------------------------
check:
	$(CARGO) check --all-targets
build:
	$(CARGO) build --release
build-backtest:
	$(CARGO) build --example gridtrading_backtest_args --release
	@echo "Built: $(BACKTEST_EXAMPLE)"

# -------- Pipeline steps ------------------------------------
ticker:
	@$(ACTIVATE) python $(PY_TICKER)

convert:
	@$(ACTIVATE) python $(PY_CONVERT) -c $(CONVERT_CFG) --strict

latency:
	@$(ACTIVATE) python $(PY_LATENCY) -c $(LATENCY_CFG)

convert-latency:
	@$(ACTIVATE) python $(PY_CONVERT_LATENCY) -c $(CONVERT_LATENCY_CFG) --strict

snapshot:
	@$(ACTIVATE) python $(PY_SNAPSHOT) -c $(SNAPSHOT_CFG)

backtest: build-backtest
	@$(ACTIVATE) python $(PY_BACKTEST) -c $(BACKTEST_CFG)

gridsearch: build-backtest
	@$(ACTIVATE) python $(PY_GRIDSEARCH) -c $(GRIDSEARCH_CFG)

stats:
	@$(ACTIVATE) python $(PY_STATS) -c $(BACKTEST_CFG)

pipeline: convert-latency snapshot build-backtest backtest

# -------- Cleaning ------------------------------------------
clean:
	$(CARGO) clean
clean-out:
	@if [ -d "$(OUT_DIR)" ]; then rm -rf "$(OUT_DIR)"; else echo "[skip] $(OUT_DIR) does not exist"; fi
clean-all: clean clean-out

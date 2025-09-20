# -------- Settings (override with `make VAR=...`) -----------------------------

# Conda env used for the Python steps
CONDA_ENV ?= hftbacktest
CONDA     ?= conda
CONDA_RUN ?= $(CONDA) run -n $(CONDA_ENV)

# Paths to configs (default to ./pipeline/*.yaml)
CFG_DIR           ?= pipeline
CONVERT_CFG       ?= $(CFG_DIR)/convert_config.yaml
LATENCY_CFG       ?= $(CFG_DIR)/latency_config.yaml
CONVERT_LATENCY_CFG       ?= $(CFG_DIR)/convert_latency_config.yaml
BACKTEST_CFG      ?= $(CFG_DIR)/backtest_config.yaml

# Python entry points
PY_CONVERT  ?= pipeline/1_convert.py
PY_LATENCY  ?= pipeline/2_latency.py
PY_CONVERT_LATENCY  ?= pipeline/1_2_convert_latency.py
PY_BACKTEST ?= pipeline/3_backtest.py

# Rust binary name (you exposed it as [[bin]] in Cargo.toml)
BACKTEST_BIN ?= target/release/gridtrading_backtest_args

# Optional output dir for reports (only used by `clean-out`)
OUT_DIR ?= out

# Tools
CARGO ?= cargo
RUSTC ?= rustc
PY    ?= python

# -------- Phonies -------------------------------------------------------------
.PHONY: help lint lint-rs lint-py fmt fmt-rs fmt-py fmt-check check build \
        convert latency backtest pipeline clean clean-out clean-all

# -------- Help ----------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  make fmt           - format Rust (cargo fmt) and Python (black/isort if installed)"
	@echo "  make fmt-check     - check formatting without writing (rustfmt/black/isort --check)"
	@echo "  make lint          - lint Rust (clippy) and Python (ruff/flake8 if installed)"
	@echo "  make check         - cargo check (fast type-check)"
	@echo "  make build         - cargo build --release (builds $(BACKTEST_BIN))"
	@echo "  make convert       - python $(PY_CONVERT) -c $(CONVERT_CFG) in $(CONDA_ENV)"
	@echo "  make latency       - python $(PY_LATENCY) -c $(LATENCY_CFG) in $(CONDA_ENV)"
	@echo "  make backtest      - python $(PY_BACKTEST) -c $(BACKTEST_CFG) in $(CONDA_ENV)"
	@echo "  make pipeline      - convert -> latency -> build -> backtest"
	@echo "  make clean         - cargo clean"
	@echo "  make clean-out     - rm -rf $(OUT_DIR) (if it exists)"
	@echo "  make clean-all     - clean + clean-out"
	@echo ""
	@echo "Overrides (examples):"
	@echo "  make pipeline CONDA_ENV=hftbt CFG_DIR=pipeline"
	@echo "  make convert CONVERT_CFG=pipeline/convert_config.yaml"
	@echo "  make backtest BACKTEST_CFG=pipeline/backtest_config.yaml"

# -------- Formatting ----------------------------------------------------------
fmt: fmt-rs fmt-py

fmt-rs:
	$(CARGO) fmt --all

fmt-py:
	@command -v black >/dev/null 2>&1 && $(CONDA_RUN) black pipeline || echo "[skip] black not found"
	@command -v isort >/dev/null 2>&1 && $(CONDA_RUN) isort pipeline || echo "[skip] isort not found"

fmt-check:
	$(CARGO) fmt --all -- --check
	@command -v black >/dev/null 2>&1 && $(CONDA_RUN) black --check pipeline || echo "[skip] black --check not found"
	@command -v isort >/dev/null 2>&1 && $(CONDA_RUN) isort --check-only pipeline || echo "[skip] isort --check not found"

# -------- Linting -------------------------------------------------------------
lint: lint-rs lint-py

lint-rs:
	$(CARGO) clippy --all-targets --all-features -- -D warnings

lint-py:
	@command -v ruff >/dev/null 2>&1 && $(CONDA_RUN) ruff check pipeline || \
	 (command -v flake8 >/dev/null 2>&1 && $(CONDA_RUN) flake8 pipeline || echo "[skip] ruff/flake8 not found")

# -------- Build / Check -------------------------------------------------------
check:
	$(CARGO) check --all-targets

build:
	$(CARGO) build --release

# -------- Pipeline steps ------------------------------------------------------
convert:
	$(CONDA_RUN) $(PY) $(PY_CONVERT) -c $(CONVERT_CFG)

latency:
	$(CONDA_RUN) $(PY) $(PY_LATENCY) -c $(LATENCY_CFG)

convert-latency:
	$(CONDA_RUN) $(PY) $(PY_CONVERT_LATENCY) -c $(CONVERT_LATENCY_CFG)	

backtest: $(BACKTEST_BIN)
	$(CONDA_RUN) $(PY) $(PY_BACKTEST) -c $(BACKTEST_CFG)

# Ensure the binary exists before backtest (useful if Python expects it)
$(BACKTEST_BIN):
	$(CARGO) build --release

# Full chain
pipeline: convert latency build backtest

# -------- Cleaning ------------------------------------------------------------
clean:
	$(CARGO) clean

clean-out:
	@if [ -d "$(OUT_DIR)" ]; then rm -rf "$(OUT_DIR)"; else echo "[skip] $(OUT_DIR) does not exist"; fi

clean-all: clean clean-out

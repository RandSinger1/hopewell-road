#!/usr/bin/env bash
# Set up the conda environment for the hopewell-road repo on Linux / macOS.
#
# Usage:
#   ./setup_env.sh [env_name [python_version]]
#
# Examples:
#   ./setup_env.sh
#   ./setup_env.sh khthonlidar 3.12

set -euo pipefail

ENV_NAME="${1:-khthonlidar}"
PYTHON_VERSION="${2:-3.12}"

step()  { echo; echo "=== $* ==="; }
info()  { echo "  $*"; }
warn()  { echo "  [warn] $*" >&2; }

# ── 0. Pre-flight checks ──────────────────────────────────────────────────────

step "Checking prerequisites"

if ! command -v conda &>/dev/null; then
    echo "Error: conda not found. Install Anaconda or Miniconda first." >&2
    exit 1
fi
info "conda OK"

if ! command -v git &>/dev/null; then
    echo "Error: git not found." >&2
    exit 1
fi
info "git OK"

if ! command -v cmake &>/dev/null; then
    warn "cmake not found — laszip will fail to build. Install CMake >= 3.15."
fi

# Source conda so 'conda activate' works inside the script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ── 1. Submodules ─────────────────────────────────────────────────────────────

step "Initialising git submodules (AOI2List, laszip-python)"
git submodule update --init
info "Submodules ready"

# ── 2. Conda environment ──────────────────────────────────────────────────────

step "Conda environment: $ENV_NAME (Python $PYTHON_VERSION)"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    info "Environment already exists — skipping creation."
else
    info "Creating environment..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# ── 3. GDAL + PDAL (conda-forge) ─────────────────────────────────────────────

step "Installing GDAL and PDAL via conda-forge"
conda install -n "$ENV_NAME" -c conda-forge gdal python-pdal -y

# ── 4. Core pip requirements ──────────────────────────────────────────────────

step "Installing core pip requirements"
conda run -n "$ENV_NAME" pip install -r requirements.txt

# ── 5. AOI2List ───────────────────────────────────────────────────────────────

step "Installing AOI2List requirements"
conda run -n "$ENV_NAME" pip install -r AOI2List/requirements.txt

# ── 6. laszip (build from source) ────────────────────────────────────────────

step "Building and installing laszip from source"
# Requires CMake >= 3.15 and a C++14 compiler (gcc/clang).
conda run -n "$ENV_NAME" pip install ./laszip-python

# ── 7. Web viewer ─────────────────────────────────────────────────────────────

step "Installing app (web viewer) requirements"
conda run -n "$ENV_NAME" pip install -r app/requirements.txt

# ── Done ──────────────────────────────────────────────────────────────────────

echo
echo "Setup complete."
echo "Activate with:  conda activate $ENV_NAME"
echo "Run the pipeline:  python -m lidar.run_pipeline --help"

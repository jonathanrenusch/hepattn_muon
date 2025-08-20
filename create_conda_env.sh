#!/usr/bin/env bash
set -euo pipefail

# setup_pixi_cache.sh
# Create a conda environment at a custom prefix while using an explicit cache
# location (XDG_CACHE_HOME) and installing pinned wheel URLs from pixi.lock.
#
# Usage:
#   ./setup_pixi_cache.sh \
#       --env-prefix /path/to/env \
#       --cache-dir /eos/path/to/explicit/temp/cache \
#       [--repo-dir /path/to/repo] \
#       [--lock-file /path/to/pixi.lock] \
#       [--env-file /path/to/environment.yml] \
#       [--force]

show_help(){
	sed -n '1,120p' "$0" | sed -n '1,120p'
}

ENV_PREFIX=""
CACHE_DIR=""
REPO_DIR="$(pwd)"
LOCK_FILE="$REPO_DIR/pixi.lock"
ENV_FILE=""
FORCE=0

while [[ ${#} -gt 0 ]]; do
	case "$1" in
		--env-prefix) ENV_PREFIX="$2"; shift 2;;
		--cache-dir) CACHE_DIR="$2"; shift 2;;
		--repo-dir) REPO_DIR="$2"; shift 2;;
		--lock-file) LOCK_FILE="$2"; shift 2;;
		--env-file) ENV_FILE="$2"; shift 2;;
		--force) FORCE=1; shift;;
		-h|--help) show_help; exit 0;;
		*) echo "Unknown arg: $1"; show_help; exit 2;;
	esac
done

if [[ -z "$ENV_PREFIX" ]]; then
	echo "ERROR: --env-prefix is required." >&2
	show_help
	exit 2
fi

# If cache dir not provided, default to a .cache folder inside the repo
if [[ -z "$CACHE_DIR" ]]; then
	CACHE_DIR="$REPO_DIR/../.cache"
	echo "No --cache-dir provided; defaulting to: $CACHE_DIR"
fi

echo "Using repo dir: $REPO_DIR"
echo "Using pixi.lock: $LOCK_FILE"
echo "Env prefix: $ENV_PREFIX"
echo "Cache dir: $CACHE_DIR"

# Set the cache dir for conda/pip and other tools
export XDG_CACHE_HOME="$CACHE_DIR"
mkdir -p "$XDG_CACHE_HOME"
echo "XDG_CACHE_HOME set to $XDG_CACHE_HOME"
# Prefer conda and pip caches under the chosen XDG cache to avoid using $HOME
mkdir -p "$XDG_CACHE_HOME/conda/pkgs" "$XDG_CACHE_HOME/pip"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda/pkgs"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
echo "CONDA_PKGS_DIRS set to $CONDA_PKGS_DIRS"
echo "PIP_CACHE_DIR set to $PIP_CACHE_DIR"

# Use a separate environment-deps.yml (clean, no here-doc) unless user provided an env file
if [[ -z "$ENV_FILE" ]]; then
	ENV_FILE="$REPO_DIR/environment-deps.yml"
	if [[ ! -f "$ENV_FILE" ]]; then
		echo "ERROR: default env file $ENV_FILE not found" >&2
		exit 6
	fi
	echo "Using default dependency file: $ENV_FILE"
else
	echo "Using provided environment file $ENV_FILE"
fi

# Helper: check for mamba/conda
use_mamba=0
if command -v mamba >/dev/null 2>&1; then
	use_mamba=1
	echo "Found mamba -> will prefer mamba for speed"
elif command -v conda >/dev/null 2>&1; then
	echo "mamba not found, will use conda"
else
	echo "ERROR: neither mamba nor conda is available in PATH" >&2
	exit 3
fi

# If env exists and force set, remove it
if [[ -e "$ENV_PREFIX" || -d "$ENV_PREFIX" ]]; then
	if [[ "$FORCE" -eq 1 ]]; then
		echo "Removing existing env at $ENV_PREFIX";
		rm -rf "$ENV_PREFIX"
	else
		echo "Environment path $ENV_PREFIX already exists. Use --force to overwrite." >&2
		exit 4
	fi
fi

echo "Creating conda environment at prefix: $ENV_PREFIX"
if [[ "$use_mamba" -eq 1 ]]; then
	mamba env create -f "$ENV_FILE" -p "$ENV_PREFIX"
else
	conda env create -f "$ENV_FILE" -p "$ENV_PREFIX"
fi

# Activate the created environment in this script
CONDA_BASE="$(conda info --base)"
if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
	# shellcheck disable=SC1090
	source "$CONDA_BASE/etc/profile.d/conda.sh"
	conda activate "$ENV_PREFIX"
else
	echo "ERROR: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh" >&2
	exit 5
fi

echo "Conda environment active: $(which python)" 

# Install pip wheel URLs from pixi.lock (if present)
if [[ -f "$LOCK_FILE" ]]; then
	echo "Parsing $LOCK_FILE for pypi wheel URLs..."
	# extract lines containing 'pypi:' and take the URL or '.'
	mapfile -t urls < <(awk '/pypi:/{gsub(/^[ \t-]*pypi:[ \t]*/,"",$0); print $0}' "$LOCK_FILE" | sed 's/^ *//;s/[",]//g' )

	# Deduplicate while preserving order
	declare -A seen
	to_install=()
	for u in "${urls[@]}"; do
		# trim
		u_trimmed="$(echo "$u" | sed 's/^ *//;s/ *$//')"
		if [[ -z "$u_trimmed" ]]; then
			continue
		fi
		if [[ -z "${seen[$u_trimmed]:-}" ]]; then
			seen[$u_trimmed]=1
			to_install+=("$u_trimmed")
		fi
	done

	echo "Found ${#to_install[@]} pypi entries in lock file"

	for entry in "${to_install[@]}"; do
		if [[ "$entry" == "." ]]; then
			echo "Installing local package in editable mode from $REPO_DIR"
			pip install -e "$REPO_DIR"
			continue
		fi
		# if entry begins with 'https' or 'http' install via pip
		if [[ "$entry" =~ ^https?:// ]]; then
			echo "pip installing: $entry"
			pip install --progress-bar=on "$entry" || {
				echo "WARNING: pip install failed for $entry; continuing" >&2
			}
		else
			# otherwise try to pip install the token (package name)
			echo "pip installing package token: $entry"
			pip install "$entry" || {
				echo "WARNING: pip install failed for $entry; continuing" >&2
			}
		fi
	done
else
	echo "No lock file found at $LOCK_FILE, skipping locked pip installs"
fi

echo "Ensure editable installation of project (pip install -e)"
pip install -e "$REPO_DIR" || echo "pip editable install failed" >&2

echo "Done. Quick checks:"
python - <<'PY'
import sys
print('python:', sys.version.split()[0])
try:
		import torch
		print('torch:', torch.__version__, 'cuda_available=', torch.cuda.is_available())
except Exception as e:
		print('torch import failed:', e)
try:
		import flash_attn
		print('flash_attn import OK')
except Exception as e:
		print('flash_attn import failed or not installed:', e)
try:
		import hepattn
		print('hepattn import OK')
except Exception as e:
		print('hepattn import failed:', e)
PY

echo "Environment creation script finished. If anything failed, inspect the console output above." 

#!/usr/bin/env bash
# Prepare an isolated project runtime on an already provisioned Linux host.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="$repo_root/.venv"
python_bin=python3.12
profile=gpu
dry_run=false

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_cloud_env.sh [options]
  --venv PATH       Fresh virtual environment (default: repository/.venv)
  --python COMMAND Python 3.12 interpreter (default: python3.12)
  --cpu            Prepare a CPU development environment instead of GPU training
  --dry-run        Print the installation plan without executing commands
  --help           Show this help

The host must already exist. GPU setup requires a working NVIDIA driver.
No cloud resources, host drivers or global Git settings are changed.
Inherited proxy, certificate and package-index settings are preserved.
EOF
}

while (($#)); do
  case "$1" in
    --venv|--python)
      if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
        echo "Missing value for $1" >&2
        exit 2
      fi
      if [[ "$1" == --venv ]]; then venv_path="$2"; else python_bin="$2"; fi
      shift 2
      ;;
    --cpu) profile=cpu; shift ;;
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Match Python venv's lexical abspath normalization before checking freshness.
venv_path="$(realpath -m -s -- "$venv_path")"
if [[ -e "$venv_path" || -L "$venv_path" ]]; then
  echo "Environment already exists: $venv_path; choose a fresh --venv." >&2
  exit 1
fi
requirements="$repo_root/cloud/requirements-$profile.txt"
runtime_python="$venv_path/bin/python"

if $dry_run; then
  printf 'Profile: %s\n' "$profile"
  printf '%q ' "$python_bin" -m venv "$venv_path"; printf '\n'
  printf '%q ' "$runtime_python" -m pip install -r "$requirements"; printf '\n'
  printf '%q ' "$runtime_python" -m pip install --no-deps -e "$repo_root"
  printf '\n'
  printf '%q ' "$runtime_python" -m pip check; printf '\n'
  echo "Then save package versions and verify GCS support and $profile execution."
  exit 0
fi

command -v "$python_bin" >/dev/null || {
  echo "Python interpreter not found: $python_bin" >&2; exit 1;
}
"$python_bin" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 required")'
if [[ "$profile" == gpu ]]; then
  command -v nvidia-smi >/dev/null || {
    echo "GPU setup requires a host with nvidia-smi; use --cpu for development." >&2
    exit 1
  }
  nvidia-smi -L >/dev/null
fi

"$python_bin" -m venv "$venv_path"
"$runtime_python" -m pip install -r "$requirements"
"$runtime_python" -m pip install --no-deps -e "$repo_root"
"$runtime_python" -m pip check
"$runtime_python" -m pip freeze > "$venv_path/cloud-packages.txt"
if [[ "$profile" == gpu ]]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
    > "$venv_path/cloud-gpu.csv"
fi
if command -v git >/dev/null && git -C "$repo_root" rev-parse --is-inside-work-tree \
    >/dev/null 2>&1; then
  git -C "$repo_root" rev-parse HEAD > "$venv_path/cloud-source-commit.txt"
  git -C "$repo_root" status --short > "$venv_path/cloud-source-status.txt"
fi

runtime_flags=()
if [[ "$profile" == cpu ]]; then runtime_flags=(--cpu); fi
MPLCONFIGDIR="${MPLCONFIGDIR:-$venv_path/cache/matplotlib}" \
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$venv_path/cache}" \
  "$runtime_python" "$repo_root/cloud/check_runtime.py" "${runtime_flags[@]}" \
    --output "$venv_path/cloud-readiness.json"

echo "Environment ready: $venv_path"
echo "Run the source preflight and calibration in docs/gpu-validation.md next."

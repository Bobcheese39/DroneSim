#!/usr/bin/env bash
# Launch the DroneSim Panel app.
#
# Usage:
#   ./run_app.sh              # normal mode
#   ./run_app.sh --debug      # verbose Cesium + sim logging in the GUI console
#   ./run_app.sh --debug --port 5007
#
# Extra arguments are forwarded to ``panel serve`` (e.g. --port, --address).

set -euo pipefail

DEBUG=0
PANEL_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --debug)
      DEBUG=1
      ;;
    *)
      PANEL_ARGS+=("$arg")
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "$DEBUG" -eq 1 ]]; then
  export DRONESIM_DEBUG=1
  export PYTHONUNBUFFERED=1
  echo "DroneSim debug mode: DRONESIM_DEBUG=1 (Cesium + sim logs -> GUI console)"
fi

exec panel serve app.py --show --autoreload "${PANEL_ARGS[@]}"

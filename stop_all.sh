#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HOSTFILE="${HOSTFILE:-$ROOT/hostfile}"
export SSH_PORT="${SSH_PORT:-62218}"
exec python3 "$ROOT/recipes/common/musa_stop.py"

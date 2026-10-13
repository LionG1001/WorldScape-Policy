#!/usr/bin/env bash
# One controller, all nodes. See docs/musa.md for hostfile and environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$ROOT/recipes/common/musa_multinode_launch.py" "$@"

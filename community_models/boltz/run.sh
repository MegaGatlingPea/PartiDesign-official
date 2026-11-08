#!/usr/bin/env bash
set -euo pipefail

boltz_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${boltz_dir}/source/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PARTIDESIGN_BOLTZ_PYTHON:-python}" -B -m boltz.main "$@"

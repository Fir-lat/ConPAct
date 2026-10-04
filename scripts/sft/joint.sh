set -euo pipefail

CONPACT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ROLE=joint
exec bash "${CONPACT_SCRIPT_DIR}/run_sft.sh"

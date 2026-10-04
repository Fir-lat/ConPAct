set -euo pipefail

: "${PLANNER_MODEL:?Set PLANNER_MODEL to a local planner model directory}"
: "${ACTOR_MODEL:?Set ACTOR_MODEL to a local actor model directory}"
: "${DATASET:?Set DATASET to an exported ConPAct JSONL file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new experiment directory}"

CONPACT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${PLANNER_MODEL}" \
ROLE=planner \
OUTPUT_DIR="${OUTPUT_DIR}/planner" \
DS_STAGE="${PLANNER_DS_STAGE:-zero3_offload}" \
TRAIN_BSZ="${PLANNER_TRAIN_BSZ:-1}" \
GRAD_ACCUM="${PLANNER_GRAD_ACCUM:-}" \
NUM_EPOCHS="${PLANNER_EPOCHS:-5}" \
bash "${CONPACT_SCRIPT_DIR}/run_sft.sh"

MODEL_PATH="${ACTOR_MODEL}" \
ROLE=actor \
OUTPUT_DIR="${OUTPUT_DIR}/actor" \
DS_STAGE="${ACTOR_DS_STAGE:-zero1}" \
TRAIN_BSZ="${ACTOR_TRAIN_BSZ:-4}" \
GRAD_ACCUM="${ACTOR_GRAD_ACCUM:-}" \
NUM_EPOCHS="${ACTOR_EPOCHS:-5}" \
bash "${CONPACT_SCRIPT_DIR}/run_sft.sh"

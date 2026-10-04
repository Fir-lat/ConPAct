set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local model directory}"
: "${DATASET:?Set DATASET to an exported ConPAct JSONL file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new experiment directory}"

CONPACT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONPACT_PYTHON="${CONPACT_PYTHON:-python3}"
ROLE="${ROLE:-joint}"
DRY_RUN="${DRY_RUN:-0}"

case "${ROLE}" in
    joint|planner|actor) ;;
    *) printf '%s\n' 'ROLE must be joint, planner, or actor' >&2; exit 1 ;;
esac

[[ -d "${MODEL_PATH}" ]] || { printf '%s\n' 'MODEL_PATH must be an existing model directory' >&2; exit 1; }
[[ -s "${DATASET}" ]] || { printf '%s\n' 'DATASET must be a nonempty JSONL file' >&2; exit 1; }

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
[[ "${NNODES}" == "1" ]] || { printf '%s\n' 'These example scripts launch one GPU node' >&2; exit 1; }

TRAIN_BSZ="${TRAIN_BSZ:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
for value in "${TRAIN_BSZ}" "${GLOBAL_BATCH_SIZE}" "${NPROC_PER_NODE}" "${NNODES}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' 'Batch sizes and process counts must be positive integers' >&2; exit 1; }
done
CONPACT_BATCH_FACTOR=$((TRAIN_BSZ * NPROC_PER_NODE * NNODES))
if [[ -z "${GRAD_ACCUM:-}" ]]; then
    ((GLOBAL_BATCH_SIZE % CONPACT_BATCH_FACTOR == 0)) || { printf '%s\n' 'Global batch size is not divisible by microbatch size and world size' >&2; exit 1; }
    GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / CONPACT_BATCH_FACTOR))
fi
[[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' 'GRAD_ACCUM must be a positive integer' >&2; exit 1; }
((GRAD_ACCUM * CONPACT_BATCH_FACTOR == GLOBAL_BATCH_SIZE)) || { printf '%s\n' 'Gradient accumulation does not match GLOBAL_BATCH_SIZE' >&2; exit 1; }

if [[ "${DRY_RUN}" != "1" ]]; then
    "${CONPACT_PYTHON}" -c 'from importlib.metadata import version; value = version("ms-swift"); assert value.split(".")[0] == "4", "These scripts require ms-swift 4.x"'
fi

CONPACT_TRAIN_FILE="${OUTPUT_DIR}/data/${ROLE}.jsonl"
"${CONPACT_PYTHON}" "${CONPACT_SCRIPT_DIR}/prepare_data.py" \
    --input "${DATASET}" \
    --output "${CONPACT_TRAIN_FILE}" \
    --role "${ROLE}" \
    --max-frames "${HISTORY_FRAMES:-3}"

SWIFT_CMD=(
    swift sft
    --model "${MODEL_PATH}"
    --dataset "${CONPACT_TRAIN_FILE}"
    --tuner_type full
    --load_from_cache_file true
    --split_dataset_ratio 0
    --torch_dtype bfloat16
    --num_train_epochs "${NUM_EPOCHS:-3}"
    --per_device_train_batch_size "${TRAIN_BSZ}"
    --attn_impl "${ATTN_IMPL:-sdpa}"
    --freeze_vit true
    --freeze_aligner true
    --freeze_llm false
    --learning_rate "${LR:-1e-5}"
    --gradient_checkpointing true
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --save_strategy epoch
    --save_total_limit 1
    --save_only_model true
    --logging_steps 5
    --max_length "${MAX_LENGTH:-16384}"
    --output_dir "${OUTPUT_DIR}"
    --logging_dir "${OUTPUT_DIR}/logs"
    --add_version false
    --warmup_ratio 0.05
    --deepspeed "${DS_STAGE:-zero1}"
    --dataset_num_proc "${DATASET_NUM_PROC:-8}"
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}"
    --loss_scale last_round
    --strict true
    --lazy_tokenize false
    --truncation_strategy delete
    --seed "${SEED:-42}"
    --data_seed "${SEED:-42}"
    --lr_scheduler_type cosine
    --weight_decay 0.1
    --max_grad_norm 1.0
    --report_to none
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${SWIFT_CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${SWIFT_CMD[@]}"

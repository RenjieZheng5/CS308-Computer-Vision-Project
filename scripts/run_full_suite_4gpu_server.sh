#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/server_2026_06}"
LOG_ROOT="${LOG_ROOT:-logs/server_2026_06}"
COCO_MAX_IMAGES="${COCO_MAX_IMAGES:-0}"
COCO_SAMPLING="${COCO_SAMPLING:-random}"
COCO_SEED="${COCO_SEED:-308}"
COCO_TOP_K="${COCO_TOP_K:-100}"
OWL_SCORE_THRESHOLD="${OWL_SCORE_THRESHOLD:-0.01}"
OWL_NMS_THRESHOLD="${OWL_NMS_THRESHOLD:--1.0}"
GROUNDING_BOX_THRESHOLD="${GROUNDING_BOX_THRESHOLD:-0.20}"
GROUNDING_TEXT_THRESHOLD="${GROUNDING_TEXT_THRESHOLD:-0.20}"
YOLO_CONFIDENCE="${YOLO_CONFIDENCE:-0.001}"
YOLO_IOU_THRESHOLD="${YOLO_IOU_THRESHOLD:-0.7}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
REFCOCO_SPLIT="${REFCOCO_SPLIT:-val}"
REFCOCO_MAX_ROWS="${REFCOCO_MAX_ROWS:-0}"
REFCOCO_EXPRESSION_MODE="${REFCOCO_EXPRESSION_MODE:-all}"
SUBSET_WORKERS="${SUBSET_WORKERS:-16}"
OWL_THRESHOLDS="${OWL_THRESHOLDS:-0.01,0.03,0.05,0.10,0.20}"
GROUNDING_THRESHOLDS="${GROUNDING_THRESHOLDS:-0.20,0.25,0.30,0.35,0.40}"
YOLO_THRESHOLDS="${YOLO_THRESHOLDS:-0.001,0.01,0.05,0.10,0.25}"

export OUTPUT_ROOT
export COCO_MAX_IMAGES
export COCO_SAMPLING
export COCO_SEED
export REFCOCO_SPLIT
export REFCOCO_MAX_ROWS
export REFCOCO_EXPRESSION_MODE

COCO_DATA_DIR="${DATA_ROOT}/coco"
REFCOCO_DATA_DIR="${DATA_ROOT}/refcoco"
if [[ "${COCO_MAX_IMAGES}" -le 0 ]]; then
  COCO_TAG="full_val2017"
else
  COCO_TAG="${COCO_MAX_IMAGES}_${COCO_SAMPLING}_seed${COCO_SEED}"
fi
if [[ "${REFCOCO_MAX_ROWS}" -le 0 ]]; then
  REF_ROWS_TAG="full"
else
  REF_ROWS_TAG="rows${REFCOCO_MAX_ROWS}"
fi
REFCOCO_TAG="${REFCOCO_SPLIT}_${REF_ROWS_TAG}_${REFCOCO_EXPRESSION_MODE}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${OUTPUT_ROOT}/threshold_sensitivity"

run_gpu_job() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log_path="${LOG_ROOT}/${name}.log"
  echo "Starting ${name} on GPU ${gpu}; log: ${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" "$@" >"${log_path}" 2>&1 &
  echo "$! ${name}" >>"${LOG_ROOT}/pids.txt"
}

wait_for_jobs() {
  local status=0
  while read -r pid name; do
    if [[ -z "${pid:-}" ]]; then
      continue
    fi
    if wait "${pid}"; then
      echo "Finished ${name}"
    else
      echo "FAILED ${name}; see ${LOG_ROOT}/${name}.log" >&2
      status=1
    fi
  done <"${LOG_ROOT}/pids.txt"
  : >"${LOG_ROOT}/pids.txt"
  return "${status}"
}

: >"${LOG_ROOT}/pids.txt"

echo "Preparing COCO val2017 images..."
python scripts/prepare_coco_subset.py \
  --data-dir "${COCO_DATA_DIR}" \
  --max-images "${COCO_MAX_IMAGES}" \
  --sampling "${COCO_SAMPLING}" \
  --seed "${COCO_SEED}" \
  --workers "${SUBSET_WORKERS}"

echo "Batch 1: full COCO evaluations plus RefCOCO Grounding DINO."
run_gpu_job 0 "coco_grounding_dino_${COCO_TAG}" \
  python scripts/evaluate_coco_grounding_dino.py \
    --data-dir "${COCO_DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/coco_grounding_dino_eval_${COCO_TAG}" \
    --max-images "${COCO_MAX_IMAGES}" \
    --sampling "${COCO_SAMPLING}" \
    --seed "${COCO_SEED}" \
    --box-threshold "${GROUNDING_BOX_THRESHOLD}" \
    --text-threshold "${GROUNDING_TEXT_THRESHOLD}" \
    --top-k "${COCO_TOP_K}"

run_gpu_job 1 "coco_owlvit_${COCO_TAG}" \
  python scripts/evaluate_coco_owlvit.py \
    --data-dir "${COCO_DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/coco_owlvit_eval_${COCO_TAG}" \
    --max-images "${COCO_MAX_IMAGES}" \
    --sampling "${COCO_SAMPLING}" \
    --seed "${COCO_SEED}" \
    --score-threshold "${OWL_SCORE_THRESHOLD}" \
    --nms-threshold "${OWL_NMS_THRESHOLD}" \
    --top-k "${COCO_TOP_K}"

run_gpu_job 2 "coco_yolo_world_${COCO_TAG}" \
  python scripts/evaluate_coco_yolo_world.py \
    --data-dir "${COCO_DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/coco_yolo_world_eval_${COCO_TAG}" \
    --max-images "${COCO_MAX_IMAGES}" \
    --sampling "${COCO_SAMPLING}" \
    --seed "${COCO_SEED}" \
    --confidence "${YOLO_CONFIDENCE}" \
    --iou-threshold "${YOLO_IOU_THRESHOLD}" \
    --top-k "${COCO_TOP_K}" \
    --image-size "${IMAGE_SIZE}"

run_gpu_job 3 "refcoco_grounding_dino_${REFCOCO_TAG}" \
  python scripts/evaluate_refcoco.py \
    --model-type grounding-dino \
    --data-dir "${REFCOCO_DATA_DIR}" \
    --split "${REFCOCO_SPLIT}" \
    --max-rows "${REFCOCO_MAX_ROWS}" \
    --expression-mode "${REFCOCO_EXPRESSION_MODE}" \
    --refresh-manifest \
    --output-dir "${OUTPUT_ROOT}/refcoco_grounding_dino_eval_${REFCOCO_TAG}"

wait_for_jobs

echo "Batch 2: remaining RefCOCO evaluations and OWL-ViT NMS ablation."
run_gpu_job 0 "refcoco_owlvit_${REFCOCO_TAG}" \
  python scripts/evaluate_refcoco.py \
    --model-type owlvit \
    --data-dir "${REFCOCO_DATA_DIR}" \
    --split "${REFCOCO_SPLIT}" \
    --max-rows "${REFCOCO_MAX_ROWS}" \
    --expression-mode "${REFCOCO_EXPRESSION_MODE}" \
    --output-dir "${OUTPUT_ROOT}/refcoco_owlvit_eval_${REFCOCO_TAG}"

run_gpu_job 1 "refcoco_yolo_world_${REFCOCO_TAG}" \
  python scripts/evaluate_refcoco.py \
    --model-type yolo-world \
    --data-dir "${REFCOCO_DATA_DIR}" \
    --split "${REFCOCO_SPLIT}" \
    --max-rows "${REFCOCO_MAX_ROWS}" \
    --expression-mode "${REFCOCO_EXPRESSION_MODE}" \
    --image-size "${IMAGE_SIZE}" \
    --output-dir "${OUTPUT_ROOT}/refcoco_yolo_world_eval_${REFCOCO_TAG}"

run_gpu_job 2 "coco_owlvit_100_nms" \
  python scripts/evaluate_coco_owlvit.py \
    --data-dir "${COCO_DATA_DIR}" \
    --output-dir "${OUTPUT_ROOT}/coco_owlvit_eval_100_nms" \
    --max-images 100 \
    --sampling "${COCO_SAMPLING}" \
    --seed "${COCO_SEED}" \
    --score-threshold "${OWL_SCORE_THRESHOLD}" \
    --nms-threshold 0.5 \
    --top-k "${COCO_TOP_K}"

wait_for_jobs

echo "Running threshold sensitivity on full COCO predictions..."
python scripts/evaluate_threshold_sensitivity.py \
  --annotation-file "${COCO_DATA_DIR}/annotations/instances_val2017.json" \
  --predictions "${OUTPUT_ROOT}/coco_owlvit_eval_${COCO_TAG}/coco_predictions.json" \
  --metrics "${OUTPUT_ROOT}/coco_owlvit_eval_${COCO_TAG}/metrics.json" \
  --thresholds "${OWL_THRESHOLDS}" \
  --output "${OUTPUT_ROOT}/threshold_sensitivity/owlvit.json"

python scripts/evaluate_threshold_sensitivity.py \
  --annotation-file "${COCO_DATA_DIR}/annotations/instances_val2017.json" \
  --predictions "${OUTPUT_ROOT}/coco_grounding_dino_eval_${COCO_TAG}/coco_predictions.json" \
  --metrics "${OUTPUT_ROOT}/coco_grounding_dino_eval_${COCO_TAG}/metrics.json" \
  --thresholds "${GROUNDING_THRESHOLDS}" \
  --output "${OUTPUT_ROOT}/threshold_sensitivity/grounding_dino.json"

python scripts/evaluate_threshold_sensitivity.py \
  --annotation-file "${COCO_DATA_DIR}/annotations/instances_val2017.json" \
  --predictions "${OUTPUT_ROOT}/coco_yolo_world_eval_${COCO_TAG}/coco_predictions.json" \
  --metrics "${OUTPUT_ROOT}/coco_yolo_world_eval_${COCO_TAG}/metrics.json" \
  --thresholds "${YOLO_THRESHOLDS}" \
  --output "${OUTPUT_ROOT}/threshold_sensitivity/yolo_world.json"

echo "Regenerating figures..."
python scripts/generate_report_figures.py

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  echo "Compiling report..."
  (
    cd report
    pdflatex example_paper.tex
    bibtex example_paper
    pdflatex example_paper.tex
    pdflatex example_paper.tex
  )
else
  echo "Skipping report compile because pdflatex or bibtex is not available."
fi

echo "Done. Metrics are under ${OUTPUT_ROOT}; logs are under ${LOG_ROOT}."

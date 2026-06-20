#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/server_2026_06}"
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
REFCOCO_REFRESH_MANIFEST="${REFCOCO_REFRESH_MANIFEST:-0}"
SUBSET_WORKERS="${SUBSET_WORKERS:-8}"
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
export REFCOCO_REFRESH_MANIFEST

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

echo "Preparing COCO images..."
python scripts/prepare_coco_subset.py \
  --data-dir "${COCO_DATA_DIR}" \
  --max-images "${COCO_MAX_IMAGES}" \
  --sampling "${COCO_SAMPLING}" \
  --seed "${COCO_SEED}" \
  --workers "${SUBSET_WORKERS}"

echo "Running COCO evaluations..."
python scripts/evaluate_coco_owlvit.py \
  --data-dir "${COCO_DATA_DIR}" \
  --output-dir "${OUTPUT_ROOT}/coco_owlvit_eval_${COCO_TAG}" \
  --max-images "${COCO_MAX_IMAGES}" \
  --sampling "${COCO_SAMPLING}" \
  --seed "${COCO_SEED}" \
  --score-threshold "${OWL_SCORE_THRESHOLD}" \
  --nms-threshold "${OWL_NMS_THRESHOLD}" \
  --top-k "${COCO_TOP_K}"

python scripts/evaluate_coco_grounding_dino.py \
  --data-dir "${COCO_DATA_DIR}" \
  --output-dir "${OUTPUT_ROOT}/coco_grounding_dino_eval_${COCO_TAG}" \
  --max-images "${COCO_MAX_IMAGES}" \
  --sampling "${COCO_SAMPLING}" \
  --seed "${COCO_SEED}" \
  --box-threshold "${GROUNDING_BOX_THRESHOLD}" \
  --text-threshold "${GROUNDING_TEXT_THRESHOLD}" \
  --top-k "${COCO_TOP_K}"

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

echo "Running full RefCOCO evaluations..."
REFCOCO_REFRESH_ARG=""
if [[ "${REFCOCO_REFRESH_MANIFEST}" -ne 0 ]]; then
  REFCOCO_REFRESH_ARG="--refresh-manifest"
fi
python scripts/evaluate_refcoco.py \
  --model-type owlvit \
  --data-dir "${REFCOCO_DATA_DIR}" \
  --split "${REFCOCO_SPLIT}" \
  --max-rows "${REFCOCO_MAX_ROWS}" \
  --expression-mode "${REFCOCO_EXPRESSION_MODE}" \
  ${REFCOCO_REFRESH_ARG} \
  --output-dir "${OUTPUT_ROOT}/refcoco_owlvit_eval_${REFCOCO_TAG}"

python scripts/evaluate_refcoco.py \
  --model-type grounding-dino \
  --data-dir "${REFCOCO_DATA_DIR}" \
  --split "${REFCOCO_SPLIT}" \
  --max-rows "${REFCOCO_MAX_ROWS}" \
  --expression-mode "${REFCOCO_EXPRESSION_MODE}" \
  --output-dir "${OUTPUT_ROOT}/refcoco_grounding_dino_eval_${REFCOCO_TAG}"

python scripts/evaluate_refcoco.py \
  --model-type yolo-world \
  --data-dir "${REFCOCO_DATA_DIR}" \
  --split "${REFCOCO_SPLIT}" \
  --max-rows "${REFCOCO_MAX_ROWS}" \
  --expression-mode "${REFCOCO_EXPRESSION_MODE}" \
  --image-size "${IMAGE_SIZE}" \
  --output-dir "${OUTPUT_ROOT}/refcoco_yolo_world_eval_${REFCOCO_TAG}"

echo "Running threshold sensitivity..."
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

echo "Running OWL-ViT NMS ablation..."
python scripts/evaluate_coco_owlvit.py \
  --data-dir "${COCO_DATA_DIR}" \
  --output-dir "${OUTPUT_ROOT}/coco_owlvit_eval_100_nms" \
  --max-images 100 \
  --sampling "${COCO_SAMPLING}" \
  --seed "${COCO_SEED}" \
  --score-threshold "${OWL_SCORE_THRESHOLD}" \
  --nms-threshold 0.5 \
  --top-k "${COCO_TOP_K}"

echo "Regenerating figures..."
python scripts/generate_report_figures.py

echo "Compiling report..."
(
  cd report
  pdflatex example_paper.tex
  bibtex example_paper
  pdflatex example_paper.tex
  pdflatex example_paper.tex
)

echo "Done."

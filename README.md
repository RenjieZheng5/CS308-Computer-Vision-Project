# Open-Vocabulary Object Detection and Visual Grounding

CS308 Computer Vision final project comparing three text-conditioned detectors:

- OWL-ViT Base (`google/owlvit-base-patch32`)
- Grounding DINO Tiny (`IDEA-Research/grounding-dino-tiny`)
- YOLO-World v2 Small (`yolov8s-worldv2.pt`)

The project contains arbitrary-prompt demos, COCO detection evaluation, RefCOCO
referring-expression evaluation, threshold and NMS ablations, runtime/VRAM
benchmarking, visualizations, and an ICML-style LaTeX report.

## Environment

The initial local experiments used Windows, an NVIDIA RTX 4060 Laptop GPU
(8 GB), CUDA-enabled PyTorch 2.11.0, and Python 3.10.

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

For the 4 x TITAN RTX server with NVIDIA driver 550.120 and `nvidia-smi`
reporting CUDA 12.4, install the CUDA 12.4 PyTorch profile instead:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cu124.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

The server should use `requirements-cu124.txt`, not the default `cu128`
requirements. PyTorch wheels bundle the CUDA runtime, so the local CUDA Toolkit
is not required for these evaluation scripts as long as the NVIDIA driver is
compatible.

Model weights (`*.pt`), Hugging Face caches, and downloaded datasets are ignored.
Selected result JSON files and qualitative figures under `outputs/` are kept as
project artifacts; therefore `outputs/` is not globally ignored.

## Server rerun variables

Use one shared configuration block when rerunning the full suite on a server:

```bash
export DATA_ROOT=data
export OUTPUT_ROOT=outputs/server_2026_06
export COCO_MAX_IMAGES=500
export COCO_SAMPLING=random
export COCO_SEED=308
export COCO_TOP_K=100
export REFCOCO_SPLIT=val
export REFCOCO_MAX_ROWS=0
export REFCOCO_EXPRESSION_MODE=all
export IMAGE_SIZE=640
```

`REFCOCO_MAX_ROWS=0` means the full split. Set it to a small number only for
smoke tests.

You can run the whole suite with:

```bash
bash scripts/run_full_suite_server.sh
```

On the 4-GPU server, the three RefCOCO evaluations can also be run in parallel
after the COCO runs finish:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_refcoco.py --model-type grounding-dino --split val --expression-mode all --max-rows 0 --data-dir data/refcoco --refresh-manifest --output-dir outputs/server_2026_06/refcoco_grounding_dino_eval_val_full_all
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_refcoco.py --model-type owlvit --split val --expression-mode all --max-rows 0 --data-dir data/refcoco --output-dir outputs/server_2026_06/refcoco_owlvit_eval_val_full_all
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_refcoco.py --model-type yolo-world --split val --expression-mode all --max-rows 0 --data-dir data/refcoco --image-size 640 --output-dir outputs/server_2026_06/refcoco_yolo_world_eval_val_full_all
```

## Single-image demos

```powershell
python scripts\owlvit_demo.py `
  --image-url "http://images.cocodataset.org/val2017/000000039769.jpg" `
  --queries "cat, remote control, couch" `
  --output-dir outputs\owlvit_demo

python scripts\grounding_dino_demo.py `
  --image-url "http://images.cocodataset.org/val2017/000000039769.jpg" `
  --queries "cat, remote control, couch" `
  --output-dir outputs\grounding_dino_demo
```

Each demo writes `predictions.json` and `visualization.jpg`.

## COCO 500-image evaluation

All models use the same fixed random subset of COCO val2017:

- 500 images
- random seed 308
- all 80 category names as text prompts
- at most 100 predictions per image
- standard COCO bbox AP/AR from `pycocotools`
- one unmeasured warm-up followed by CUDA-synchronized timing

Prepare the shared subset:

```powershell
python scripts\prepare_coco_subset.py `
  --data-dir data\coco --max-images 500 --sampling random --seed 308 --workers 8
```

Run the models:

```powershell
python scripts\evaluate_coco_owlvit.py `
  --data-dir data\coco --output-dir outputs\coco_owlvit_eval_500_random `
  --max-images 500 --sampling random --seed 308 `
  --score-threshold 0.01 --nms-threshold -1 --top-k 100

python scripts\evaluate_coco_grounding_dino.py `
  --data-dir data\coco --output-dir outputs\coco_grounding_dino_eval_500_random `
  --max-images 500 --sampling random --seed 308 `
  --box-threshold 0.20 --text-threshold 0.20 --top-k 100

python scripts\evaluate_coco_yolo_world.py `
  --data-dir data\coco --output-dir outputs\coco_yolo_world_eval_500_random `
  --max-images 500 --sampling random --seed 308 `
  --confidence 0.001 --iou-threshold 0.7 --top-k 100 --image-size 640
```

Results on the RTX 4060 Laptop GPU:

| Model | AP | AP50 | AP75 | AP small | AP medium | AP large | AR100 | Pipeline FPS | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OWL-ViT Base | 0.270 | 0.428 | 0.287 | 0.134 | 0.303 | 0.454 | 0.487 | 11.6 | 351 MB |
| Grounding DINO Tiny | **0.452** | **0.589** | **0.498** | **0.325** | **0.497** | 0.549 | 0.548 | 1.5 | 2356 MB |
| YOLO-World v2 Small | 0.394 | 0.549 | 0.427 | 0.234 | 0.452 | **0.556** | **0.561** | **62.0** | 716 MB |

Threshold values are model-specific because the three APIs produce differently
calibrated scores. They should not be interpreted as equivalent confidence
levels.

## RefCOCO grounding evaluation

The grounding experiment now defaults to the full `lmms-lab/RefCOCO` `val`
split and evaluates all referring expressions attached to each region row. The
highest-scoring predicted box is correct when IoU with the target box is at
least 0.5.

```powershell
python scripts\evaluate_refcoco.py --model-type owlvit `
  --split val --expression-mode all `
  --output-dir outputs\refcoco_owlvit_eval_val_full_all
python scripts\evaluate_refcoco.py --model-type grounding-dino `
  --split val --expression-mode all `
  --output-dir outputs\refcoco_grounding_dino_eval_val_full_all
python scripts\evaluate_refcoco.py --model-type yolo-world `
  --split val --expression-mode all `
  --output-dir outputs\refcoco_yolo_world_eval_val_full_all
```

Use `--max-rows 100` for a quick debug run if you do not want to wait for the
full split.

The refreshed full-split results will be written to
`outputs/refcoco_*_eval_val_full_all/metrics.json` and should replace the
earlier 100-row diagnostic numbers in the report once the server rerun
finishes.

## Ablations and report

Re-evaluate saved detections under different score thresholds:

```powershell
python scripts\evaluate_threshold_sensitivity.py `
  --annotation-file data\coco\annotations\instances_val2017.json `
  --predictions outputs\coco_owlvit_eval_500_random\coco_predictions.json `
  --metrics outputs\coco_owlvit_eval_500_random\metrics.json `
  --thresholds 0.01,0.03,0.05,0.10,0.20 `
  --output outputs\threshold_sensitivity\owlvit.json
```

The earlier 100-image OWL-ViT NMS ablation is retained under
`outputs/coco_owlvit_eval_100_nms`: class-wise NMS at IoU 0.5 reduced AP from
0.336 to 0.323 and AR100 from 0.520 to 0.475.

Regenerate report charts and compile the paper:

```powershell
python scripts\generate_report_figures.py
cd report
pdflatex example_paper.tex
bibtex example_paper
pdflatex example_paper.tex
pdflatex example_paper.tex
```

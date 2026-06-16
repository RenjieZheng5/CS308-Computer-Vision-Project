# CS308 Computer Vision Final Project - Task 4 Baseline

Topic: Open-Vocabulary Object Detection and Visual Grounding.

This repo contains a reproducible OWL-ViT baseline for open-vocabulary object
detection. The pipeline accepts arbitrary text prompts, predicts bounding boxes,
exports JSON predictions, draws visualizations, and evaluates on a COCO val2017
subset with standard COCO bbox metrics.

## Environment

Current local environment:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: `2.11.0+cu128`
- Torchvision: `0.26.0+cu128`
- CUDA available in PyTorch: `True`
- Transformers: `5.5.3`
- Model: `google/owlvit-base-patch32`

Verify GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

CUDA PyTorch install command used locally:

```powershell
python -m pip install --force-reinstall --no-cache-dir torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

## Single Image Demo

```powershell
python scripts\owlvit_demo.py `
  --image-url "http://images.cocodataset.org/val2017/000000039769.jpg" `
  --queries "cat, remote control, couch" `
  --output-dir outputs\owlvit_demo_gpu
```

The script writes:

- `predictions.json`
- `visualization.jpg`

For local COCO images, run the COCO subset evaluation once first. That command
downloads the annotations and selected images into `data/coco`.

For cleaner display after the COCO subset has been downloaded, use score
thresholding and class-wise NMS:

```powershell
python scripts\owlvit_demo.py `
  --image-path data\coco\val2017\000000001503.jpg `
  --queries "laptop, keyboard, mouse, tv" `
  --score-threshold 0.12 `
  --nms-threshold 0.5 `
  --output-dir outputs\qualitative_clean\workspace
```

## COCO Subset Evaluation

`data/` is intentionally not committed. The evaluation script automatically
downloads the COCO val2017 annotation archive, extracts
`instances_val2017.json`, and downloads only the images required for the selected
subset.

```powershell
python scripts\evaluate_coco_owlvit.py `
  --data-dir data\coco `
  --output-dir outputs\coco_owlvit_eval_100 `
  --max-images 100 `
  --top-k 100
```

Current 100-image result without NMS:

| Metric | Value |
| --- | ---: |
| AP@[IoU=.50:.95] | 0.336 |
| AP@0.50 | 0.510 |
| AP@0.75 | 0.353 |
| AP small | 0.178 |
| AP medium | 0.368 |
| AP large | 0.563 |
| AR@100 | 0.520 |

Full metrics:

- `outputs/coco_owlvit_eval_100/metrics.json`
- `outputs/coco_owlvit_eval_100/coco_predictions.json`

These output files are also intentionally not committed. Re-run the command
above to regenerate them.

NMS ablation:

```powershell
python scripts\evaluate_coco_owlvit.py `
  --data-dir data\coco `
  --output-dir outputs\coco_owlvit_eval_100_nms `
  --max-images 100 `
  --top-k 100 `
  --nms-threshold 0.5
```

NMS made visualizations cleaner, but on the 100-image subset AP decreased from
`0.336` to `0.323`, and AR@100 decreased from `0.520` to `0.475`.

## Generated Qualitative Results

The existing local run generated these clean visualizations for report or
slides:

- `outputs/qualitative_clean/workspace/visualization.jpg`
- `outputs/qualitative_clean/traffic/visualization.jpg`
- `outputs/qualitative_clean/tennis/visualization.jpg`
- `outputs/qualitative_clean/cat_keyboard/visualization.jpg`

Lower-threshold visualizations for failure analysis:

- `outputs/qualitative_nms/workspace/visualization.jpg`
- `outputs/qualitative_nms/traffic/visualization.jpg`
- `outputs/qualitative_nms/tennis/visualization.jpg`
- `outputs/qualitative_nms/cat_keyboard/visualization.jpg`

Report notes are summarized in:

- `report_notes.md`

Because `outputs/` is ignored, regenerate the visualizations locally with
`scripts/owlvit_demo.py` before preparing the final report package.

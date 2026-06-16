# Report Notes: Open-Vocabulary Object Detection and Visual Grounding

## Project Scope

We reproduced an open-vocabulary object detection pipeline using OWL-ViT. The
model receives an image and arbitrary text prompts, then predicts bounding boxes
for regions matching the prompts. This covers the Task 4 requirement of
reproducing an open-vocabulary detection or visual grounding pipeline and
evaluating it on a public dataset subset.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM
- PyTorch: 2.11.0+cu128
- Torchvision: 0.26.0+cu128
- CUDA runtime used by PyTorch: 12.8
- Transformers: 5.5.3
- Model: `google/owlvit-base-patch32`

## Reproduction Pipeline

1. Load an input image.
2. Define text prompts, such as `cat`, `remote control`, or `stop sign`.
3. Use OWL-ViT to compute image-text matching scores and bounding boxes.
4. Apply score thresholding.
5. Optionally apply class-wise NMS for cleaner visualization.
6. Save predictions in JSON format and draw bounding boxes on the image.

Main script:

```powershell
python scripts\owlvit_demo.py --image-path data\coco\val2017\000000001503.jpg --queries "laptop, keyboard, mouse, tv"
```

## COCO Subset Evaluation

Dataset:

- COCO val2017
- Subset size: first 100 images by COCO image id
- Categories: all 80 COCO detection categories used as text prompts
- Metric: standard COCO bounding-box AP/AR using `pycocotools`

Command:

```powershell
python scripts\evaluate_coco_owlvit.py --data-dir data\coco --output-dir outputs\coco_owlvit_eval_100 --max-images 100 --top-k 100
```

Results without NMS:

| Metric | Value |
| --- | ---: |
| AP@[IoU=.50:.95] | 0.336 |
| AP@0.50 | 0.510 |
| AP@0.75 | 0.353 |
| AP small | 0.178 |
| AP medium | 0.368 |
| AP large | 0.563 |
| AR maxDets=1 | 0.341 |
| AR maxDets=10 | 0.510 |
| AR maxDets=100 | 0.520 |
| AR small | 0.254 |
| AR medium | 0.528 |
| AR large | 0.733 |

NMS ablation:

```powershell
python scripts\evaluate_coco_owlvit.py --data-dir data\coco --output-dir outputs\coco_owlvit_eval_100_nms --max-images 100 --top-k 100 --nms-threshold 0.5
```

Class-wise NMS reduced duplicate visual boxes, but AP on this subset decreased
from 0.336 to 0.323 and AR@100 decreased from 0.520 to 0.475. This suggests that
NMS is useful for presentation-quality visualizations but may remove valid
overlapping hypotheses in evaluation.

## Qualitative Results

Clean visualizations:

- `outputs/qualitative_clean/workspace/visualization.jpg`
- `outputs/qualitative_clean/traffic/visualization.jpg`
- `outputs/qualitative_clean/tennis/visualization.jpg`
- `outputs/qualitative_clean/cat_keyboard/visualization.jpg`

Lower-threshold visualizations for failure analysis:

- `outputs/qualitative_nms/workspace/visualization.jpg`
- `outputs/qualitative_nms/traffic/visualization.jpg`
- `outputs/qualitative_nms/tennis/visualization.jpg`
- `outputs/qualitative_nms/cat_keyboard/visualization.jpg`

Observed strengths:

- Detects common COCO categories without task-specific fine-tuning.
- Works with multi-word prompts such as `remote control`, `tennis racket`, and
  `stop sign`.
- Large objects are localized better than small objects, consistent with AP
  large being much higher than AP small.

Observed limitations:

- Small objects are difficult. AP small is only 0.178 on the 100-image subset.
- Low score thresholds improve recall but introduce many duplicate or noisy
  boxes.
- Phrase-level grounding is weaker than category-level detection. In the
  `cat on keyboard` example, the model detects the cat but does not reliably
  localize the keyboard or the full relation.
- Open-vocabulary prompts are sensitive to wording. Dataset category names may
  not always match the phrases a human would naturally use.
- Similar categories and nearby objects can cause label confusion, especially in
  cluttered scenes.

## Recommended Presentation Structure

1. Motivation: closed-set detection vs. open-vocabulary detection.
2. Method: OWL-ViT image encoder, text encoder, image-text matching, box output.
3. Implementation: scripts, GPU setup, COCO subset evaluation.
4. Quantitative results: COCO 100-image AP/AR table.
5. Qualitative results: successful examples and failure cases.
6. Discussion: prompt sensitivity, small-object weakness, NMS tradeoff.
7. Future work: compare with Grounding DINO or YOLO-World, evaluate on RefCOCO
   for stronger visual grounding analysis.


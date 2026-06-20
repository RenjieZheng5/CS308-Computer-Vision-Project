# Report Notes: Open-Vocabulary Object Detection and Visual Grounding

## Project Scope

We reproduced an open-vocabulary object detection and visual grounding pipeline
using three pretrained models: OWL-ViT, Grounding DINO, and YOLO-World. This
meets the Task 4 requirement and now uses a formal full-split evaluation
protocol rather than a diagnostic subset.

## Environment

- Development machine: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM
- Server rerun: 4 x TITAN RTX
- Server PyTorch: 2.6.0+cu124
- Server Torchvision: 0.21.0+cu124
- Transformers: 5.5.3
- Models: `google/owlvit-base-patch32`, `IDEA-Research/grounding-dino-tiny`,
  `yolov8s-worldv2.pt`

## What Was Evaluated

1. Full COCO val2017 detection using all 5,000 images.
2. Full RefCOCO validation grounding using all referring expressions.
3. Threshold sensitivity on saved COCO predictions.
4. OWL-ViT NMS ablation as a development diagnostic.

## Final Results Snapshot

### COCO val2017

| Model | AP | AP50 | AR100 | FPS |
| --- | ---: | ---: | ---: | ---: |
| OWL-ViT Base | 0.237 | 0.385 | 0.469 | 24.3 |
| Grounding DINO Tiny | 0.421 | 0.557 | 0.536 | 4.5 |
| YOLO-World v2 Small | 0.366 | 0.510 | 0.547 | 84.6 |

### RefCOCO val

| Model | Acc@0.5 | Acc@0.75 | Mean IoU | FPS |
| --- | ---: | ---: | ---: | ---: |
| OWL-ViT Base | 0.425 | 0.342 | 0.423 | 29.9 |
| Grounding DINO Tiny | 0.511 | 0.456 | 0.514 | 3.1 |
| YOLO-World v2 Small | 0.414 | 0.362 | 0.422 | 115.1 |

## Main Takeaways

- Grounding DINO is the most accurate on both COCO and RefCOCO.
- YOLO-World is the best efficiency choice.
- OWL-ViT is the simplest baseline, but weaker on accuracy.
- Thresholding and NMS are useful for visualization, but harmful if pushed too
  aggressively in evaluation.

## 15-Minute Defense Structure

1. Motivation: why open-vocabulary detection and grounding matter.
2. Method: the three models and the shared evaluation pipeline.
3. COCO full-split results: accuracy vs. efficiency.
4. RefCOCO full-split results: grounding quality.
5. Ablations: threshold sensitivity and NMS.
6. Qualitative examples and failure cases.
7. Limitations and future work.

## What to Emphasize in Q&A

- The project is now based on full-split evaluation, not a subset.
- The RefCOCO experiment evaluates all expressions, which is the strongest
  grounding evidence in the project.
- The reported score tables in the paper and README are aligned.

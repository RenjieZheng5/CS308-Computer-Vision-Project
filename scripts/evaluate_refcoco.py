import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import requests
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    GroundingDinoForObjectDetection,
    OwlViTForObjectDetection,
)
from ultralytics import YOLOWorld

from coco_eval_utils import RuntimeTracker


REFCOCO_ROWS_URL = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=lmms-lab%2FRefCOCO&config=default&split=val"
)
OWL_MODEL = "google/owlvit-base-patch32"
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
YOLO_WORLD_MODEL = "yolov8s-worldv2.pt"


def download_file(url: str, target: Path, retries: int = 5) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            temporary.replace(target)
            return
        except requests.RequestException as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def ensure_manifest(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl.exe",
        "--http1.1",
        "-L",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        REFCOCO_ROWS_URL,
        "-o",
        str(path),
    ]
    subprocess.run(command, check=True)
    return path


def coco_file_name(refcoco_name: str) -> str:
    match = re.search(r"COCO_train2014_(\d{12})", refcoco_name)
    if not match:
        raise ValueError(f"Cannot extract COCO image id from {refcoco_name}")
    return f"COCO_train2014_{match.group(1)}.jpg"


def load_samples(manifest_path: Path, max_samples: int) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload["rows"][:max_samples]
    samples = []
    for item in rows:
        row = item["row"]
        file_name = coco_file_name(row["file_name"])
        samples.append(
            {
                "row_idx": item["row_idx"],
                "question_id": row["question_id"],
                "file_name": file_name,
                "expression": row["answer"][0],
                "all_expressions": row["answer"],
                "bbox_xywh": [float(value) for value in row["bbox"]],
            }
        )
    return samples


def ensure_image(sample: dict, image_dir: Path) -> Path:
    target = image_dir / sample["file_name"]
    url = f"http://images.cocodataset.org/train2014/{sample['file_name']}"
    download_file(url, target)
    return target


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def top_owlvit_box(processor, model, image, expression, device, tracker):
    inputs = tracker.measure(
        "preprocess",
        lambda: {
            key: value.to(device)
            for key, value in processor(
                text=[[expression]], images=image, return_tensors="pt"
            ).items()
        },
    )
    outputs = tracker.measure(
        "inference",
        lambda: model(**inputs),
    )

    def postprocess():
        target_sizes = torch.tensor([image.size[::-1]], device=device)
        result = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=0.0,
            target_sizes=target_sizes,
            text_labels=[[expression]],
        )[0]
        if not len(result["scores"]):
            return None, None
        best = int(torch.argmax(result["scores"]))
        return (
            [float(value) for value in result["boxes"][best].detach().cpu().tolist()],
            float(result["scores"][best].detach().cpu()),
        )

    return tracker.measure("postprocess", postprocess)


def top_grounding_dino_box(processor, model, image, expression, device, tracker):
    prompt = expression.lower().strip().rstrip(".") + "."
    inputs = tracker.measure(
        "preprocess",
        lambda: {
            key: value.to(device)
            for key, value in processor(
                images=image, text=prompt, return_tensors="pt"
            ).items()
        },
    )
    outputs = tracker.measure("inference", lambda: model(**inputs))

    def postprocess():
        target_sizes = torch.tensor([image.size[::-1]], device=device)
        result = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.get("input_ids"),
            threshold=0.0,
            text_threshold=0.0,
            target_sizes=target_sizes,
            text_labels=[[prompt]],
        )[0]
        if not len(result["scores"]):
            return None, None
        best = int(torch.argmax(result["scores"]))
        return (
            [float(value) for value in result["boxes"][best].detach().cpu().tolist()],
            float(result["scores"][best].detach().cpu()),
        )

    return tracker.measure("postprocess", postprocess)


def top_yolo_world_box(model, image, expression, device, image_size):
    classes = [expression]
    core = model.model
    core_device = next(core.model.parameters()).device
    if not getattr(core, "clip_model", None):
        model.set_classes(classes)
    else:
        tokens = core.clip_model.tokenize(classes).to(core_device)
        features = core.clip_model.encode_text(tokens).detach()
        core.txt_feats = features.reshape(1, len(classes), features.shape[-1])
        core.model[-1].nc = len(classes)
        core.names = classes
        if model.predictor:
            model.predictor.model.names = classes
    result = model.predict(
        image,
        device=0 if device == "cuda" else "cpu",
        verbose=False,
        conf=0.001,
        iou=0.7,
        max_det=1,
        imgsz=image_size,
    )[0]
    if not len(result.boxes):
        return None, None, result.speed
    return (
        [float(value) for value in result.boxes.xyxy[0].detach().cpu().tolist()],
        float(result.boxes.conf[0].detach().cpu()),
        result.speed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate top-1 referring-expression localization on 100 RefCOCO val regions."
    )
    parser.add_argument(
        "--model-type",
        choices=["owlvit", "grounding-dino", "yolo-world"],
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--data-dir", default="data/refcoco")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    defaults = {
        "owlvit": OWL_MODEL,
        "grounding-dino": GROUNDING_DINO_MODEL,
        "yolo-world": YOLO_WORLD_MODEL,
    }
    model_name = args.model or defaults[args.model_type]
    data_dir = Path(args.data_dir)
    image_dir = data_dir / "train2014"
    output_dir = Path(
        args.output_dir or f"outputs/refcoco_{args.model_type}_eval_100"
    )
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ensure_manifest(data_dir / "refcoco_val_first_rows.json")
    samples = load_samples(manifest_path, args.max_samples)
    for sample in tqdm(samples, desc="downloading RefCOCO images"):
        ensure_image(sample, image_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = RuntimeTracker(device)
    processor = None
    if args.model_type == "owlvit":
        processor = AutoProcessor.from_pretrained(model_name)
        model = OwlViTForObjectDetection.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        model.eval()
    elif args.model_type == "grounding-dino":
        processor = AutoProcessor.from_pretrained(model_name)
        model = GroundingDinoForObjectDetection.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        model.eval()
    else:
        model = YOLOWorld(model_name)

    warmup = samples[0]
    warmup_image = Image.open(ensure_image(warmup, image_dir)).convert("RGB")
    with torch.inference_mode():
        if args.model_type == "owlvit":
            top_owlvit_box(
                processor, model, warmup_image, warmup["expression"], device, tracker
            )
        elif args.model_type == "grounding-dino":
            top_grounding_dino_box(
                processor, model, warmup_image, warmup["expression"], device, tracker
            )
        else:
            top_yolo_world_box(
                model, warmup_image, warmup["expression"], device, args.image_size
            )
    tracker = RuntimeTracker(device)

    predictions = []
    for sample in tqdm(samples, desc=f"evaluating {args.model_type}"):
        image = Image.open(ensure_image(sample, image_dir)).convert("RGB")
        with torch.inference_mode():
            if args.model_type == "owlvit":
                predicted_box, score = top_owlvit_box(
                    processor, model, image, sample["expression"], device, tracker
                )
            elif args.model_type == "grounding-dino":
                predicted_box, score = top_grounding_dino_box(
                    processor, model, image, sample["expression"], device, tracker
                )
            else:
                predicted_box, score, speed = top_yolo_world_box(
                    model, image, sample["expression"], device, args.image_size
                )
                tracker.preprocess_seconds += float(speed["preprocess"]) / 1000.0
                tracker.inference_seconds += float(speed["inference"]) / 1000.0
                tracker.postprocess_seconds += float(speed["postprocess"]) / 1000.0
        tracker.add_image()
        ground_truth = xywh_to_xyxy(sample["bbox_xywh"])
        iou = box_iou(predicted_box, ground_truth) if predicted_box else 0.0
        predictions.append(
            {
                **sample,
                "ground_truth_xyxy": ground_truth,
                "predicted_xyxy": predicted_box,
                "score": score,
                "iou": iou,
                "correct_at_0.5": iou >= 0.5,
            }
        )

    ious = [item["iou"] for item in predictions]
    metrics = {
        "dataset": "RefCOCO validation mirror (lmms-lab/RefCOCO)",
        "protocol": "first 100 region rows; first human expression per region; top-1 box",
        "model_type": args.model_type,
        "model": model_name,
        "device": device,
        "samples": len(predictions),
        "accuracy_at_iou_0.5": sum(iou >= 0.5 for iou in ious) / len(ious),
        "accuracy_at_iou_0.75": sum(iou >= 0.75 for iou in ious) / len(ious),
        "mean_iou": sum(ious) / len(ious),
        "median_iou": sorted(ious)[len(ious) // 2],
        "runtime": tracker.summary(),
    }
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

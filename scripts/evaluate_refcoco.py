import argparse
import json
import re
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


REFCOCO_DATASET = "lmms-lab/RefCOCO"
REFCOCO_CONFIG = "default"
REFCOCO_ROWS_URL = "https://datasets-server.huggingface.co/rows"
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
            if temporary.exists():
                temporary.unlink()
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
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def fetch_rows_page(split: str, offset: int, length: int, retries: int = 5) -> list[dict]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                REFCOCO_ROWS_URL,
                params={
                    "dataset": REFCOCO_DATASET,
                    "config": REFCOCO_CONFIG,
                    "split": split,
                    "offset": offset,
                    "length": length,
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("rows", [])
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(
        f"Failed to fetch RefCOCO rows for split '{split}' at offset {offset}"
    ) from last_error


def ensure_manifest(path: Path, split: str, page_size: int, refresh: bool = False) -> Path:
    if path.exists() and not refresh:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset = 0

    while True:
        page_rows = fetch_rows_page(split=split, offset=offset, length=page_size)
        if not page_rows:
            break
        rows.extend(page_rows)
        if len(page_rows) < page_size:
            break
        offset += len(page_rows)

    if not rows:
        raise RuntimeError(
            f"No RefCOCO rows were fetched for split '{split}'. Check network access."
        )

    manifest = {
        "dataset": REFCOCO_DATASET,
        "config": REFCOCO_CONFIG,
        "split": split,
        "row_count": len(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def coco_file_name(refcoco_name: str) -> str:
    match = re.search(r"COCO_train2014_(\d{12})", refcoco_name)
    if not match:
        raise ValueError(f"Cannot extract COCO image id from {refcoco_name}")
    return f"COCO_train2014_{match.group(1)}.jpg"


def normalize_expressions(raw_expression) -> list[str]:
    if raw_expression is None:
        return []
    if isinstance(raw_expression, str):
        values = [raw_expression]
    else:
        values = list(raw_expression)
    expressions = []
    for value in values:
        text = str(value).strip()
        if text:
            expressions.append(text)
    return expressions


def load_samples(
    manifest_path: Path, max_rows: int, expression_mode: str
) -> tuple[list[dict], dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if max_rows > 0:
        rows = rows[:max_rows]

    samples: list[dict] = []
    total_expressions = 0
    skipped_rows = 0

    for item in rows:
        row = item["row"]
        file_name = coco_file_name(row["file_name"])
        expressions = normalize_expressions(
            row.get("answer")
            or row.get("answers")
            or row.get("expressions")
            or row.get("sentences")
        )
        if expression_mode == "first":
            expressions = expressions[:1]
        if not expressions:
            skipped_rows += 1
            continue

        total_expressions += len(expressions)
        bbox_xywh = [float(value) for value in row["bbox"]]
        for expression_idx, expression in enumerate(expressions):
            samples.append(
                {
                    "row_idx": item["row_idx"],
                    "question_id": row["question_id"],
                    "expression_idx": expression_idx,
                    "expression": expression,
                    "all_expressions": expressions,
                    "file_name": file_name,
                    "bbox_xywh": bbox_xywh,
                }
            )

    metadata = {
        "row_count": len(rows),
        "sample_count": len(samples),
        "expression_mode": expression_mode,
        "rows_skipped_without_expressions": skipped_rows,
        "mean_expressions_per_row": total_expressions / max(len(rows) - skipped_rows, 1),
    }
    return samples, metadata


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
    outputs = tracker.measure("inference", lambda: model(**inputs))

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
        description="Evaluate referring-expression localization on a RefCOCO split."
    )
    parser.add_argument(
        "--model-type",
        choices=["owlvit", "grounding-dino", "yolo-world"],
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--data-dir", default="data/refcoco")
    parser.add_argument("--output-dir")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--max-rows",
        "--max-samples",
        dest="max_rows",
        type=int,
        default=0,
        help="Limit the number of RefCOCO region rows. 0 means the full split.",
    )
    parser.add_argument(
        "--expression-mode",
        choices=["all", "first"],
        default="all",
        help="Evaluate all referring expressions per row, or only the first one.",
    )
    parser.add_argument(
        "--manifest-path",
        help="Optional cache path for the downloaded RefCOCO rows manifest.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Number of rows to fetch per datasets-server request.",
    )
    parser.add_argument("--refresh-manifest", action="store_true")
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
    manifest_path = Path(
        args.manifest_path or data_dir / f"refcoco_{args.split}_rows.json"
    )
    row_tag = "full" if args.max_rows <= 0 else f"rows{args.max_rows}"
    output_dir = Path(
        args.output_dir
        or f"outputs/refcoco_{args.model_type}_eval_{args.split}_{row_tag}_{args.expression_mode}"
    )
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_manifest(
        manifest_path,
        split=args.split,
        page_size=args.page_size,
        refresh=args.refresh_manifest,
    )
    samples, sample_metadata = load_samples(
        manifest_path, args.max_rows, args.expression_mode
    )
    if not samples:
        raise RuntimeError(
            "No RefCOCO evaluation samples were loaded. "
            "Check the split and expression parsing."
        )

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
        "dataset": f"RefCOCO {args.split} split ({REFCOCO_DATASET})",
        "protocol": (
            "all region rows" if args.max_rows <= 0 else f"first {args.max_rows} region rows"
        )
        + f"; expression mode={args.expression_mode}; top-1 box",
        "model_type": args.model_type,
        "model": model_name,
        "device": device,
        "samples": len(predictions),
        "rows": sample_metadata["row_count"],
        "mean_expressions_per_row": sample_metadata["mean_expressions_per_row"],
        "rows_skipped_without_expressions": sample_metadata[
            "rows_skipped_without_expressions"
        ],
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

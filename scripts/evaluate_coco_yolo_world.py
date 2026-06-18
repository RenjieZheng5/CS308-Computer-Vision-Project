import argparse
import json
import time
import zipfile
from pathlib import Path

import requests
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import YOLOWorld

from coco_eval_utils import select_image_ids


DEFAULT_MODEL = "yolov8s-worldv2.pt"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url: str, target: Path, retries: int = 5) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if tmp_target.exists():
                tmp_target.unlink()
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with tmp_target.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp_target.replace(target)
            return
        except requests.RequestException as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def ensure_coco_annotations(data_dir: Path) -> Path:
    annotations_path = data_dir / "annotations" / "instances_val2017.json"
    if annotations_path.exists():
        return annotations_path
    zip_path = data_dir / "annotations_trainval2017.zip"
    download_file(ANNOTATIONS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extract("annotations/instances_val2017.json", data_dir)
    return annotations_path


def ensure_image(image_info: dict, image_dir: Path) -> Path:
    path = image_dir / image_info["file_name"]
    if not path.exists():
        download_file(image_info["coco_url"], path)
    return path


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x0, y0, x1, y1 = box
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO-World on a COCO val2017 subset.")
    parser.add_argument("--data-dir", default="data/coco")
    parser.add_argument("--output-dir", default="outputs/coco_yolo_world_eval_500")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--sampling", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=308)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "val2017"
    output_dir = Path(args.output_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = ensure_coco_annotations(data_dir)
    coco = COCO(str(annotation_file))
    categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda item: item["id"])
    class_names = [category["name"] for category in categories]
    class_to_category_id = {idx: category["id"] for idx, category in enumerate(categories)}
    image_ids = select_image_ids(coco.getImgIds(), args.max_images, args.sampling, args.seed)
    image_infos = coco.loadImgs(image_ids)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLOWorld(args.model)
    model.set_classes(class_names)
    warmup_image = Image.open(ensure_image(image_infos[0], image_dir)).convert("RGB")
    model.predict(
        warmup_image,
        device=device,
        verbose=False,
        conf=args.confidence,
        iou=args.iou_threshold,
        max_det=args.top_k,
        imgsz=args.image_size,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    detections = []
    preprocess_ms = 0.0
    inference_ms = 0.0
    postprocess_ms = 0.0

    for image_info in tqdm(image_infos, desc="evaluating"):
        image_path = ensure_image(image_info, image_dir)
        image = Image.open(image_path).convert("RGB")
        result = model.predict(
            image,
            device=device,
            verbose=False,
            conf=args.confidence,
            iou=args.iou_threshold,
            max_det=args.top_k,
            imgsz=args.image_size,
        )[0]
        preprocess_ms += float(result.speed["preprocess"])
        inference_ms += float(result.speed["inference"])
        postprocess_ms += float(result.speed["postprocess"])

        boxes = result.boxes.xyxy.detach().cpu().float()
        scores = result.boxes.conf.detach().cpu().float()
        labels = result.boxes.cls.detach().cpu().long()
        order = torch.argsort(scores, descending=True)[: args.top_k]
        for idx in order.tolist():
            detections.append(
                {
                    "image_id": image_info["id"],
                    "category_id": class_to_category_id[int(labels[idx])],
                    "bbox": xyxy_to_xywh([float(value) for value in boxes[idx].tolist()]),
                    "score": float(scores[idx]),
                }
            )

    predictions_path = output_dir / "coco_predictions.json"
    predictions_path.write_text(json.dumps(detections), encoding="utf-8")
    if not detections:
        raise RuntimeError("No detections were produced. Lower --confidence and retry.")

    coco_dt = coco.loadRes(str(predictions_path))
    evaluator = COCOeval(coco, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metric_names = [
        "AP@[IoU=.50:.95]",
        "AP@0.50",
        "AP@0.75",
        "AP small",
        "AP medium",
        "AP large",
        "AR maxDets=1",
        "AR maxDets=10",
        "AR maxDets=100",
        "AR small",
        "AR medium",
        "AR large",
    ]
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    runtime = {
        "images": len(image_infos),
        "preprocess_seconds": preprocess_ms / 1000.0,
        "inference_seconds": inference_ms / 1000.0,
        "postprocess_seconds": postprocess_ms / 1000.0,
        "pipeline_seconds": total_ms / 1000.0,
        "inference_ms_per_image": inference_ms / max(len(image_infos), 1),
        "pipeline_ms_per_image": total_ms / max(len(image_infos), 1),
        "inference_fps": 1000.0 * len(image_infos) / max(inference_ms, 1e-9),
        "pipeline_fps": 1000.0 * len(image_infos) / max(total_ms, 1e-9),
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
        ),
    }
    metrics = {
        "model": args.model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_images": args.max_images,
        "sampling": args.sampling,
        "seed": args.seed,
        "image_ids": image_ids,
        "confidence": args.confidence,
        "iou_threshold": args.iou_threshold,
        "top_k": args.top_k,
        "image_size": args.image_size,
        "runtime": runtime,
        "metrics": {name: float(value) for name, value in zip(metric_names, evaluator.stats)},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved: {predictions_path}")
    print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()

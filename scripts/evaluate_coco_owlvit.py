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
from transformers import AutoProcessor, OwlViTForObjectDetection

from owlvit_demo import DEFAULT_MODEL, detections_to_rows, post_process_detections


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
                total = int(response.headers.get("content-length", 0))
                with tqdm(total=total, unit="B", unit_scale=True, desc=target.name) as progress:
                    with tmp_target.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                                progress.update(len(chunk))
            tmp_target.replace(target)
            return
        except requests.RequestException as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(2 * attempt)

    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


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
    if path.exists():
        return path
    download_file(image_info["coco_url"], path)
    return path


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x0, y0, x1, y1 = box
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OWL-ViT on a COCO val2017 subset.")
    parser.add_argument("--data-dir", default="data/coco")
    parser.add_argument("--output-dir", default="outputs/coco_owlvit_eval")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.01)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    image_dir = data_dir / "val2017"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    annotation_file = ensure_coco_annotations(data_dir)
    coco = COCO(str(annotation_file))
    categories = coco.loadCats(coco.getCatIds())
    categories = sorted(categories, key=lambda item: item["id"])
    queries = [category["name"] for category in categories]
    label_to_category_id = {idx: category["id"] for idx, category in enumerate(categories)}

    image_ids = sorted(coco.getImgIds())[: args.max_images]
    image_infos = coco.loadImgs(image_ids)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = OwlViTForObjectDetection.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    detections = []
    for image_info in tqdm(image_infos, desc="evaluating"):
        image_path = ensure_image(image_info, image_dir)
        image = Image.open(image_path).convert("RGB")
        texts = [queries]
        inputs = processor(text=texts, images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = post_process_detections(
            processor=processor,
            outputs=outputs,
            threshold=args.score_threshold,
            target_sizes=target_sizes,
            text_labels=texts,
        )[0]

        rows = detections_to_rows(results, args.nms_threshold)
        for score_value, label_idx, box_xyxy in rows[: args.top_k]:
            detections.append(
                {
                    "image_id": image_info["id"],
                    "category_id": label_to_category_id[label_idx],
                    "bbox": xyxy_to_xywh(box_xyxy),
                    "score": score_value,
                }
            )

    predictions_path = output_dir / "coco_predictions.json"
    predictions_path.write_text(json.dumps(detections), encoding="utf-8")

    if not detections:
        raise RuntimeError("No detections were produced. Lower --score-threshold and retry.")

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
    metrics = {
        "model": args.model,
        "device": device,
        "max_images": args.max_images,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "top_k": args.top_k,
        "metrics": {name: float(value) for name, value in zip(metric_names, evaluator.stats)},
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved: {predictions_path}")
    print(f"saved: {metrics_path}")


if __name__ == "__main__":
    main()

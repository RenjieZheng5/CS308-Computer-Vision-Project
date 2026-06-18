import argparse
import inspect
import json
from pathlib import Path
from typing import Iterable

import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from transformers import AutoProcessor, OwlViTForObjectDetection


DEFAULT_MODEL = "google/owlvit-base-patch32"
DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_QUERIES = ["cat", "remote control", "couch"]


def load_image(image_path: str | None, image_url: str | None) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")

    url = image_url or DEFAULT_IMAGE_URL
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    return Image.open(response.raw).convert("RGB")


def parse_queries(raw_queries: str | None) -> list[str]:
    if not raw_queries:
        return DEFAULT_QUERIES
    separators = [",", ";", "|"]
    for separator in separators:
        if separator in raw_queries:
            return [item.strip() for item in raw_queries.split(separator) if item.strip()]
    return [raw_queries.strip()]


def post_process_detections(processor, outputs, threshold, target_sizes, text_labels):
    if hasattr(processor, "post_process_object_detection"):
        return processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )

    method = processor.post_process_grounded_object_detection
    kwargs = {
        "outputs": outputs,
        "threshold": threshold,
        "target_sizes": target_sizes,
    }
    if "text_labels" in inspect.signature(method).parameters:
        kwargs["text_labels"] = text_labels
    return method(**kwargs)


def detections_to_rows(results: dict, nms_threshold: float | None) -> list[tuple[float, int, list[float]]]:
    scores = results["scores"].detach().cpu().float()
    labels = results["labels"].detach().cpu()
    boxes = results["boxes"].detach().cpu().float()

    if nms_threshold is None or nms_threshold < 0:
        keep_indices = list(range(len(scores)))
    else:
        keep_indices = []
        for label in labels.unique():
            label_indices = torch.where(labels == label)[0]
            selected = nms(boxes[label_indices], scores[label_indices], nms_threshold)
            keep_indices.extend(label_indices[selected].tolist())

    keep_indices.sort(key=lambda idx: float(scores[idx]), reverse=True)
    rows = []
    for idx in keep_indices:
        rows.append(
            (
                float(scores[idx]),
                int(labels[idx]),
                [float(value) for value in boxes[idx].tolist()],
            )
        )
    return rows


def draw_predictions(
    image: Image.Image,
    predictions: Iterable[dict],
    output_path: Path,
) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    palette = ["red", "lime", "dodgerblue", "yellow", "magenta", "cyan", "orange"]
    for idx, pred in enumerate(predictions):
        box = pred["box_xyxy"]
        color = palette[idx % len(palette)]
        label = f'{pred["label"]} {pred["score"]:.2f}'
        draw.rectangle(box, outline=color, width=4)
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), label, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OWL-ViT open-vocabulary object detection.")
    parser.add_argument("--image-path", default=None, help="Local image path.")
    parser.add_argument("--image-url", default=None, help="Image URL used when --image-path is omitted.")
    parser.add_argument("--queries", default=None, help='Text queries, e.g. "cat, remote control, couch".')
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id.")
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default="outputs/owlvit_demo")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    output_dir = Path(args.output_dir)

    image = load_image(args.image_path, args.image_url)
    queries = parse_queries(args.queries)
    texts = [queries]

    processor = AutoProcessor.from_pretrained(args.model)
    model = OwlViTForObjectDetection.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

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

    predictions = []
    for score, label, box in detections_to_rows(results, args.nms_threshold):
        box_values = [round(value, 2) for value in box]
        label_text = queries[label]
        predictions.append(
            {
                "label": label_text,
                "score": round(score, 4),
                "box_xyxy": box_values,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "device": device,
                "queries": queries,
                "score_threshold": args.score_threshold,
                "nms_threshold": args.nms_threshold,
                "predictions": predictions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    draw_predictions(image, predictions, output_dir / "visualization.jpg")

    print(f"device: {device}")
    print(f"queries: {queries}")
    print(f"predictions: {len(predictions)}")
    for pred in predictions:
        print(f'{pred["label"]:>16} {pred["score"]:.3f} {pred["box_xyxy"]}')
    print(f"saved: {output_dir / 'predictions.json'}")
    print(f"saved: {output_dir / 'visualization.jpg'}")


if __name__ == "__main__":
    main()

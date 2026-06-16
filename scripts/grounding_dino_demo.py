import argparse
import json
from pathlib import Path
from typing import Iterable

import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from transformers import AutoProcessor, GroundingDinoForObjectDetection


DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"
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
    for separator in [",", ";", "|"]:
        if separator in raw_queries:
            return [item.strip() for item in raw_queries.split(separator) if item.strip()]
    return [raw_queries.strip()]


def build_prompt(queries: list[str]) -> str:
    return ". ".join(query.strip().lower().rstrip(".") for query in queries) + "."


def draw_predictions(image: Image.Image, predictions: Iterable[dict], output_path: Path) -> None:
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


def apply_nms(predictions: list[dict], nms_threshold: float) -> list[dict]:
    if not predictions:
        return []

    grouped: dict[str, list[int]] = {}
    for idx, pred in enumerate(predictions):
        grouped.setdefault(pred["label"], []).append(idx)

    keep: list[int] = []
    for indices in grouped.values():
        boxes = torch.tensor([predictions[idx]["box_xyxy"] for idx in indices], dtype=torch.float32)
        scores = torch.tensor([predictions[idx]["score"] for idx in indices], dtype=torch.float32)
        selected = nms(boxes, scores, nms_threshold)
        keep.extend(indices[int(idx)] for idx in selected)

    keep.sort(key=lambda idx: predictions[idx]["score"], reverse=True)
    return [predictions[idx] for idx in keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grounding DINO open-vocabulary detection.")
    parser.add_argument("--image-path", default=None, help="Local image path.")
    parser.add_argument("--image-url", default=None, help="Image URL used when --image-path is omitted.")
    parser.add_argument("--queries", default=None, help='Text queries, e.g. "cat, remote control, couch".')
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model id.")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default="outputs/grounding_dino_demo")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Grounding DINO's Swin backbone path in Transformers 5.5 is safest in fp32
    # on Windows; fp16 can produce pixel/input dtype mismatches.
    dtype = torch.float32
    output_dir = Path(args.output_dir)

    image = load_image(args.image_path, args.image_url)
    queries = parse_queries(args.queries)
    text_prompt = build_prompt(queries)

    processor = AutoProcessor.from_pretrained(args.model)
    model = GroundingDinoForObjectDetection.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.get("input_ids"),
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=target_sizes,
        text_labels=[[text_prompt]],
    )[0]

    predictions = []
    labels = results.get("text_labels") or results.get("labels") or []
    for score, label, box in zip(results["scores"], labels, results["boxes"]):
        label_text = str(label).strip().rstrip(".")
        predictions.append(
            {
                "label": label_text,
                "score": round(float(score.detach().cpu()), 4),
                "box_xyxy": [round(float(value), 2) for value in box.detach().cpu().tolist()],
            }
        )

    predictions = apply_nms(predictions, args.nms_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "device": device,
                "queries": queries,
                "prompt": text_prompt,
                "box_threshold": args.box_threshold,
                "text_threshold": args.text_threshold,
                "nms_threshold": args.nms_threshold,
                "predictions": predictions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    draw_predictions(image, predictions, output_dir / "visualization.jpg")

    print(f"device: {device}")
    print(f"prompt: {text_prompt}")
    print(f"predictions: {len(predictions)}")
    for pred in predictions:
        print(f'{pred["label"]:>20} {pred["score"]:.3f} {pred["box_xyxy"]}')
    print(f"saved: {output_dir / 'predictions.json'}")
    print(f"saved: {output_dir / 'visualization.jpg'}")


if __name__ == "__main__":
    main()

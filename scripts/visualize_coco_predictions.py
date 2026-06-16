import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def color_for_category(category_id: int) -> str:
    palette = [
        "red",
        "lime",
        "dodgerblue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "white",
        "deepskyblue",
        "springgreen",
    ]
    return palette[category_id % len(palette)]


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def draw_predictions(image: Image.Image, rows: list[dict], category_names: dict[int, str]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = get_font()

    if not rows:
        label = "no detections above threshold"
        box = draw.textbbox((8, 8), label, font=font)
        draw.rectangle(box, fill="black")
        draw.text((8, 8), label, fill="white", font=font)
        return canvas

    for pred in rows:
        category_id = int(pred["category_id"])
        label = category_names.get(category_id, str(category_id))
        score = float(pred["score"])
        box = xywh_to_xyxy(pred["bbox"])
        color = color_for_category(category_id)
        text = f"{label} {score:.2f}"

        draw.rectangle(box, outline=color, width=3)
        text_box = draw.textbbox((box[0], box[1]), text, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), text, fill="black", font=font)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw COCO-format prediction JSON on COCO images.")
    parser.add_argument("--annotation-file", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--image-dir", default="data/coco/val2017")
    parser.add_argument("--predictions", required=True, help="COCO detection results JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=None, help="Use first N COCO image ids from annotations.")
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=20, help="Maximum predictions drawn per image.")
    args = parser.parse_args()

    annotation_file = Path(args.annotation_file)
    image_dir = Path(args.image_dir)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_json(annotation_file)
    predictions = load_json(predictions_path)

    category_names = {int(category["id"]): category["name"] for category in annotations["categories"]}
    images = {int(image["id"]): image for image in annotations["images"]}

    predictions_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in predictions:
        if float(pred.get("score", 0.0)) >= args.score_threshold:
            predictions_by_image[int(pred["image_id"])].append(pred)

    if args.max_images is not None:
        image_ids = sorted(images)[: args.max_images]
    else:
        image_ids = sorted({int(pred["image_id"]) for pred in predictions})

    written = 0
    for image_id in image_ids:
        image_info = images.get(image_id)
        if not image_info:
            continue

        image_path = image_dir / image_info["file_name"]
        if not image_path.exists():
            print(f"skip missing image: {image_path}")
            continue

        rows = sorted(
            predictions_by_image.get(image_id, []),
            key=lambda item: float(item["score"]),
            reverse=True,
        )[: args.top_k]

        image = Image.open(image_path).convert("RGB")
        canvas = draw_predictions(image, rows, category_names)
        output_path = output_dir / f"{image_id:012d}_{image_info['file_name']}"
        canvas.save(output_path)
        written += 1

    print(f"predictions: {predictions_path}")
    print(f"output_dir: {output_dir}")
    print(f"images_written: {written}")
    print(f"score_threshold: {args.score_threshold}")
    print(f"top_k: {args.top_k}")


if __name__ == "__main__":
    main()

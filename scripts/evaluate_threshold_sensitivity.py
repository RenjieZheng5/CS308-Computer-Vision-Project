import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = [
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


def evaluate(coco: COCO, predictions: list[dict], image_ids: list[int]) -> dict:
    coco_dt = coco.loadRes(predictions)
    evaluator = COCOeval(coco, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved COCO predictions at several score thresholds."
    )
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--thresholds", required=True, help="Comma-separated scores.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    thresholds = [float(value) for value in args.thresholds.split(",")]
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    source_metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    image_ids = source_metrics["image_ids"]
    coco = COCO(args.annotation_file)

    results = []
    for threshold in thresholds:
        filtered = [
            prediction
            for prediction in predictions
            if float(prediction["score"]) >= threshold
        ]
        if not filtered:
            results.append(
                {"threshold": threshold, "detections": 0, "metrics": None}
            )
            continue
        results.append(
            {
                "threshold": threshold,
                "detections": len(filtered),
                "detections_per_image": len(filtered) / len(image_ids),
                "metrics": evaluate(coco, filtered, image_ids),
            }
        )

    output = {
        "source_predictions": args.predictions,
        "images": len(image_ids),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

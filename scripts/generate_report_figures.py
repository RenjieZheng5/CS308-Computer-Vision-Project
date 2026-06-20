import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get("OUTPUT_ROOT", ROOT / "outputs"))
FIGURES = ROOT / "report" / "figures"
COLORS = ["#4C78A8", "#F58518", "#54A24B"]
MODELS = ["OWL-ViT", "Grounding DINO", "YOLO-World"]
COCO_MAX_IMAGES = os.environ.get("COCO_MAX_IMAGES", "0")
COCO_SAMPLING = os.environ.get("COCO_SAMPLING", "random")
COCO_SEED = os.environ.get("COCO_SEED")
REFCOCO_SPLIT = os.environ.get("REFCOCO_SPLIT", "val")
REFCOCO_MAX_ROWS = os.environ.get("REFCOCO_MAX_ROWS", "0")
REFCOCO_EXPRESSION_MODE = os.environ.get("REFCOCO_EXPRESSION_MODE", "all")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def add_labels(axis, bars, fmt="{:.3f}") -> None:
    for bar in bars:
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    if int(COCO_MAX_IMAGES) <= 0:
        coco_tags = ["full_val2017"]
    else:
        coco_base = f"{COCO_MAX_IMAGES}_{COCO_SAMPLING}"
        coco_tags = [coco_base]
        if COCO_SEED:
            coco_tags.insert(0, f"{coco_base}_seed{COCO_SEED}")

    ref_row_tag = "full" if int(REFCOCO_MAX_ROWS) <= 0 else f"rows{REFCOCO_MAX_ROWS}"
    refcoco_tag = f"{REFCOCO_SPLIT}_{ref_row_tag}_{REFCOCO_EXPRESSION_MODE}"

    coco_paths = [
        first_existing(
            [OUTPUT / f"coco_owlvit_eval_{tag}" / "metrics.json" for tag in coco_tags]
        ),
        first_existing(
            [OUTPUT / f"coco_grounding_dino_eval_{tag}" / "metrics.json" for tag in coco_tags]
        ),
        first_existing(
            [OUTPUT / f"coco_yolo_world_eval_{tag}" / "metrics.json" for tag in coco_tags]
        ),
    ]
    coco = [load(path) for path in coco_paths]

    metric_keys = ["AP@[IoU=.50:.95]", "AP@0.50", "AR maxDets=100"]
    metric_labels = ["AP", "AP50", "AR100"]
    x = np.arange(len(metric_keys))
    width = 0.24
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for index, (name, result, color) in enumerate(zip(MODELS, coco, COLORS)):
        values = [result["metrics"][key] for key in metric_keys]
        bars = axis.bar(x + (index - 1) * width, values, width, label=name, color=color)
        add_labels(axis, bars)
    axis.set_xticks(x, metric_labels)
    axis.set_ylabel("COCO metric")
    axis.set_ylim(0, 0.68)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, loc="upper center", frameon=False)
    save(figure, "coco_metrics")

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fps = [result["runtime"]["pipeline_fps"] for result in coco]
    memory = [result["runtime"]["peak_gpu_memory_mb"] for result in coco]
    bars = axes[0].bar(MODELS, fps, color=COLORS)
    add_labels(axes[0], bars, "{:.1f}")
    axes[0].set_ylabel("Pipeline FPS")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.25)
    bars = axes[1].bar(MODELS, memory, color=COLORS)
    add_labels(axes[1], bars, "{:.0f}")
    axes[1].set_ylabel("Peak allocated GPU memory (MB)")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    save(figure, "efficiency")

    grounding_paths = [
        OUTPUT / f"refcoco_owlvit_eval_{refcoco_tag}" / "metrics.json",
        OUTPUT / f"refcoco_grounding_dino_eval_{refcoco_tag}" / "metrics.json",
        OUTPUT / f"refcoco_yolo_world_eval_{refcoco_tag}" / "metrics.json",
    ]
    grounding = [load(path) for path in grounding_paths]
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(7.2, 3.7))
    acc50 = [result["accuracy_at_iou_0.5"] for result in grounding]
    acc75 = [result["accuracy_at_iou_0.75"] for result in grounding]
    bars1 = axis.bar(x - 0.18, acc50, 0.36, label="Acc@0.5", color="#4C78A8")
    bars2 = axis.bar(x + 0.18, acc75, 0.36, label="Acc@0.75", color="#E45756")
    add_labels(axis, bars1)
    add_labels(axis, bars2)
    axis.set_xticks(x, MODELS)
    axis.set_ylim(0, 0.72)
    axis.set_ylabel("Top-1 grounding accuracy")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save(figure, "refcoco_grounding")

    sensitivity_paths = [
        OUTPUT / "threshold_sensitivity" / "owlvit.json",
        OUTPUT / "threshold_sensitivity" / "grounding_dino.json",
        OUTPUT / "threshold_sensitivity" / "yolo_world.json",
    ]
    sensitivity = [load(path) for path in sensitivity_paths]
    figure, axes = plt.subplots(1, 3, figsize=(8.2, 2.8))
    for axis, name, result, color in zip(axes, MODELS, sensitivity, COLORS):
        thresholds = [row["threshold"] for row in result["results"]]
        ap = [row["metrics"]["AP@[IoU=.50:.95]"] for row in result["results"]]
        ar = [row["metrics"]["AR maxDets=100"] for row in result["results"]]
        axis.plot(thresholds, ap, "o-", color=color, label="AP")
        axis.plot(thresholds, ar, "s--", color="#777777", label="AR100")
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("Score threshold")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Metric")
    axes[-1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    save(figure, "threshold_sensitivity")


if __name__ == "__main__":
    main()

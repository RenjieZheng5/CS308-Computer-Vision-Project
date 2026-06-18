import random
import time

import torch


def select_image_ids(all_image_ids: list[int], max_images: int, sampling: str, seed: int) -> list[int]:
    image_ids = sorted(all_image_ids)
    if max_images <= 0 or max_images >= len(image_ids):
        return image_ids
    if sampling == "first":
        return image_ids[:max_images]
    return sorted(random.Random(seed).sample(image_ids, max_images))


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


class RuntimeTracker:
    def __init__(self, device: str) -> None:
        self.device = device
        self.preprocess_seconds = 0.0
        self.inference_seconds = 0.0
        self.postprocess_seconds = 0.0
        self.images = 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def measure(self, stage: str, function):
        synchronize(self.device)
        start = time.perf_counter()
        result = function()
        synchronize(self.device)
        elapsed = time.perf_counter() - start
        setattr(self, f"{stage}_seconds", getattr(self, f"{stage}_seconds") + elapsed)
        return result

    def add_image(self) -> None:
        self.images += 1

    def summary(self) -> dict:
        total = self.preprocess_seconds + self.inference_seconds + self.postprocess_seconds
        return {
            "images": self.images,
            "preprocess_seconds": self.preprocess_seconds,
            "inference_seconds": self.inference_seconds,
            "postprocess_seconds": self.postprocess_seconds,
            "pipeline_seconds": total,
            "inference_ms_per_image": 1000.0 * self.inference_seconds / max(self.images, 1),
            "pipeline_ms_per_image": 1000.0 * total / max(self.images, 1),
            "inference_fps": self.images / max(self.inference_seconds, 1e-9),
            "pipeline_fps": self.images / max(total, 1e-9),
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated() / (1024 * 1024) if self.device == "cuda" else 0.0
            ),
        }

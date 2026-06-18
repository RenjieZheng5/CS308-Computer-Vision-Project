import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pycocotools.coco import COCO
from tqdm import tqdm

from coco_eval_utils import select_image_ids
from evaluate_coco_owlvit import ensure_coco_annotations, ensure_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a deterministic COCO val2017 subset.")
    parser.add_argument("--data-dir", default="data/coco")
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--sampling", choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=308)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "val2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_file = ensure_coco_annotations(data_dir)
    coco = COCO(str(annotation_file))
    image_ids = select_image_ids(coco.getImgIds(), args.max_images, args.sampling, args.seed)
    image_infos = coco.loadImgs(image_ids)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(ensure_image, info, image_dir) for info in image_infos]
        for future in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
            future.result()

    manifest = {
        "dataset": "COCO val2017",
        "sampling": args.sampling,
        "seed": args.seed,
        "max_images": len(image_ids),
        "image_ids": image_ids,
    }
    manifest_path = data_dir / f"subset_{args.sampling}_{len(image_ids)}_seed_{args.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"downloaded: {len(image_ids)} images")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

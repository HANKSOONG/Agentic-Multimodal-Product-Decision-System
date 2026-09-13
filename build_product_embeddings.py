#!/usr/bin/env python3
"""Build weighted-concatenated DINOv2 image and BGE title embeddings.

The image and text models are loaded sequentially to keep peak VRAM low enough
for an 8 GB RTX 4060. Missing or corrupt images are skipped and recorded.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent if (ROOT.parent / "dataset").is_dir() else ROOT
DEFAULT_DATASET_DIR = WORKSPACE_ROOT / "dataset"
DEFAULT_IMAGE_DIR = WORKSPACE_ROOT / "image"
DEFAULT_METADATA_PATH = DEFAULT_DATASET_DIR / "meta_All_Beauty_2021_2023_users_5plus.jsonl"
DEFAULT_OUTPUT_PATH = (
    DEFAULT_DATASET_DIR / "All_Beauty_2021_2023_users_5plus_dinov2_bge_embeddings.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--image-model", default="facebook/dinov2-base")
    parser.add_argument("--text-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--image-weight", type=float, default=0.4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def load_metadata(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    malformed = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if row.get("parent_asin") and isinstance(row.get("title"), str):
                rows.append(row)
    return rows, malformed


def image_path_for(image_dir: Path, parent_asin: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        path = image_dir / f"{parent_asin}{suffix}"
        if path.is_file():
            return path
    return None


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.image_weight <= 1.0:
        raise ValueError("--image-weight must be between 0 and 1")
    if args.image_batch_size < 1 or args.text_batch_size < 1:
        raise ValueError("Batch sizes must be positive")

    device = resolve_device(args.device)
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True

    metadata_rows, malformed_rows = load_metadata(args.metadata)
    stats = {
        "total_metadata_rows": len(metadata_rows),
        "encoded": 0,
        "missing_image": 0,
        "missing_title": 0,
        "corrupt_image": 0,
        "malformed_metadata_rows": malformed_rows,
    }
    errors = []
    candidates = []
    for row_number, row in enumerate(metadata_rows, start=1):
        image_path = image_path_for(args.image_dir, str(row["parent_asin"]))
        if image_path is None:
            stats["missing_image"] += 1
        elif not row["title"].strip():
            stats["missing_title"] += 1
        else:
            candidates.append((row_number, row, image_path))
    if not candidates:
        raise RuntimeError("No products have both a valid title and a local image")

    print(f"device: {device}")
    print(f"products ready for encoding: {len(candidates)}")
    print(f"image/text batch sizes: {args.image_batch_size}/{args.text_batch_size}")

    image_processor = AutoImageProcessor.from_pretrained(args.image_model)
    image_model = AutoModel.from_pretrained(args.image_model).to(device).eval()
    parent_asins: list[str] = []
    encoded_rows: list[dict[str, Any]] = []
    image_chunks = []

    for start in tqdm(range(0, len(candidates), args.image_batch_size), desc="Encoding images"):
        batch = candidates[start : start + args.image_batch_size]
        images = []
        valid_rows = []
        for row_number, row, image_path in batch:
            try:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
                valid_rows.append(row)
            except Exception as exc:
                stats["corrupt_image"] += 1
                errors.append(
                    {
                        "row_number": row_number,
                        "parent_asin": row.get("parent_asin"),
                        "error": repr(exc),
                    }
                )
        if not images:
            continue

        inputs = image_processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            outputs = image_model(**inputs)
            vectors = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=-1)
        image_chunks.append(vectors.float().cpu())
        parent_asins.extend(str(row["parent_asin"]) for row in valid_rows)
        encoded_rows.extend(valid_rows)
        stats["encoded"] += len(valid_rows)

    if not image_chunks:
        raise RuntimeError("No images were encoded")
    image_vectors = torch.cat(image_chunks, dim=0)

    del image_model, image_processor, image_chunks
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_model = AutoModel.from_pretrained(args.text_model).to(device).eval()
    text_chunks = []
    for start in tqdm(range(0, len(encoded_rows), args.text_batch_size), desc="Encoding titles"):
        rows = encoded_rows[start : start + args.text_batch_size]
        titles = [row["title"].strip() for row in rows]
        inputs = text_tokenizer(
            titles,
            padding=True,
            truncation=True,
            max_length=args.text_max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp
        ):
            outputs = text_model(**inputs)
            vectors = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=-1)
        text_chunks.append(vectors.float().cpu())

    text_vectors = torch.cat(text_chunks, dim=0)
    if image_vectors.shape[0] != text_vectors.shape[0]:
        raise ValueError("Image and title embedding counts differ")

    text_weight = 1.0 - args.image_weight
    product_vectors = torch.cat(
        (
            math.sqrt(args.image_weight) * image_vectors,
            math.sqrt(text_weight) * text_vectors,
        ),
        dim=-1,
    )
    product_vectors = F.normalize(product_vectors, p=2, dim=-1)
    result = {
        "parent_asins": parent_asins,
        "image_vec": image_vectors,
        "text_vec": text_vectors,
        "product_vec": product_vectors,
        "image_model_name": args.image_model,
        "text_model_name": args.text_model,
        "image_embedding_dim": int(image_vectors.shape[-1]),
        "text_embedding_dim": int(text_vectors.shape[-1]),
        "embedding_dim": int(product_vectors.shape[-1]),
        "image_batch_size": args.image_batch_size,
        "text_batch_size": args.text_batch_size,
        "image_weight": args.image_weight,
        "text_weight": text_weight,
        "fusion": "l2_normalize_then_sqrt_weighted_concatenation",
        "stats": stats,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"image vectors: {tuple(image_vectors.shape)}")
    print(f"title vectors: {tuple(text_vectors.shape)}")
    print(f"product vectors: {tuple(product_vectors.shape)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

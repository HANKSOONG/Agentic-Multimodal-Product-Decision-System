#!/usr/bin/env python3
"""Build rating-weighted user vectors from multimodal product vectors.

The weight is rating - 3, so ratings 1 through 5 map to -2, -1, 0, 1,
and 2. Repeated interactions with one product are reduced to the latest row.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent if (ROOT.parent / "dataset").is_dir() else ROOT
DEFAULT_DATASET_DIR = WORKSPACE_ROOT / "dataset"
DEFAULT_REVIEW_PATH = DEFAULT_DATASET_DIR / "All_Beauty_2021_2023_users_5plus.jsonl"
DEFAULT_ITEM_PATH = (
    DEFAULT_DATASET_DIR / "All_Beauty_2021_2023_users_5plus_dinov2_bge_embeddings.pt"
)
DEFAULT_OUTPUT_PATH = (
    DEFAULT_DATASET_DIR / "All_Beauty_2021_2023_users_5plus_user_embeddings.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--item-embeddings", type=Path, default=DEFAULT_ITEM_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    item_data = torch.load(args.item_embeddings, map_location="cpu", weights_only=False)
    parent_asins = [str(value) for value in item_data["parent_asins"]]
    product_vectors = item_data["product_vec"].float().contiguous()
    item_model_info = {
        "image_model_name": item_data.get("image_model_name"),
        "text_model_name": item_data.get("text_model_name"),
        "image_weight": item_data.get("image_weight"),
        "text_weight": item_data.get("text_weight"),
        "fusion": item_data.get("fusion"),
    }
    del item_data
    gc.collect()

    if len(parent_asins) != product_vectors.shape[0]:
        raise ValueError("Product identifier and vector counts differ")
    item_to_index = {asin: index for index, asin in enumerate(parent_asins)}

    user_ids: list[str] = []
    user_to_index: dict[str, int] = {}
    latest_by_user_item: dict[tuple[str, str], tuple[int, float]] = {}
    stats = {
        "total_review_rows": 0,
        "malformed_rows": 0,
        "missing_required_fields": 0,
        "duplicate_user_item_rows": 0,
        "matched_interactions": 0,
        "neutral_interactions": 0,
        "missing_item_embeddings": 0,
    }

    with args.reviews.open("r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Reading reviews"):
            stats["total_review_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["malformed_rows"] += 1
                continue
            user_id = row.get("user_id")
            parent_asin = row.get("parent_asin")
            rating = row.get("rating")
            if not user_id or not parent_asin or not isinstance(rating, (int, float)):
                stats["missing_required_fields"] += 1
                continue

            user_id = str(user_id)
            parent_asin = str(parent_asin)
            if user_id not in user_to_index:
                user_to_index[user_id] = len(user_ids)
                user_ids.append(user_id)
            key = (user_id, parent_asin)
            timestamp = int(row.get("timestamp") or 0)
            previous = latest_by_user_item.get(key)
            if previous is not None:
                stats["duplicate_user_item_rows"] += 1
            if previous is None or timestamp >= previous[0]:
                latest_by_user_item[key] = (timestamp, float(rating))

    user_indices = []
    item_indices = []
    weights = []
    for (user_id, parent_asin), (_, rating) in latest_by_user_item.items():
        item_index = item_to_index.get(parent_asin)
        if item_index is None:
            stats["missing_item_embeddings"] += 1
            continue
        stats["matched_interactions"] += 1
        weight = rating - 3.0
        if weight == 0.0:
            stats["neutral_interactions"] += 1
            continue
        user_indices.append(user_to_index[user_id])
        item_indices.append(item_index)
        weights.append(weight)

    user_indices_tensor = torch.tensor(user_indices, dtype=torch.long)
    item_indices_tensor = torch.tensor(item_indices, dtype=torch.long)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    num_users = len(user_ids)
    embedding_dim = int(product_vectors.shape[-1])
    user_vectors = torch.zeros((num_users, embedding_dim), dtype=torch.float32)
    profile_strength = torch.zeros(num_users, dtype=torch.float32)
    non_neutral_counts = torch.zeros(num_users, dtype=torch.int32)

    for start in tqdm(range(0, len(weights_tensor), args.batch_size), desc="Aggregating profiles"):
        end = start + args.batch_size
        batch_users = user_indices_tensor[start:end]
        batch_items = item_indices_tensor[start:end]
        batch_weights = weights_tensor[start:end]
        user_vectors.index_add_(
            0,
            batch_users,
            product_vectors[batch_items] * batch_weights.unsqueeze(1),
        )
        profile_strength.index_add_(0, batch_users, batch_weights.abs())
        non_neutral_counts.index_add_(
            0,
            batch_users,
            torch.ones_like(batch_users, dtype=torch.int32),
        )

    profile_norms = torch.linalg.vector_norm(user_vectors, dim=-1)
    has_preference_signal = profile_norms > 1e-12
    user_vectors[has_preference_signal] /= profile_norms[has_preference_signal].unsqueeze(1)

    stats.update(
        {
            "unique_user_item_pairs": len(latest_by_user_item),
            "total_users": num_users,
            "users_with_preference_signal": int(has_preference_signal.sum()),
            "zero_vector_users": int((~has_preference_signal).sum()),
        }
    )
    result = {
        "user_ids": user_ids,
        "user_vec": user_vectors,
        "has_preference_signal": has_preference_signal,
        "profile_strength": profile_strength,
        "non_neutral_counts": non_neutral_counts,
        "weight_formula": "rating - 3",
        "embedding_dim": embedding_dim,
        "item_embedding_path": str(args.item_embeddings),
        "item_model_info": item_model_info,
        "stats": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(stats)
    print(f"user vectors: {tuple(user_vectors.shape)}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

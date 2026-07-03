#!/usr/bin/env python
import argparse
import json
import math
import os
from pathlib import Path

import mauve
import numpy as np
from transformers import AutoTokenizer

from src.datasets.preprocessed_dataset import load_preprocessed_dataset


def load_samples(path: str) -> list[str]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if "samples" in data:
            data = data["samples"]
        elif "generated_samples" in data:
            data = data["generated_samples"]
        else:
            data = list(data.values())

    samples = []
    for item in data:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("sample")
                or item.get("generated_text")
                or item.get("output")
            )
        else:
            text = str(item)
        if text and text.strip():
            samples.append(text)
    return samples


def decode_reference_rows(dataset, tokenizer, indices: list[int]) -> list[str]:
    texts = []
    for idx in indices:
        row = dataset[int(idx)]
        ids = row["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        if text.strip():
            texts.append(text)
    return texts


def build_or_load_reference_subsets(args, tokenizer):
    reference_path = Path(args.reference_subsets_json)
    if reference_path.exists():
        return json.loads(reference_path.read_text())

    dataset = load_preprocessed_dataset(args.dataset_path)
    dataset_size = len(dataset)
    if args.reference_size > dataset_size:
        raise ValueError(
            f"reference_size={args.reference_size} exceeds dataset size {dataset_size}"
        )

    rng = np.random.default_rng(args.reference_seed)
    subsets = []
    for subset_id in range(args.num_subsets):
        indices = rng.choice(dataset_size, size=args.reference_size, replace=False)
        indices = [int(x) for x in indices.tolist()]
        texts = decode_reference_rows(dataset, tokenizer, indices)
        if len(texts) != args.reference_size:
            raise ValueError(
                f"subset {subset_id} decoded to {len(texts)} non-empty texts, "
                f"expected {args.reference_size}"
            )
        subsets.append(
            {
                "subset_id": subset_id,
                "mauve_seed": int(args.mauve_seed + subset_id),
                "indices": indices,
                "texts": texts,
            }
        )

    payload = {
        "dataset_path": args.dataset_path,
        "dataset_size": int(dataset_size),
        "reference_seed": int(args.reference_seed),
        "mauve_seed": int(args.mauve_seed),
        "reference_size": int(args.reference_size),
        "num_subsets": int(args.num_subsets),
        "subsets": subsets,
    }
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "sample_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "count": int(len(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="label=path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--reference-subsets-json", required=True)
    parser.add_argument("--reference-size", type=int, default=1000)
    parser.add_argument("--num-subsets", type=int, default=5)
    parser.add_argument("--reference-seed", type=int, default=20260701)
    parser.add_argument("--mauve-seed", type=int, default=1234)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--max-generated", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    reference_payload = build_or_load_reference_subsets(args, tokenizer)
    subsets = reference_payload["subsets"]

    all_results = {}
    for spec in args.input:
        label, sample_path = spec.split("=", 1)
        samples = load_samples(sample_path)
        if args.max_generated is not None:
            samples = samples[: args.max_generated]
        if not samples:
            raise ValueError(f"No generated samples found for {label}: {sample_path}")

        per_subset = []
        for subset in subsets:
            refs = subset["texts"]
            n = min(len(refs), len(samples))
            out = mauve.compute_mauve(
                p_text=refs[:n],
                q_text=samples[:n],
                device_id=args.device_id,
                seed=int(subset["mauve_seed"]),
                verbose=args.verbose,
            )
            row = {
                "subset_id": int(subset["subset_id"]),
                "mauve_seed": int(subset["mauve_seed"]),
                "num_reference": int(n),
                "num_generated": int(n),
                "mauve": float(out.mauve),
            }
            per_subset.append(row)
            (output_dir / f"{label}_partial.json").write_text(
                json.dumps(per_subset, indent=2) + "\n"
            )

        scores = [row["mauve"] for row in per_subset]
        summary = {
            "label": label,
            "input_path": sample_path,
            "reference_subsets_json": args.reference_subsets_json,
            "reference_seed": reference_payload["reference_seed"],
            "mauve_seed_base": reference_payload["mauve_seed"],
            "reference_size": reference_payload["reference_size"],
            "num_subsets": reference_payload["num_subsets"],
            "num_generated_available": len(samples),
            "mauve_scores": per_subset,
            "mauve": mean_std(scores),
        }
        all_results[label] = summary
        (output_dir / f"{label}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    (output_dir / "summary.json").write_text(json.dumps(all_results, indent=2) + "\n")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run two-stage baseline inference on MC-Bench.

Two stages (both using VLM):
  Stage 1: Parse prompt → structured info
  Stage 2: VLM grounding → candidate bboxes
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import MCBenchDataset
from src.vlm_inference import Qwen2VLLM
from src.two_stage_agent import TwoStageBaselineAgent


def main():
    parser = argparse.ArgumentParser(description="Run two-stage baseline on MC-Bench")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--split", type=str, default="dev", help="Split to run on (dev/test)")
    parser.add_argument("--output", type=str, required=True, help="Output path for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Load dataset
    dataset = MCBenchDataset(
        config["dataset"]["json_path"],
        config["dataset"]["image_root"],
    )

    with open(config["splits"][args.split], 'r') as f:
        split_ids = json.load(f)

    samples = [dataset.get_by_id(sid) for sid in split_ids]
    if args.limit:
        samples = samples[:args.limit]

    print(f"Loaded {len(samples)} samples from {args.split} split")

    # Load VLM (shared by parser and grounder)
    vlm = Qwen2VLLM(
        model_name=config["vlm"]["model_name"],
        device=config["vlm"]["device"],
        cache_dir=config["vlm"].get("cache_dir"),
    )
    vlm.load()

    # Create agent
    agent = TwoStageBaselineAgent(vlm, config)

    # Run inference
    print("Running inference...")
    results = []
    for idx, sample in enumerate(samples):
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(samples)}")

        images = dataset.load_images(sample)

        try:
            prediction = agent.predict(sample, images)
            results.append(prediction)
        except Exception as e:
            print(f"Error on sample {sample['sample_id']}: {e}")
            results.append({
                "sample_id": sample["sample_id"],
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "image_id": -1,
                "confidence": 0.0,
                "method": "error",
            })

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Completed! Results saved to {args.output}")
    print(f"Total predictions: {len(results)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Inference script for MC-Bench visual grounding.
Run agent on dataset and generate predictions.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import MCBenchDataset
from src.agent import create_agent, BatchAgent


def main():
    parser = argparse.ArgumentParser(description="Run inference on MC-Bench")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--split", type=str, default="dev", help="Split to run on (dev/test)")
    parser.add_argument("--output", type=str, required=True, help="Output path for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    dataset_path = config["dataset"]["json_path"]
    image_root = config["dataset"]["image_root"]
    
    dataset = MCBenchDataset(dataset_path, image_root)
    
    split_ids_path = config["splits"][args.split]
    with open(split_ids_path, 'r') as f:
        split_ids = json.load(f)
    
    samples = [dataset.get_by_id(sid) for sid in split_ids]
    
    if args.limit:
        samples = samples[:args.limit]
    
    print(f"Loaded {len(samples)} samples from {args.split} split")
    
    agent = create_agent(config["agent"])
    batch_agent = BatchAgent(agent)
    
    print(f"Running inference with model: {config['agent']['model_name']}")
    
    results = batch_agent.predict_batch(
        samples=samples,
        dataset=dataset,
        output_path=args.output,
        verbose=args.verbose,
    )
    
    print(f"Completed! Results saved to {args.output}")
    print(f"Total predictions: {len(results)}")


if __name__ == "__main__":
    main()

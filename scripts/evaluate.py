#!/usr/bin/env python3
"""
Evaluation script for MC-Bench visual grounding.
Uses official MC-Bench evaluation metrics.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcbench.eval_mc_bench import mcbench_eval
from src.dataset import MCBenchDataset


def convert_to_mcbench_format(predictions: list, dataset: MCBenchDataset) -> list:
    """
    Convert agent predictions to MC-Bench format for evaluation.
    
    MC-Bench expects:
    - image_id: the image containing the target
    - category_id: object category
    - bbox: [x, y, w, h]
    - score: confidence score
    """
    results = []
    
    for pred in predictions:
        sample_id = pred["sample_id"]
        sample = dataset.get_by_id(sample_id)
        
        bbox = pred["bbox"]
        
        if bbox == [0, 0, 0, 0]:
            continue
        
        image_id = pred.get("image_id", sample["images"][0]["id"])
        
        category_id = 1
        if sample.get("annotations"):
            category_id = sample["annotations"][0].get("category_id", 1)
        
        results.append({
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": pred.get("confidence", 1.0),
        })
    
    return results


def evaluate_predictions(
    predictions_path: str,
    dataset_json: str,
    image_root: str,
    coco_format_path: str = None,
):
    """
    Evaluate predictions using MC-Bench official evaluation.
    """
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    dataset = MCBenchDataset(dataset_json, image_root)
    
    mcbench_results = convert_to_mcbench_format(predictions, dataset)
    
    print(f"Converted to MC-Bench format: {len(mcbench_results)} results")
    
    if coco_format_path:
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(mcbench_results, f)
            temp_path = f.name
        
        try:
            ap50, acc = mcbench_eval(coco_format_path, temp_path, 'all')
            
            print("\n=== Evaluation Results ===")
            if ap50 is not None:
                print(f"Weighted AP@0.5: {ap50:.4f}")
            if acc is not None:
                print(f"Accuracy: {acc:.4f}")
            
            return {"ap50": ap50, "accuracy": acc}
        finally:
            os.unlink(temp_path)
    
    return mcbench_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MC-Bench predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--dataset-json", type=str, required=True, help="Path to dataset JSON")
    parser.add_argument("--image-root", type=str, required=True, help="Path to images directory")
    parser.add_argument("--coco-format", type=str, help="Path to COCO format ground truth")
    args = parser.parse_args()
    
    evaluate_predictions(
        predictions_path=args.predictions,
        dataset_json=args.dataset_json,
        image_root=args.image_root,
        coco_format_path=args.coco_format,
    )


if __name__ == "__main__":
    main()

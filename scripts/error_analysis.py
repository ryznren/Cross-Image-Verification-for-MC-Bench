#!/usr/bin/env python3
"""
Error analysis for MC-Bench predictions.

Breaks down AP@0.5 and Accuracy by:
  - text_style (Referring / Comparison / Reasoning)
  - positive_sample (positive / negative)
  - target_position (first / second / both / none)

Also compares multiple prediction files side-by-side.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_dataset_meta(json_path: str) -> dict:
    """Build per-sample metadata: text_style, positive_sample, target_position."""
    with open(json_path) as f:
        data = json.load(f)

    # image_id → inter_img_id (0=first, 1=second)
    img_inter = {img["id"]: img["inter_img_id"] for img in data["images"]}
    # image_id → text_id
    img_text = {img["id"]: img["text_id"] for img in data["images"]}

    # text_id → annotation image_ids
    ann_by_text = defaultdict(set)
    for ann in data["annotations"]:
        ann_by_text[ann["text_id"]].add(ann["image_id"])

    meta = {}
    for desc in data["descriptions"]:
        tid = desc["id"]
        ann_img_ids = ann_by_text[tid]
        positions = {img_inter[iid] for iid in ann_img_ids if iid in img_inter}

        if len(positions) == 0:
            target_pos = "none"
        elif positions == {0}:
            target_pos = "first"
        elif positions == {1}:
            target_pos = "second"
        else:
            target_pos = "both"

        meta[tid] = {
            "text_style": desc["text_style"],
            "positive_sample": desc["positive_sample"],
            "target_position": target_pos,
        }
    return meta


def compute_iou(pred_bbox, gt_bbox):
    """Compute IoU between two [x,y,w,h] boxes."""
    px, py, pw, ph = pred_bbox
    gx, gy, gw, gh = gt_bbox

    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = pw * ph + gw * gh - inter
    return inter / union if union > 0 else 0.0


def load_predictions(pred_path: str) -> dict:
    """Load predictions keyed by sample_id."""
    with open(pred_path) as f:
        preds = json.load(f)
    return {p["sample_id"]: p for p in preds}


def load_gt_by_text(json_path: str) -> dict:
    """Load GT annotations keyed by text_id."""
    with open(json_path) as f:
        data = json.load(f)
    gt = defaultdict(list)
    for ann in data["annotations"]:
        gt[ann["text_id"]].append(ann)
    return gt


def evaluate_subset(pred_dict, gt_by_text, sample_ids, iou_thresh=0.5):
    """
    Compute per-sample hit/miss and aggregate AP@0.5 (precision-recall style)
    and Accuracy for a subset of sample_ids.

    Returns dict with ap, accuracy, n_samples, n_correct, n_hit.
    """
    hits = []
    correct_image = []

    for sid in sample_ids:
        pred = pred_dict.get(sid)
        gts = gt_by_text.get(sid, [])

        if not gts:
            # Negative sample: correct if pred bbox is [0,0,0,0] or image_id=-1
            if pred is None or pred["image_id"] == -1 or pred["bbox"] == [0.0, 0.0, 0.0, 0.0]:
                hits.append(1)
                correct_image.append(1)
            else:
                hits.append(0)
                correct_image.append(0)
            continue

        if pred is None or pred["image_id"] == -1:
            hits.append(0)
            correct_image.append(0)
            continue

        # Check if predicted image_id matches any GT image_id
        gt_image_ids = {g["image_id"] for g in gts}
        img_correct = pred["image_id"] in gt_image_ids
        correct_image.append(int(img_correct))

        # Check IoU against GT boxes in the predicted image
        gt_in_pred_img = [g for g in gts if g["image_id"] == pred["image_id"]]
        if not gt_in_pred_img:
            hits.append(0)
            continue

        best_iou = max(compute_iou(pred["bbox"], g["bbox"]) for g in gt_in_pred_img)
        hits.append(int(best_iou >= iou_thresh))

    n = len(sample_ids)
    if n == 0:
        return {"ap": 0.0, "accuracy": 0.0, "n": 0, "n_hit": 0, "n_correct_img": 0}

    return {
        "ap": sum(hits) / n,
        "accuracy": sum(correct_image) / n,
        "n": n,
        "n_hit": sum(hits),
        "n_correct_img": sum(correct_image),
    }


def print_breakdown(label, groups, pred_dict, gt_by_text, meta, dev_ids):
    """Print breakdown table for a grouping dimension."""
    print(f"\n--- By {label} ---")
    print(f"{'Category':<20} {'N':>5} {'AP@0.5':>8} {'Accuracy':>10} {'ImgAcc':>8}")
    print("-" * 55)

    totals = defaultdict(list)
    for group_val in sorted(groups.keys()):
        ids = [sid for sid in dev_ids if meta.get(sid, {}).get(label.lower().replace(" ", "_")) == group_val]
        res = evaluate_subset(pred_dict, gt_by_text, ids)
        print(f"{str(group_val):<20} {res['n']:>5} {res['ap']:>8.4f} {res['accuracy']:>10.4f} {res['n_correct_img']/max(res['n'],1):>8.4f}")


def compare_methods(method_results: list, gt_by_text, meta, dev_ids, dimension, key):
    """Print side-by-side comparison across methods for a dimension."""
    groups = sorted({meta.get(sid, {}).get(key, "?") for sid in dev_ids})

    header = f"{'Category':<20}"
    for name, _ in method_results:
        header += f"  {name[:12]:>12}"
    print(header)
    print("-" * (20 + 14 * len(method_results)))

    for g in groups:
        ids = [sid for sid in dev_ids if meta.get(sid, {}).get(key) == g]
        row = f"{str(g):<20}"
        for name, pred_dict in method_results:
            res = evaluate_subset(pred_dict, gt_by_text, ids)
            row += f"  {res['ap']:>12.4f}"
        print(row)

    # Overall
    row = f"{'Overall':<20}"
    for name, pred_dict in method_results:
        res = evaluate_subset(pred_dict, gt_by_text, dev_ids)
        row += f"  {res['ap']:>12.4f}"
    print(row)


def main():
    parser = argparse.ArgumentParser(description="Error analysis for MC-Bench predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Primary prediction JSON")
    parser.add_argument("--compare", type=str, nargs="*", default=[], help="Additional prediction JSONs to compare")
    parser.add_argument("--dataset-json", type=str, default="mcbench/mc-bench_v0.2_val.json")
    parser.add_argument("--split", type=str, default="experiments/splits/dev_ids.json")
    args = parser.parse_args()

    # Load data
    with open(args.split) as f:
        dev_ids = json.load(f)

    meta = load_dataset_meta(args.dataset_json)
    gt_by_text = load_gt_by_text(args.dataset_json)
    pred_dict = load_predictions(args.predictions)

    pred_name = Path(args.predictions).stem

    print(f"\n{'='*60}")
    print(f"Error Analysis: {pred_name}")
    print(f"Dev samples: {len(dev_ids)}")
    print(f"{'='*60}")

    # Overall
    overall = evaluate_subset(pred_dict, gt_by_text, dev_ids)
    print(f"\nOverall  AP@0.5: {overall['ap']:.4f}  Accuracy: {overall['accuracy']:.4f}")

    # Breakdown by text_style
    styles = defaultdict(list)
    for sid in dev_ids:
        styles[meta.get(sid, {}).get("text_style", "?")].append(sid)

    print(f"\n--- By text_style ---")
    print(f"{'Style':<20} {'N':>5} {'AP@0.5':>8} {'Accuracy':>10}")
    print("-" * 47)
    for style in sorted(styles.keys()):
        ids = styles[style]
        res = evaluate_subset(pred_dict, gt_by_text, ids)
        print(f"{style:<20} {res['n']:>5} {res['ap']:>8.4f} {res['accuracy']:>10.4f}")

    # Breakdown by positive/negative
    print(f"\n--- By positive_sample ---")
    print(f"{'Type':<20} {'N':>5} {'AP@0.5':>8} {'Accuracy':>10}")
    print("-" * 47)
    for pos_val in [True, False]:
        ids = [sid for sid in dev_ids if meta.get(sid, {}).get("positive_sample") == pos_val]
        res = evaluate_subset(pred_dict, gt_by_text, ids)
        label = "Positive" if pos_val else "Negative"
        print(f"{label:<20} {res['n']:>5} {res['ap']:>8.4f} {res['accuracy']:>10.4f}")

    # Breakdown by target_position
    print(f"\n--- By target_position ---")
    print(f"{'Position':<20} {'N':>5} {'AP@0.5':>8} {'Accuracy':>10}")
    print("-" * 47)
    positions = defaultdict(list)
    for sid in dev_ids:
        positions[meta.get(sid, {}).get("target_position", "?")].append(sid)
    for pos in ["first", "second", "both", "none"]:
        ids = positions.get(pos, [])
        if not ids:
            continue
        res = evaluate_subset(pred_dict, gt_by_text, ids)
        print(f"{pos:<20} {res['n']:>5} {res['ap']:>8.4f} {res['accuracy']:>10.4f}")

    # Multi-method comparison
    if args.compare:
        compare_preds = [(Path(p).stem, load_predictions(p)) for p in args.compare]
        all_methods = [(pred_name, pred_dict)] + compare_preds

        print(f"\n{'='*60}")
        print("Method Comparison (AP@0.5)")
        print(f"{'='*60}")

        print("\n--- By text_style ---")
        compare_methods(all_methods, gt_by_text, meta, dev_ids, "text_style", "text_style")

        print("\n--- By positive_sample ---")
        groups_pos = sorted({meta.get(sid, {}).get("positive_sample", "?") for sid in dev_ids})
        header = f"{'Type':<20}"
        for name, _ in all_methods:
            header += f"  {name[:12]:>12}"
        print(header)
        print("-" * (20 + 14 * len(all_methods)))
        for g in [True, False]:
            ids = [sid for sid in dev_ids if meta.get(sid, {}).get("positive_sample") == g]
            row = f"{'Positive' if g else 'Negative':<20}"
            for name, pd in all_methods:
                res = evaluate_subset(pd, gt_by_text, ids)
                row += f"  {res['ap']:>12.4f}"
            print(row)
        row = f"{'Overall':<20}"
        for name, pd in all_methods:
            res = evaluate_subset(pd, gt_by_text, dev_ids)
            row += f"  {res['ap']:>12.4f}"
        print(row)


if __name__ == "__main__":
    main()

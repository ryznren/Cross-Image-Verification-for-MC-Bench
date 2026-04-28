#!/usr/bin/env python3
"""
Visualize qualitative cases for paper.

Finds samples where CIV (C) succeeds but two_stage (A) fails,
categorized by text_style. Saves side-by-side images with
GT boxes (green) and predictions (red=two_stage, blue=CIV).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import MCBenchDataset
from scripts.error_analysis import compute_iou, load_gt_by_text


def load_predictions(path):
    with open(path) as f:
        preds = json.load(f)
    return {p["sample_id"]: p for p in preds}


def is_hit(pred, gts, iou_thresh=0.5):
    """Check if prediction is a hit (correct image + IoU >= thresh)."""
    if not gts:
        # Negative sample: correct if no box predicted
        return pred is None or pred["image_id"] == -1 or pred["bbox"] == [0.0, 0.0, 0.0, 0.0]
    if pred is None or pred["image_id"] == -1:
        return False
    gt_in_pred_img = [g for g in gts if g["image_id"] == pred["image_id"]]
    if not gt_in_pred_img:
        return False
    best_iou = max(compute_iou(pred["bbox"], g["bbox"]) for g in gt_in_pred_img)
    return best_iou >= iou_thresh


def draw_bbox(draw, bbox, color, label=None, width=3):
    """Draw [x,y,w,h] bbox on image."""
    x, y, w, h = [int(v) for v in bbox]
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    if label:
        draw.text((x + 2, y + 2), label, fill=color)


def make_case_image(sample, images, gt_anns, pred_a, pred_b, pred_c, out_path):
    """
    Create a side-by-side visualization:
      Left panel: Image 1 | Right panel: Image 2
      GT: green, Two-stage (A): red, CIV-Local (B): orange, CIV-Full (C): blue
    """
    img1, img2 = images[0].copy(), images[1].copy()

    # Map image_id → img index
    img_id_to_idx = {sample["images"][i]["id"]: i for i in range(len(sample["images"]))}

    draws = [ImageDraw.Draw(img1), ImageDraw.Draw(img2)]

    # Draw GT (green)
    for ann in gt_anns:
        idx = img_id_to_idx.get(ann["image_id"])
        if idx is not None:
            draw_bbox(draws[idx], ann["bbox"], color="green", label="GT", width=3)

    # Draw Two-stage A (red)
    for pred, color, label in [(pred_a, "red", "A"), (pred_b, "orange", "B"), (pred_c, "blue", "C")]:
        if pred and pred["image_id"] != -1 and pred["bbox"] != [0.0, 0.0, 0.0, 0.0]:
            idx = img_id_to_idx.get(pred["image_id"])
            if idx is not None:
                draw_bbox(draws[idx], pred["bbox"], color=color, label=label, width=2)

    # Combine side by side with padding
    pad = 10
    h1, w1 = img1.height, img1.width
    h2, w2 = img2.height, img2.width
    max_h = max(h1, h2) + 60  # extra for caption
    total_w = w1 + w2 + pad * 3

    canvas = Image.new("RGB", (total_w, max_h), color=(240, 240, 240))
    canvas.paste(img1, (pad, pad))
    canvas.paste(img2, (w1 + pad * 2, pad))

    # Caption
    d = ImageDraw.Draw(canvas)
    text = sample["text"]
    if len(text) > 90:
        text = text[:87] + "..."
    caption = f"[{sample['text_style']}] {text}  |  GT:green  A:red  B:orange  C:blue"
    d.text((pad, max_h - 55), caption, fill=(50, 50, 50))

    canvas.save(out_path)

def find_interesting_cases(pred_a, pred_b, pred_c, gt_by_text, test_ids, meta, n_per_style=3):
    """
    Find cases where CIV (C) succeeds but two_stage (A) fails,
    grouped by text_style.
    """
    by_style = defaultdict(list)

    for sid in test_ids:
        gts = gt_by_text.get(sid, [])
        pa = pred_a.get(sid)
        pb = pred_b.get(sid)
        pc = pred_c.get(sid)

        hit_a = is_hit(pa, gts)
        hit_c = is_hit(pc, gts)

        # CIV wins, baseline fails
        if hit_c and not hit_a:
            style = meta.get(sid, {}).get("text_style", "?")
            by_style[style].append(sid)

    # Also find failure cases: everything fails (interesting for error analysis)
    failures = []
    for sid in test_ids:
        gts = gt_by_text.get(sid, [])
        if not gts:
            continue
        pc = pred_c.get(sid)
        if not is_hit(pc, gts):
            style = meta.get(sid, {}).get("text_style", "?")
            failures.append((style, sid))

    return by_style, failures


def main():
    parser = argparse.ArgumentParser(description="Visualize qualitative cases")
    parser.add_argument("--pred-a", default="experiments/results/two_stage_test_full.json")
    parser.add_argument("--pred-b", default="experiments/results/civ_local_only_test_full.json")
    parser.add_argument("--pred-c", default="experiments/results/civ_agent_test_full.json")
    parser.add_argument("--dataset-json", default="mcbench/mc-bench_v0.2_val.json")
    parser.add_argument("--image-root", default="mcbench/MC-Bench_images")
    parser.add_argument("--split", default="experiments/splits/test_ids.json")
    parser.add_argument("--output-dir", default="experiments/qualitative")
    parser.add_argument("--n-per-style", type=int, default=3)
    args = parser.parse_args()

    with open(args.split) as f:
        test_ids = json.load(f)

    dataset = MCBenchDataset(args.dataset_json, args.image_root)
    gt_by_text = load_gt_by_text(args.dataset_json)
    pred_a = load_predictions(args.pred_a)
    pred_b = load_predictions(args.pred_b)
    pred_c = load_predictions(args.pred_c)

    # Load metadata
    with open(args.dataset_json) as f:
        raw = json.load(f)
    meta = {}
    for desc in raw["descriptions"]:
        meta[desc["id"]] = {
            "text_style": desc["text_style"],
            "positive_sample": desc["positive_sample"],
        }

    by_style, failures = find_interesting_cases(
        pred_a, pred_b, pred_c, gt_by_text, test_ids, meta, args.n_per_style
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    print("\n=== CIV wins, baseline fails ===")
    for style, sids in by_style.items():
        print(f"  {style}: {len(sids)} cases")

    # Save visualizations: success cases
    saved = 0
    for style, sids in by_style.items():
        for sid in sids[:args.n_per_style]:
            sample = dataset.get_by_id(sid)
            images = dataset.load_images(sample)
            gts = gt_by_text.get(sid, [])
            out_path = out_dir / f"success_{style.lower()}_{sid}.png"
            make_case_image(
                sample, images, gts,
                pred_a.get(sid), pred_b.get(sid), pred_c.get(sid),
                out_path
            )
            print(f"Saved: {out_path}")
            print(f"  Text: {sample['text']}")
            saved += 1

    # Save failure cases (2 per style)
    fail_by_style = defaultdict(list)
    for style, sid in failures:
        fail_by_style[style].append(sid)

    for style, sids in fail_by_style.items():
        for sid in sids[:2]:
            sample = dataset.get_by_id(sid)
            images = dataset.load_images(sample)
            gts = gt_by_text.get(sid, [])
            out_path = out_dir / f"failure_{style.lower()}_{sid}.png"
            make_case_image(
                sample, images, gts,
                pred_a.get(sid), pred_b.get(sid), pred_c.get(sid),
                out_path
            )
            print(f"Saved (failure): {out_path}")
            saved += 1

    print(f"\nTotal saved: {saved} images → {out_dir}")


if __name__ == "__main__":
    main()


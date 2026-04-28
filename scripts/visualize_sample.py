"""
Visualize MC-Bench samples
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw_boxes(image: Image.Image, boxes: list[dict], color: str = "red") -> Image.Image:
    """Draw bounding boxes on image."""
    draw = ImageDraw.Draw(image)
    
    for box_info in boxes:
        bbox = box_info['bbox']  # [x, y, w, h]
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    
    return image


def visualize_sample(
    json_path: str,
    image_root: str,
    sample_id: int,
    output_path: str | None = None
):
    """Visualize a single sample with ground truth boxes."""
    
    from dataset import MCBenchDataset
    
    dataset = MCBenchDataset(json_path, image_root)
    sample = dataset[sample_id]
    
    print(f"Sample {sample_id}:")
    print(f"  Text: {sample['text']}")
    print(f"  Style: {sample['text_style']}")
    print(f"  Positive: {sample['positive_sample']}")
    print(f"  Target positions: {sample['target_positions']}")
    
    # Load images
    images = dataset.load_images(sample)
    
    # Draw GT boxes
    for i, img in enumerate(images):
        img_anns = [a for a in sample['annotations'] if a['image_id'] == img.info['id']]
        if img_anns:
            draw_boxes(img, img_anns, color="green")
            print(f"  Image {i}: {img.info['file_name']} - {len(img_anns)} GT box(es)")
        else:
            print(f"  Image {i}: {img.info['file_name']} - no GT box")
    
    # Combine images side by side
    combined = Image.new('RGB', (images[0].width * 2 + 10, images[0].height))
    combined.paste(images[0], (0, 0))
    combined.paste(images[1], (images[0].width + 10, 0))
    
    # Add text
    draw = ImageDraw.Draw(combined)
    draw.text((10, 10), f"Image 0", fill="white")
    draw.text((images[0].width + 20, 10), f"Image 1", fill="white")
    draw.text((10, images[0].height - 30), sample['text'][:80], fill="yellow")
    
    if output_path:
        combined.save(output_path)
        print(f"\nSaved to: {output_path}")
    
    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='/Users/ryan/Project/MC-Bench/mcbench/mc-bench_v0.2_val.json')
    parser.add_argument('--image_root', default='/Users/ryan/Project/MC-Bench/mcbench/MC-Bench_images/')
    parser.add_argument('--sample_id', type=int, default=0)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    visualize_sample(args.json, args.image_root, args.sample_id, args.output)

"""
MC-Bench Dataset Loader
"""
import json
from pathlib import Path
from typing import Any, TypedDict

from PIL import Image


class MCBenchSample(TypedDict):
    """Single sample from MC-Bench."""
    sample_id: int
    text: str
    positive_sample: bool
    text_style: str  # "Referring" | "Comparison" | "Reasoning"
    images: list[dict]  # List of 2 images
    annotations: list[dict]  # Ground truth boxes


class MCBenchDataset:
    def __init__(self, json_path: str, image_root: str):
        self.json_path = json_path
        self.image_root = Path(image_root)
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.descriptions = self.data['descriptions']
        self.images = {img['id']: img for img in self.data['images']}
        self.annotations = self.data['annotations']
        self.categories = self.data['categories']
        
        self._build_index()
    
    def _build_index(self):
        """Build index from text_id to sample."""
        self.samples = {}
        for desc in self.descriptions:
            text_id = desc['id']
            image_ids = desc['images_id']
            
            # Get images
            images = [self.images[iid] for iid in image_ids]
            
            # Get annotations for this text
            anns = [ann for ann in self.annotations if ann['text_id'] == text_id]
            
            # Determine target position
            target_positions = []
            for img in images:
                img_anns = [a for a in anns if a['image_id'] == img['id']]
                if len(img_anns) > 0:
                    target_positions.append(img['inter_img_id'])
            
            self.samples[text_id] = {
                'sample_id': text_id,
                'text': desc['text'],
                'positive_sample': desc['positive_sample'],
                'text_style': desc['text_style'],
                'images': images,
                'annotations': anns,
                'target_positions': target_positions,
            }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> MCBenchSample:
        return self.samples[idx]
    
    def get_by_id(self, sample_id: int) -> MCBenchSample:
        return self.samples[sample_id]
    
    def load_images(self, sample: MCBenchSample) -> list[Image.Image]:
        """Load PIL Images for a sample."""
        images = []
        for img_info in sample['images']:
            img_path = self.image_root / img_info['file_name']
            img = Image.open(img_path).convert('RGB')
            img.info = {'id': img_info['id'], 'file_name': img_info['file_name']}
            images.append(img)
        return images
    
    def get_image_paths(self, sample: MCBenchSample) -> list[Path]:
        """Get image paths for a sample."""
        return [self.image_root / img_info['file_name'] for img_info in sample['images']]
    
    def get_text_styles(self) -> dict[str, int]:
        """Count samples by text style."""
        styles = {}
        for desc in self.descriptions:
            style = desc['text_style']
            styles[style] = styles.get(style, 0) + 1
        return styles
    
    def get_positive_negative_counts(self) -> tuple[int, int]:
        """Get positive and negative sample counts."""
        positive = sum(1 for d in self.descriptions if d['positive_sample'])
        negative = len(self.descriptions) - positive
        return positive, negative
    
    def get_target_position_counts(self) -> dict[str, int]:
        """Count by target position: first only, second only, both, none."""
        counts = {'first': 0, 'second': 0, 'both': 0, 'none': 0}
        
        for sample in self.samples.values():
            positions = set(sample['target_positions'])
            if len(positions) == 0:
                counts['none'] += 1
            elif len(positions) == 1:
                if 0 in positions:
                    counts['first'] += 1
                else:
                    counts['second'] += 1
            else:
                counts['both'] += 1
        
        return counts
    
    def get_categories(self) -> list[dict]:
        """Get category list."""
        return self.categories


def create_result_dict(
    image_id: int,
    category_id: int,
    bbox: list[float],
    score: float = 1.0
) -> dict[str, Any]:
    """Create result entry in MC-Bench format."""
    return {
        'image_id': image_id,
        'category_id': category_id,
        'bbox': bbox,  # [x, y, w, h]
        'score': score,
    }


def save_results(results: list[dict], output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    import sys
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else 'mc-bench_v0.2_val.json'
    image_root = sys.argv[2] if len(sys.argv) > 2 else 'MC-Bench_images'
    
    dataset = MCBenchDataset(json_path, image_root)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Text styles: {dataset.get_text_styles()}")
    print(f"Positive/Negative: {dataset.get_positive_negative_counts()}")
    print(f"Target positions: {dataset.get_target_position_counts()}")
    print(f"Categories: {len(dataset.categories)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Text: {sample['text']}")
    print(f"  Style: {sample['text_style']}")
    print(f"  Positive: {sample['positive_sample']}")
    print(f"  Images: {[img['file_name'] for img in sample['images']]}")
    print(f"  Annotations: {len(sample['annotations'])}")

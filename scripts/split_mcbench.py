"""
Split MC-Bench into dev and test sets with stratified sampling.
"""
import json
import random
from pathlib import Path
from collections import defaultdict


def stratified_split(
    json_path: str,
    output_dir: str,
    dev_size: int = 300,
    seed: int = 42
):
    """Split dataset into dev and test with stratified sampling."""
    
    random.seed(seed)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Build sample index with stratification keys
    samples = []
    for desc in data['descriptions']:
        text_id = desc['id']
        
        # Determine target position
        image_ids = desc['images_id']
        target_positions = []
        
        for img_info in data['images']:
            if img_info['id'] in image_ids:
                anns = [a for a in data['annotations'] if a['text_id'] == text_id and a['image_id'] == img_info['id']]
                if anns:
                    target_positions.append(img_info['inter_img_id'])
        
        if len(target_positions) == 0:
            position = 'none'
        elif len(target_positions) == 1:
            position = 'first' if target_positions[0] == 0 else 'second'
        else:
            position = 'both'
        
        samples.append({
            'id': text_id,
            'text_style': desc['text_style'],
            'positive_sample': 'positive' if desc['positive_sample'] else 'negative',
            'position': position,
        })
    
    # Group by stratification keys
    groups = defaultdict(list)
    for s in samples:
        key = (s['text_style'], s['positive_sample'], s['position'])
        groups[key].append(s['id'])
    
    # Print distribution
    print("Stratification distribution:")
    for key, ids in sorted(groups.items()):
        print(f"  {key}: {len(ids)}")
    
    # Proportional sampling for dev set
    dev_ids = []
    test_ids = []
    
    for key, ids in groups.items():
        random.shuffle(ids)
        n_dev = max(1, int(len(ids) * dev_size / 2000))  # Proportional
        n_dev = min(n_dev, len(ids) - 1)  # Leave at least 1 for test
        
        dev_ids.extend(ids[:n_dev])
        test_ids.extend(ids[n_dev:])
    
    # Ensure no overlap
    dev_set = set(dev_ids)
    test_set = set(test_ids)
    overlap = dev_set & test_set
    if overlap:
        print(f"Warning: {len(overlap)} overlapping IDs, fixing...")
        # Move overlap to test
        for oid in overlap:
            dev_ids.remove(oid)
            test_ids.append(oid)
    
    print(f"\nSplit result:")
    print(f"  Dev: {len(dev_ids)}")
    print(f"  Test: {len(test_ids)}")
    
    # Verify distribution
    for split_name, split_ids in [('dev', dev_ids), ('test', test_ids)]:
        text_styles = defaultdict(int)
        positives = defaultdict(int)
        positions = defaultdict(int)
        
        for s in samples:
            if s['id'] in split_ids:
                text_styles[s['text_style']] += 1
                positives[s['positive_sample']] += 1
                positions[s['position']] += 1
        
        print(f"\n{split_name.upper()} distribution:")
        print(f"  Text styles: {dict(text_styles)}")
        print(f"  Positive/Negative: {dict(positives)}")
        print(f"  Positions: {dict(positions)}")
    
    # Save split files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'dev_ids.json', 'w') as f:
        json.dump(sorted(dev_ids), f, indent=2)
    
    with open(output_path / 'test_ids.json', 'w') as f:
        json.dump(sorted(test_ids), f, indent=2)
    
    print(f"\nSaved to {output_path}/")


if __name__ == '__main__':
    import sys
    
    json_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/ryan/Project/MC-Bench/mcbench/mc-bench_v0.2_val.json'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/Users/ryan/Project/MC-Bench/experiments/splits'
    dev_size = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    
    stratified_split(json_path, output_dir, dev_size)

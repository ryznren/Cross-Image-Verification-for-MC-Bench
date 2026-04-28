# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing a **Cross-Image Verification Agent** for multi-context visual grounding on the MC-Bench benchmark. The core contribution is a two-stage agent that:

1. Performs initial visual grounding across image pairs
2. Uses cross-image verification to reduce false positives

MC-Bench is a benchmark with 2,000 image pairs and open-ended text descriptions requiring instance-level localization across multiple images. Text prompts fall into three categories: Referring, Comparison, and Reasoning.

## Architecture

### Core Components

**Agent Pipeline** (`src/agent.py`):
- `CrossImageVerificationAgent`: Main agent with two modes
  - Direct mode: Single-pass prediction
  - Verification mode: Iterative refinement with cross-image verification
- `BatchAgent`: Wrapper for batch processing with progress tracking

**Dataset** (`src/dataset.py`):
- `MCBenchDataset`: Loads MC-Bench annotations and images
- Indexes samples by text_id with metadata (text_style, positive_sample, target_positions)
- Provides utilities for stratification analysis

**VLM Inference** (`src/vlm_inference.py`):
- Wrapper for Qwen2.5-VL model
- Handles bbox prediction from multi-image input

### Data Flow

```
Sample → Load Images → Agent.predict() → VLM Inference → Bbox Prediction
                                ↓
                        Verification Loop (optional)
                                ↓
                        Convert to MC-Bench format → Evaluation
```

### Configuration

注意：整个项目使用的是cv这个conda环境，不要一些相关的包安装在了base环境中！

Experiments are configured via JSON files in `configs/`:
- `baseline_qwen.json`: Direct prediction without verification
- `verification_agent.json`: With cross-image verification enabled

Key config parameters:
- `use_verification`: Enable/disable verification loop
- `verification_iterations`: Number of refinement iterations
- `confidence_threshold`: Threshold for early stopping

## Development Commands

### Setup

```bash
# Install dependencies
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install scipy torch transformers pillow

# Download MC-Bench dataset from Google Drive or Baidu Pan (see mcbench/README.md)
# Extract to mcbench/MC-Bench_images/
```

### Data Preparation

```bash
# Split dataset into dev/test with stratified sampling
python scripts/split_mcbench.py \
  mcbench/mc-bench_v0.2_val.json \
  experiments/splits \
  300  # dev set size

# Visualize a sample
python scripts/visualize_sample.py \
  --json mcbench/mc-bench_v0.2_val.json \
  --image-root mcbench/MC-Bench_images \
  --sample-id 0
```

### Running Experiments

```bash
# Run inference with a config
python scripts/inference.py --config configs/baseline_qwen.json

# Run inference on specific split
python scripts/inference.py \
  --config configs/verification_agent.json \
  --split dev
```

### Evaluation

```bash
# Evaluate predictions using official MC-Bench metrics
python scripts/evaluate.py \
  --predictions experiments/results/predictions.json \
  --dataset-json mcbench/mc-bench_v0.2_val.json \
  --image-root mcbench/MC-Bench_images \
  --coco-format mcbench/MC-Bench_coco_format.json

# Or use the official evaluation script directly
python mcbench/eval_mc_bench.py \
  --gt_json_path mcbench/MC-Bench_coco_format.json \
  --dt_json_path experiments/results/predictions.json \
  --eval_type all  # 'instance', 'image', or 'all'
```

## Key Implementation Details

### Bbox Format

All bounding boxes use **normalized coordinates** [x, y, width, height] where values are in range [0, 1]:
- (x, y): top-left corner
- (width, height): box dimensions
- [0, 0, 0, 0]: indicates no target found

### Target Image Determination

The agent determines which image contains the target using:
1. `target_positions` from dataset (ground truth during development)
2. Text analysis (e.g., "first image", "second image")
3. Fallback to first image if ambiguous

### Verification Strategy

The verification loop:
1. Makes initial prediction on both images
2. Prompts VLM to verify/refine the bbox considering cross-image context
3. Iterates until confidence threshold met or max iterations reached
4. Returns [0, 0, 0, 0] if verification determines no valid target

### Stratified Splitting

The dataset split maintains proportional distribution across:
- Text style: Referring / Comparison / Reasoning
- Sample type: Positive / Negative
- Target position: first / second / both / none

This ensures dev/test sets are representative for ablation studies.

## Research Context

**Problem**: Multi-image visual grounding suffers from cross-image false positives where models confuse similar objects across images.

**Solution**: Explicit verification step that re-evaluates candidates in full multi-image context.

**Baselines**:
- End-to-end VLM (Qwen2.5-VL direct prediction)
- Two-stage without verification (parse + ground)
- Two-stage with verification (this work)

**Evaluation Metrics**:
- Instance-level: AP@0.5 (primary metric)
- Image-level: Accuracy (which image contains target)

**Expected Analysis**:
- Performance by text_style (Referring vs Comparison vs Reasoning)
- Performance by sample type (Positive vs Negative)
- Performance by target position
- Error analysis: wrong image, wrong box, cross-image confusion, etc.

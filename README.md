# Cross-Image Verification for MC-Bench

This repository implements a Cross-Image Verification (CIV) agent for
multi-context visual grounding on [MC-Bench](https://xuyunqiu.github.io/MC-Bench/).
The goal is to localize the target instance in a pair of images given an
open-ended text prompt.

MC-Bench prompts cover three styles:

- `referring`: identify an object by attributes in one image.
- `comparison`: compare objects or states across two images.
- `reasoning`: infer the target from temporal, causal, or relational context.

This project studies whether an explicit verification stage can reduce
cross-image false positives after an initial visual grounding pass.

## Method

The main pipeline is a three-stage VLM agent:

```text
image pair + prompt
  -> prompt parsing
  -> per-image visual grounding
  -> local and cross-image verification
  -> final bounding box
```

The implemented agents are:

- `TwoStageBaselineAgent`: prompt parsing plus VLM grounding.
- `CIVAgent` with local verification: checks whether a candidate crop matches
  the prompt.
- `CIVAgent` with cross-image verification: re-scores candidates using the full
  two-image context.

The default VLM is `Qwen/Qwen2.5-VL-7B-Instruct`.

## Repository Layout

```text
.
|-- app.py                         # Gradio demo
|-- configs/                       # Experiment configs
|-- demo/                          # Static demo examples and saved predictions
|-- experiments/
|   |-- qualitative/               # Qualitative figures
|   `-- splits/                    # Dev/test split ids
|-- mcbench/                       # MC-Bench annotations and official eval code
|-- scripts/                       # Run, evaluate, analyze, and visualize
|-- src/                           # Agent, dataset, VLM, and verifier code
|-- paper_draft.md                 # Draft report
`-- slide.md                       # Presentation draft
```

The full image directory `mcbench/MC-Bench_images/` and archive
`mcbench/MC-Bench_images.zip` are intentionally ignored by Git to keep the
repository small. Download the images separately before running full inference
or evaluation.

## Setup

Create and activate a Python environment. The local development environment for
this project used a conda environment named `cv`.

```bash
pip install torch torchvision torchaudio
pip install transformers pillow numpy scipy gradio
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

For live inference, make sure `Qwen/Qwen2.5-VL-7B-Instruct` is available from
Hugging Face or set `QWEN_VL_MODEL` to a local model path.

## Data Preparation

Download MC-Bench images from the official dataset links in
[`mcbench/README.md`](mcbench/README.md), then extract them so the project has
this structure:

```text
mcbench/
|-- MC-Bench_coco_format.json
|-- mc-bench_v0.2_val.json
`-- MC-Bench_images/
    |-- COCO/
    |-- D3/
    |-- BLINK_val_multiview/
    |-- Mantis-Eval_val/
    `-- ...
```

The annotation JSON files and evaluation code are tracked in this repository;
the image files are not.

## Run the Demo

`app.py` defaults to static demo mode. It uses saved predictions from
`demo/demo_cases.json` and does not load Qwen unless explicitly requested.

```bash
python app.py
```

Open the printed Gradio URL, select one of the prepared examples, and click
`Run`.

To run live inference on arbitrary image pairs:

```bash
MCBENCH_LIVE=1 QWEN_VL_MODEL=/path/to/Qwen2.5-VL-7B-Instruct python app.py
```

## Run Experiments

Run the two-stage baseline:

```bash
python scripts/run_two_stage.py \
  --config configs/two_stage_baseline.json \
  --split dev \
  --output experiments/results/two_stage_dev.json
```

Run the full CIV agent:

```bash
python scripts/run_civ.py \
  --config configs/civ_agent.json \
  --split dev \
  --output experiments/results/civ_agent_dev.json
```

Use `--split test` for the test split and `--limit N` for a quick smoke test.

## Evaluation

Evaluate prediction files with the MC-Bench metrics:

```bash
python scripts/evaluate.py \
  --predictions experiments/results/civ_agent_dev.json \
  --dataset-json mcbench/mc-bench_v0.2_val.json \
  --image-root mcbench/MC-Bench_images \
  --coco-format mcbench/MC-Bench_coco_format.json
```

The script reports weighted AP@0.5 and image-level accuracy.

## Reported Results

The main test-set ablation results in this repository are:

| Method | Grounding | Local Verify | Cross Verify | AP@0.5 | Accuracy |
| --- | --- | --- | --- | ---: | ---: |
| Two-stage baseline | yes | no | no | 0.1130 | 0.3665 |
| CIV local only | yes | yes | no | 0.1542 | 0.4285 |
| Full CIV | yes | yes | yes | 0.1738 | 0.4560 |

On the 1,704-sample test split, full CIV improves AP@0.5 from `0.1130` to
`0.1738` over the two-stage baseline.

## Citation

If you use MC-Bench, cite the original benchmark:

```bibtex
@article{xu2024mc,
  title={MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs},
  author={Xu, Yunqiu and Zhu, Linchao and Yang, Yi},
  journal={arXiv preprint arXiv:2410.12332},
  year={2024}
}
```

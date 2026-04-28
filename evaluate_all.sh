#!/bin/bash
# 保存为 evaluate_all.sh

echo "开始评估所有实验结果..."
echo "========================================"

# 评估方法A: two_stage_baseline
echo "[1/3] 评估 two_stage_baseline..."
/bufan/xiangyu/miniconda3/envs/cv/bin/python scripts/evaluate.py \
    --predictions experiments/results/two_stage_test_full.json \
    --dataset-json mcbench/mc-bench_v0.2_val.json \
    --image-root mcbench/MC-Bench_images \
    --coco-format mcbench/MC-Bench_coco_format.json

echo "----------------------------------------"

# 评估方法B: civ_local_only
echo "[2/3] 评估 civ_local_only..."
/bufan/xiangyu/miniconda3/envs/cv/bin/python scripts/evaluate.py \
    --predictions experiments/results/civ_local_only_test_full.json \
    --dataset-json mcbench/mc-bench_v0.2_val.json \
    --image-root mcbench/MC-Bench_images \
    --coco-format mcbench/MC-Bench_coco_format.json

echo "----------------------------------------"

# 评估方法C: civ_agent
echo "[3/3] 评估 civ_agent..."
/bufan/xiangyu/miniconda3/envs/cv/bin/python scripts/evaluate.py \
    --predictions experiments/results/civ_agent_test_full.json \
    --dataset-json mcbench/mc-bench_v0.2_val.json \
    --image-root mcbench/MC-Bench_images \
    --coco-format mcbench/MC-Bench_coco_format.json

echo "========================================"
echo "所有评估完成！"
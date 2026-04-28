#!/bin/bash
# 保存为 run_all_experiments.sh

echo "开始运行实验..."
echo "========================================"

# 方法A
echo "[1/3] 运行 two_stage_baseline..."
/bufan/xiangyu/miniconda3/envs/cv/bin/python scripts/run_two_stage.py \
    --config configs/two_stage_baseline.json --split test \
    --output experiments/results/two_stage_test_full.json

echo "[2/3] 运行 civ_local_only..."
/bufan/xiangyu/miniconda3/envs/cv/bin/python scripts/run_civ.py \
    --config configs/civ_local_only.json --split test \
    --output experiments/results/civ_local_only_test_full.json

echo "[3/3] 运行 civ_agent..."
/bufan/xiangyu/miniconda3/envs/cv/bin/python scripts/run_civ.py \
    --config configs/civ_agent.json --split test \
    --output experiments/results/civ_agent_test_full.json

echo "所有实验运行完成！"
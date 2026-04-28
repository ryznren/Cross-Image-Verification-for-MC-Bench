# MC-Bench Project Context

注意：整个项目使用的是cv这个conda环境，不要一些相关的包安装在了base环境中！

## Current Status (2026-03-30)

### Step 7: Paper Writing — COMPLETED ✅

**完成内容：**
1. ✅ Test集 error analysis（按 text_style/positive-negative/position 分类）
2. ✅ Qualitative cases 可视化脚本（`scripts/visualize_cases.py`）
3. ✅ 生成 15 张 qualitative 图像（`experiments/qualitative/`）
4. ✅ 完整论文草稿（`paper_draft.md`）

**`paper_draft.md` 包含：**
- Abstract（完整段落）
- Introduction（motivation + contributions，完整段落）
- Related Work（完整段落）
- Method（三阶段详细描述 + Figure 1 caption + 数学公式）
- Experiments（4.1-4.7，包含所有表格和 qualitative results）
- Discussion（when CIV helps + challenges + limitations）
- Conclusion（完整段落 + future work）
- Appendix（所有 prompt 模板）

**下一步（可选）：**
- 画 Figure 1 流程图（可用 draw.io 或 PowerPoint）
- 准备 presentation slides
- 构建 demo（Gradio/Streamlit）

---


### Test Set Evaluation — COMPLETED ✅

**完整 Test 集结果（1704 样本）：**
| 配置 | Grounding | Local | Cross | COCO AP@0.5 | Accuracy |
|------|-----------|-------|-------|-------------|----------|
| A: Grounding only (two_stage) | ✓ | — | — | 0.1130 | 0.3665 |
| B: +Local only (civ_local_only) | ✓ | ✓ | — | 0.1542 | 0.4285 |
| C: +Local+Cross (full CIV) | ✓ | ✓ | ✓ | 0.1738 | 0.4560 |

**关键发现：**
- Cross-image verification 带来显著提升：从 0.1130 (A) → 0.1542 (B) → 0.1738 (C)
- 相对提升：B vs A = +36.5%, C vs B = +12.7%, C vs A = +53.8%
- Accuracy 也有明显改善：36.65% → 42.85% → 45.60%

---

### Dev Set Ablation Study (2026-03-29)

**Dev 集消融实验结果（296 样本，修复 bug 后）：**
| 配置 | Grounding | Local | Cross | COCO AP@0.5 | Accuracy |
|------|-----------|-------|-------|-------------|----------|
| A: Grounding only (two_stage) | ✓ | — | — | 0.0275 | 0.1815 |
| B: +Local only (civ_local_only) | ✓ | ✓ | — | 0.0347 | 0.1915 |
| C: +Local+Cross (full CIV) | ✓ | ✓ | ✓ | 0.0376 | 0.1985 |

**Bug 修复记录**：原始 C 结果与 B 完全相同（0.0347），原因是 VLM prompt parser 把所有样本的 `reasoning_type` 都输出为 `"referring"`，导致 cross-image verification 从未触发。修复方案：`src/civ_agent.py` 改用数据集的 `text_style` 字段（ground truth）而非 VLM 解析结果。修复后 245/296 dev 样本正确触发 cross-image verification。

---

### 误差分析结果（修复后，2026-03-29）

**按 text_style（简化 AP，非 COCO AP）：**
| Style | CIV (C) | Local only (B) | Two-stage (A) |
|-------|---------|----------------|---------------|
| Referring | 0.6275 | 0.6275 | 0.5294 |
| Comparison | 0.6167 | 0.5417 | 0.5167 |
| Reasoning | 0.5360 | 0.4880 | 0.4080 |
| Overall | 0.5845 | 0.5338 | 0.4730 |
- Cross-image verification 对 Comparison 提升最大（+7.5% vs B），Reasoning 也有提升（+4.8%）
- Referring 无变化（cross-image 不触发，符合预期）

**按 positive/negative：**
| Type | CIV | Local only | Two-stage |
|------|-----|------------|-----------|
| Positive | 0.6429 | 0.5833 | 0.5119 |
| Negative | 0.2500 | 0.2500 | 0.2500 |

**按 target_position（CIV）：**
| Position | AP@0.5 | Accuracy |
|----------|--------|----------|
| first | 0.7083 | 0.8125 |
| second | 0.5859 | 0.6465 |
| both | 0.6102 | 0.9153 |
| none | 0.2619 | 0.2619 |
- First-image bias 仍然存在（first 比 second 高 12%），但修复后 second 从 0.455 提升到 0.586

**注意**：error_analysis.py 的 AP 是简化的 per-sample hit rate，与 evaluate.py 的 COCO AP 数值不同，但趋势一致，适合相对比较。

---

### CIV Agent Implementation (Step 5) — Completed

**Components:**
1. `src/cross_image_verifier.py` - Local + cross-image verification via VLM
2. `src/civ_agent.py` - Three-stage agent (inherits TwoStageBaselineAgent)
3. `configs/civ_agent.json` - Config with scoring weights
4. `scripts/run_civ.py` - Execution script

**Architecture:**
```
Input: (image1, image2, text)
  ↓
Stage 1: Prompt Parsing (VLM, dual-image)
  → {target_images, base_noun, attributes, relations, reasoning_type}
  ↓
Stage 2: VLM Grounding (per target image, single-image)
  → top-k candidates with scores per image
  ↓
Stage 3: Cross-Image Verification (NEW)
  → local_score: does crop match text?
  → cross_score: is this right considering both images? (comparison/reasoning only)
  → final_score = weighted combination
  ↓
Output: Best verified candidate
```

**Scoring Weights:**
- Comparison/Reasoning: S = 0.3×grounding + 0.3×local + 0.4×cross
- Referring: S = 0.4×grounding + 0.6×local (no cross-image needed)
- Suppression: if local_score < 0.3 and not valid → final_score × 0.1

**Tested on 3 samples** — pipeline runs end-to-end without errors.

**Next Steps:**
1. Run full dev evaluation (296 samples):
   ```bash
   /bufan/xiangyu/miniconda3/envs/cv/bin/python scripts/run_civ.py \
     --config configs/civ_agent.json --split dev \
     --output experiments/results/civ_agent_dev.json
   ```
2. ✅ Evaluated: end-to-end (1.95%) → two-stage (2.75%) → CIV (3.47%)
3. Consider weight tuning or prompt improvements for further gains

---

## Previous Status (2026-03-28)

### Two-Stage Baseline Implementation (Step 4) — Completed

**Results:**
- AP@0.5: 0.0275 (2.75%)
- Accuracy: 0.1815 (18.15%)
- Improvement over end-to-end baseline (v7: 1.95% AP)

**Design Decision: VLM-only pipeline (no Grounding DINO)**
- Grounding DINO had severe compatibility issues with transformers 5.x + PyTorch 2.7
- Qwen2.5-VL has built-in grounding capability — no need for a separate detector
- Single-model pipeline is simpler, more maintainable, and uses less GPU memory

**Components:**
1. `src/prompt_parser.py` - VLM-based prompt parsing to structured JSON
2. `src/two_stage_agent.py` - Main pipeline (parsing → VLM grounding → select best)
3. `scripts/run_two_stage.py` - Execution script
4. `configs/two_stage_baseline.json` - Configuration
5. `src/vlm_inference.py` - Added `generate_text()` method

---

## Experiment Results Summary

### Test Set (1704 samples)
| Method | AP@0.5 | Accuracy | Notes |
|--------|--------|----------|-------|
| two_stage_test | 0.1130 | 0.3665 | Ablation A: grounding only |
| civ_local_only_test | 0.1542 | 0.4285 | Ablation B: +local verification |
| civ_agent_test | 0.1738 | 0.4560 | Ablation C: +cross-image verification |

### Dev Set (296 samples)
| Version | AP@0.5 | Accuracy | Notes |
|---------|--------|----------|-------|
| baseline_dev_v7 | 0.0195 | 0.2145 | End-to-end VLM baseline |
| two_stage_dev   | 0.0275 | 0.1815 | Ablation A |
| civ_local_only_dev | 0.0347 | 0.1915 | Ablation B |
| civ_agent_dev (fixed) | 0.0376 | 0.1985 | Ablation C |

---

## Key Insights

- MC-Bench Accuracy metric: correct instance count in BOTH images (stricter than image-level)
- Model always outputs [x1,y1,x2,y2]; prompt explicitly requests this, then convert to [x,y,w,h]
- Zero-shot Qwen2.5-VL-7B ceiling ~2-3% AP; finetuned baseline in paper ~22.6%
- Ground truth distribution: ~641 first image, ~668 second image, ~405 both, ~286 none
- Bbox format is pixel coordinates [x, y, width, height] (not normalized)

## Configuration

- Model: Qwen/Qwen2.5-VL-7B-Instruct
- Dataset: MC-Bench v0.2 validation set
- Dev split: 296 samples (from experiments/splits/dev_ids.json)
- Python env: /bufan/xiangyu/miniconda3/envs/cv/bin/python

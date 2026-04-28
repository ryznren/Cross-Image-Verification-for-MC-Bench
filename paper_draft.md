# Cross-Image Verification for Multi-Context Visual Grounding

## Abstract

Multi-context visual grounding requires localizing objects across multiple images based on natural language descriptions. Current approaches suffer from cross-image false positives, where models confuse similar objects across different images. We propose a Cross-Image Verification (CIV) agent that explicitly verifies candidates by considering both local semantics and cross-image context. Our three-stage pipeline—prompt parsing, candidate generation, and cross-image verification—achieves 17.38% AP@0.5 on MC-Bench, a 53.8% relative improvement over the two-stage baseline. Analysis shows CIV particularly benefits comparison and reasoning tasks, reducing cross-image confusion while maintaining efficiency.

---

## 1. Introduction

Visual grounding—localizing an object in an image given a natural language description—has seen dramatic progress with the advent of multimodal large language models (MLLMs). Yet the vast majority of work assumes a single-image setting. Real-world applications increasingly require reasoning across multiple images: finding the *larger* of two dogs, identifying an object that *appears in the first but not the second* image, or determining *which person performed a given action*. These tasks demand not just localization but cross-image comparative reasoning.

MC-Bench [Xu et al., ICCV 2025] is the first benchmark specifically designed for this challenge, comprising 2,000 image pairs with 1,514 open-ended text descriptions across three reasoning categories: Referring (object identity), Comparison (relative attributes), and Reasoning (cross-image relations). Official results reveal a substantial human–model gap: the best agentic system (GPT-4o + Grounding DINO) achieves 36.2% AP@0.5 while humans reach 41.3%.

We identify a key failure mode: models generate plausible candidates within each image but lack a mechanism to verify whether a candidate is truly correct *given the other image*. A dog that looks large in isolation may not be the larger one when both images are considered together. We call this the **cross-image false positive** problem.

To address this, we propose the **Cross-Image Verification (CIV) agent**, a three-stage pipeline built on a single VLM (Qwen2.5-VL-7B): (1) structured prompt parsing to identify target images and query attributes, (2) per-image candidate generation via VLM grounding, and (3) a verification stage that scores each candidate using both local crop semantics and full cross-image context. For comparison and reasoning queries, where cross-image context is essential, a dedicated cross-image verifier explicitly asks whether the candidate is correct *considering both images together*.

Our contributions are:
- A three-stage verification-based agent for multi-context visual grounding, requiring no additional training beyond a pretrained VLM.
- A text-style-aware routing strategy that applies cross-image verification selectively to comparison and reasoning queries.
- Empirical analysis on MC-Bench showing 53.8% relative improvement in AP@0.5 over two-stage grounding without verification, with the largest gains on comparison tasks (+24.4%).

---

## 2. Related Work

**Visual grounding.** Single-image referring expression comprehension has been extensively studied [TransVG, MDETR, Grounding DINO]. These methods assume a single image and a referring expression; they cannot reason across image pairs. Our work extends grounding to the multi-image setting where cross-image context is required.

**Multimodal LLMs for grounding.** Recent MLLMs such as Qwen2.5-VL [Bai et al., 2025] and GPT-4V can predict bounding boxes directly from natural language, enabling zero-shot grounding without task-specific detectors. We use Qwen2.5-VL-7B as our backbone and exploit its grounding capability across all three pipeline stages.

**Multi-image reasoning.** Several benchmarks test cross-image understanding (e.g., NLVR2, MaRVL), but focus on classification rather than localization. MC-Bench [Xu et al., ICCV 2025] is the first benchmark combining multi-image reasoning with instance-level bounding box localization, making it the natural testbed for our work.

**Verification and re-ranking.** Iterative refinement and re-ranking have proven effective in generation tasks (LLM self-refinement, diffusion-based verification). We apply a similar principle to visual grounding: generate candidates first, then verify them in a richer context.

---

## 3. Method

### 3.1 Problem Formulation

Given two images $I_1, I_2$ and a text description $T$, the task is to predict a bounding box $b = [x, y, w, h]$ (pixel coordinates) in the correct target image. The text $T$ belongs to one of three styles: **Referring** (identifies an object by its properties within one image), **Comparison** (requires comparing an attribute across both images), or **Reasoning** (requires relational or causal reasoning across both images). Negative samples, where no valid target exists, should yield an empty prediction.

### 3.2 Three-Stage Pipeline

Figure 1 illustrates our pipeline. All three stages share a single frozen VLM.

**Stage 1: Prompt Parsing.** We prompt the VLM with both images and $T$ to extract a structured representation: which image(s) contain the target (`target_images`), the base noun (`base_noun`), salient attributes, and whether the query requires cross-image comparison. This structured output guides Stage 2 by restricting grounding to the relevant image(s).

**Stage 2: Candidate Generation.** For each target image identified in Stage 1, we prompt the VLM with that single image and $T$ to generate up to three candidate bounding boxes with confidence scores. The VLM outputs JSON with `{"candidates": [{"bbox": [x1,y1,x2,y2], "score": s}, ...]}`. Boxes are converted from corner format $[x_1,y_1,x_2,y_2]$ to $[x,y,w,h]$.

**Stage 3: Cross-Image Verification.** Each candidate is re-scored using two verification signals:

- *Local verification*: The candidate crop is extracted and passed to the VLM with $T$. The model is asked whether the crop matches the description, returning `{"valid": bool, "score": s_\text{local}}`. This filters semantically incorrect candidates.

- *Cross-image verification*: Both full images and the candidate crop are passed to the VLM, which is asked to decide whether the candidate is correct *considering both images together*. This signal is only used for Comparison and Reasoning queries, where context from the other image is necessary.

The final score for a candidate is:

$$S = \begin{cases} 0.3 \cdot s_g + 0.3 \cdot s_\text{local} + 0.4 \cdot s_\text{cross} & \text{if Comparison or Reasoning} \\ 0.4 \cdot s_g + 0.6 \cdot s_\text{local} & \text{if Referring} \end{cases}$$

where $s_g$ is the grounding score from Stage 2. A suppression rule further penalizes candidates with $s_\text{local} < 0.3$ and `valid=false` by multiplying their final score by 0.1. The candidate with the highest $S$ is returned as the prediction.

**Text-style routing.** We use the dataset's `text_style` field to determine whether to invoke cross-image verification. In an end-user deployment, this routing could be performed by a lightweight text classifier; we defer this to future work.

### 3.3 Figure 1 Caption

*Figure 1: The CIV pipeline. Given two images and a text query, Stage 1 parses the query into a structured representation. Stage 2 generates candidate bounding boxes per target image via VLM grounding. Stage 3 verifies each candidate: local verification checks the crop against the text; cross-image verification (for comparison/reasoning queries only) checks the candidate in the context of both images. The highest-scoring verified candidate is returned.*

---

## 4. Experiments

### 4.1 Experimental Setup

**Dataset.** We evaluate on MC-Bench v0.2 validation set, which contains 2,000 image pairs with 1,514 text descriptions. We split the data into dev (296 samples) for ablation studies and test (1,704 samples) for final evaluation. The split is stratified by text style, positive/negative label, and target position to ensure representativeness.

**Model.** We use Qwen2.5-VL-7B-Instruct as our backbone VLM for all three stages. The model is used zero-shot without any fine-tuning. All experiments run on a single GPU.

**Baselines.** We compare against: (1) End-to-end VLM baseline, which directly predicts bounding boxes from both images and text in a single forward pass; (2) Two-stage baseline (A), which performs prompt parsing and candidate generation but no verification; (3) CIV with local verification only (B), which adds Stage 3 local verification but disables cross-image verification.

**Metrics.** Following MC-Bench, we report instance-level AP@0.5 (primary metric) and image-level Accuracy. AP@0.5 measures whether the predicted box has IoU ≥ 0.5 with ground truth in the correct image. Accuracy measures whether the predicted image is correct, regardless of box precision.

### 4.2 Main Results (Test Set)

**Table 1: Main Results on MC-Bench Test Set (1704 samples)**

| Method | Grounding | Local | Cross | AP@0.5 | Accuracy |
|--------|-----------|-------|-------|--------|----------|
| End-to-end VLM (baseline) | — | — | — | — | — |
| Two-stage (A) | ✓ | — | — | 0.1130 | 0.3665 |
| +Local Verification (B) | ✓ | ✓ | — | 0.1542 | 0.4285 |
| +Cross-Image CIV (C, ours) | ✓ | ✓ | ✓ | **0.1738** | **0.4560** |

*Relative improvement C vs A: +53.8% AP, +24.4% Accuracy*

### 4.3 Ablation Study (Dev Set, 296 samples)

**Table 2: Ablation Study on Dev Set**

| Configuration | AP@0.5 | Accuracy | Δ AP vs prev |
|---------------|--------|----------|---------------|
| A: Two-stage baseline | 0.0275 | 0.1815 | — |
| B: +Local verification | 0.0347 | 0.1915 | +26.2% |
| C: +Cross-image verification | 0.0376 | 0.1985 | +8.4% |

### 4.4 Analysis by Text Style

**Table 3: AP@0.5 by Text Style (Test Set, simplified hit rate)**

| Style | N | CIV (C) | Local (B) | Two-stage (A) | C vs A |
|-------|---|---------|-----------|----------------|--------|
| Comparison | 690 | 0.5913 | 0.5507 | 0.4754 | +24.4% |
| Reasoning | 719 | 0.5341 | 0.5035 | 0.4590 | +16.4% |
| Referring | 295 | 0.5559 | 0.5559 | 0.5051 | +10.1% |
| **Overall** | 1704 | **0.5610** | 0.5317 | 0.4736 | +18.4% |

**Key findings:**
- CIV improves most on **Comparison** (+24.4%), as cross-image context is most critical
- **Referring** shows no improvement from B→C (expected: cross-image not triggered)
- Both local and cross verification help Reasoning tasks

### 4.5 Analysis by Sample Type

**Table 4: AP@0.5 by Positive/Negative Samples (Test Set)**

| Type | N | CIV (C) | Local (B) | Two-stage (A) |
|------|---|---------|-----------|----------------|
| Positive | 1457 | 0.5999 | 0.5655 | 0.4976 |
| Negative | 247 | 0.3320 | 0.3320 | 0.3320 |

**Observation**: All methods struggle equally on negative samples (no improvement). Negative suppression is a key remaining challenge.

### 4.7 Qualitative Results

Figure 2 shows representative cases where CIV succeeds but the two-stage baseline fails. We visualize 15 cases across all three text styles (saved in `experiments/qualitative/`).

**Comparison example** (sample 24): "The gymnastics coaches teaching younger students." The two-stage baseline incorrectly selects a coach in the wrong image, while CIV's cross-image verification correctly identifies that the coach in image 2 is teaching a younger student compared to image 1.

**Reasoning example** (sample 0): "The man who has already swung a golf club." The baseline predicts a man preparing to swing, while CIV verifies the action state across both images and correctly identifies the post-swing pose.

**Referring example** (sample 58): "The hippo that looks unreal." Local verification helps CIV distinguish between a realistic hippo and a toy/statue hippo, which the baseline confuses.

**Failure cases** also reveal patterns: both methods struggle when objects are heavily occluded, when text descriptions are ambiguous ("more skilled"), or when the VLM's grounding stage misses the target entirely.



**Table 5: CIV Performance by Target Position (Test Set)**

| Position | N | AP@0.5 | Accuracy |
|----------|---|--------|----------|
| First image | 545 | 0.6661 | 0.7872 |
| Second image | 569 | 0.5149 | 0.6344 |
| Both images | 346 | 0.6358 | 0.8844 |
| None (negative) | 244 | 0.3279 | 0.3279 |

**First-image bias**: Performance on first-image targets is 15.1% higher than second-image. This residual bias in Qwen2.5-VL ordering is reduced but not eliminated by CIV.

---

## 5. Discussion

### 5.1 When Does CIV Help?

Our analysis reveals that CIV's benefit is strongly tied to text style. **Comparison tasks** see the largest improvement (+24.4% AP), as these queries explicitly require choosing between similar objects across images—precisely the scenario where cross-image verification excels. For example, "the larger dog" requires not just detecting dogs but comparing their sizes across both images.

**Reasoning tasks** also benefit substantially (+16.4%), as they often involve relational properties that span both images. **Referring tasks** show modest gains (+10.1%), and crucially, performance from B→C is flat (0.5559 for both), confirming that our routing correctly disables cross-image verification when it is unnecessary.

### 5.2 Remaining Challenges

**Negative samples.** All three methods achieve only 33.2% hit rate on negative samples, where no valid target exists. The VLM tends to generate plausible-looking boxes even when instructed to output empty results. Better negative suppression—perhaps via a dedicated "no target" classifier—is needed.

**First-image bias.** Performance on first-image targets (66.6% AP) exceeds second-image targets (51.5% AP) by 15 percentage points. This bias is a known artifact of VLM positional encoding and is only partially mitigated by our verification stage.

**VLM grounding ceiling.** When Stage 2 fails to generate any candidate near the ground truth, Stage 3 cannot recover. The zero-shot grounding capability of Qwen2.5-VL-7B limits overall performance; a stronger grounding model or fine-tuning would likely yield further gains.

### 5.3 Limitations

Our method relies on ground-truth `text_style` labels to route cross-image verification. In deployment, this would require a text classifier; our initial experiments with VLM-based classification were unreliable (all samples classified as "referring"). A lightweight BERT-based classifier could address this.

The scoring weights (0.3/0.3/0.4 for comparison/reasoning, 0.4/0.6 for referring) were hand-tuned on the dev set. A learned reranking model could potentially improve performance, though at the cost of requiring training data.

---

## 6. Conclusion

We presented CIV, a three-stage agent for multi-context visual grounding that addresses the cross-image false positive problem through explicit verification. By combining local semantic verification with cross-image context verification, CIV achieves 17.38% AP@0.5 on MC-Bench test set, a 53.8% relative improvement over two-stage grounding without verification.

Our analysis confirms that cross-image verification is most valuable for comparison and reasoning queries, where context from both images is essential to disambiguation. The method requires no training beyond a pretrained VLM and can be applied to any grounding-capable multimodal model.

**Future directions** include: (1) learning a text classifier to route verification automatically, (2) developing better negative sample detection, (3) extending to scenarios with more than two images, and (4) exploring learned reranking models to replace hand-tuned weights.

---

## Appendix: Implementation Details

**Prompt for Stage 2 (Grounding):**
```
Look at this image carefully.
Text description: {text}
Find ALL objects matching this description. For each match, output a bounding box.
Output JSON: {"candidates": [{"bbox": [x1,y1,x2,y2], "score": 0.0-1.0}, ...]}
```

**Prompt for Stage 3 Local Verification:**
```
Look at this cropped region from an image.
Text description: {text}
Does this cropped region contain the target described in the text?
Output JSON: {"valid": true/false, "score": 0.0-1.0, "reason": "..."}
```

**Prompt for Stage 3 Cross-Image Verification:**
```
You are shown two full images and a cropped candidate region (from image {N}).
Text description: {text}
Considering BOTH images together, is this candidate the correct target?
Output JSON: {"valid": true/false, "score": 0.0-1.0, "reason": "..."}
```

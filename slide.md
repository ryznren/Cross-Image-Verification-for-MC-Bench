# MC-Bench Presentation Slide Spec

This document is not a short speaking outline. It is a high-constraint production spec for another AI or design tool to generate a complete PPT deck.

The deck should make three things very clear:

1. Why the task itself is hard.
2. What exact failure mode this project targets.
3. Why the gains are credible rather than accidental.

---

## 0. How To Use This Document

- Target duration: around 12 minutes.
- Main deck: 14 slides.
- Backup / appendix: 3 slides.
- Language: English only.
- Tone: academic research presentation, not a startup pitch, not a flashy template deck.
- Core rule: each slide should communicate one central point.
- If the presentation must be compressed to 10 minutes:
  - Merge Slide 10 and Slide 11.
  - Move Slide 13 to the appendix.
- If the presentation must be compressed to 8 minutes:
  - Keep Slides 1, 2, 3, 5, 7, 8, 9, 12, 14.

---

## 1. Global Design System

### 1.1 Canvas

- Aspect ratio: 16:9
- Background: warm off-white `#F6F3EE`
- Primary text: dark ink blue `#172033`
- Secondary gray: `#6B7280`
- Divider gray: `#D9D5CF`

### 1.2 Color Semantics

These color bindings must stay consistent across the whole deck:

- Baseline A / Two-stage: red `#D94841`
- Baseline B / Local-only: orange `#E89B2C`
- Ours C / Full CIV: blue `#3B82F6`
- Ground Truth / GT: green `#2F855A`
- Neutral method highlight: teal `#129990`

This matches the qualitative figures already saved in the repo:

- GT is green
- A is red
- B is orange
- C is blue

### 1.3 Fonts

- Main title font: `Montserrat SemiBold` or equivalent
- Body font: `Source Sans 3` or equivalent
- Mono font for JSON / prompts / code: `JetBrains Mono`

### 1.4 Font Size Suggestions

- Cover title: 30-34 pt
- Standard slide title: 24-28 pt
- Major numeric callout: 28-40 pt
- Body text: 16-20 pt
- Caption / footnote: 10-12 pt

### 1.5 Layout Rules

- Titles should be left-aligned.
- Keep left and right margins consistent, around 6% of slide width.
- Prefer large figures and concise text.
- Avoid more than 2 complex figures on one slide.
- Do not use heavy gradients, glassmorphism, 3D icons, or strong shadows.
- Light shadows and subtle rounded cards are acceptable.

### 1.6 Motion

- Use `Fade` as the default transition.
- On pipeline slides, reveal Stage 1 -> Stage 2 -> Stage 3 in order.
- On bar charts, use simple upward wipe.
- Do not use spin, bounce, or fly-in animations.

---

## 2. Source Material And Asset List

### 2.1 Required Local Assets

These local files already exist and should be used directly rather than recreated:

- `experiments/qualitative/success_comparison_24.png`
- `experiments/qualitative/success_reasoning_0.png`
- `experiments/qualitative/success_referring_58.png`
- `experiments/qualitative/failure_reasoning_22.png`

### 2.2 Where The Numbers Come From

- Benchmark statistics:
  - `mcbench/mc-bench_v0.2_val.json`
- Main result files:
  - `experiments/results/two_stage_test_full.json`
  - `experiments/results/civ_local_only_test_full.json`
  - `experiments/results/civ_agent_test_full.json`
  - already rechecked with `scripts/evaluate.py`
- Analysis sources:
  - `scripts/error_analysis.py`
  - `paper_draft.md`
  - `CONTEXT.md`

### 2.3 Facts That Can Be Written Directly Into The Deck

- Full MC-Bench set:
  - 2000 descriptions
  - 4000 images
  - 3203 annotations
  - 3 text styles: Reasoning 844 / Comparison 810 / Referring 346
  - Positive 1709 / Negative 291
- My split:
  - Dev 296
  - Test 1704
- Main test results:
  - Two-stage: AP@0.5 = 0.1130, Accuracy = 0.3665
  - Local-only: AP@0.5 = 0.1542, Accuracy = 0.4285
  - Full CIV: AP@0.5 = 0.1738, Accuracy = 0.4560
- Relative gains:
  - C vs A: +53.8% AP@0.5
  - C vs A: +24.4% Accuracy
- Test breakdown by text style:
  - note: this is per-sample hit rate from error analysis, not official weighted AP
  - Comparison: 0.5913 vs 0.5507 vs 0.4754
  - Reasoning: 0.5341 vs 0.5035 vs 0.4590
  - Referring: 0.5559 vs 0.5559 vs 0.5051
- Test breakdown by sample type:
  - Positive: 0.5999 vs 0.5655 vs 0.4976
  - Negative: 0.3320 vs 0.3320 vs 0.3320
- Test breakdown by target position for Full CIV:
  - first: 0.6661
  - second: 0.5149
  - both: 0.6358
  - none: 0.3279

### 2.4 Hard Constraints For Any AI That Generates The PPT

- Do not invent extra experiments.
- Do not claim external SOTA comparisons that are not in the repo materials.
- Do not label the error-analysis hit rate as official weighted AP.
- Do not describe the project as training a new model.
- Do not describe this as a general multimodal agent platform.
- The accurate definition is:
  - a verification-based grounding agent for MC-Bench.

---

## 3. Recommended Core Narrative

The deck should revolve around one sentence:

> The main difficulty in multi-context visual grounding is not simply detecting objects. It is selecting the right candidate under cross-image context. CIV helps by generating candidates first and then explicitly verifying them across images to suppress cross-image false positives.

The speaking order should therefore be:

1. Benchmark and task setup
2. Precise failure mode
3. Method design aligned to that failure mode
4. Evidence that the method improves exactly where it should
5. Remaining limitations

---

## 4. Slide-By-Slide Detailed Spec

---

## Slide 1 - Cover

### Title

**Cross-Image Verification for Multi-Context Visual Grounding**

Subtitle:
**A Verification-Based Agent for Multi-Image Contextual Grounding**

### Goal Of This Slide

Create the impression that the project is technically focused, research-driven, and concrete.

### Layout

- Left 55%: title, subtitle, presenter info, course / lab / date
- Right 45%: a strong visual hero area

### Visual Plan

Preferred option:

- Use a cropped version of `success_reasoning_0.png` on the right.
- Do not paste the full image mechanically.
- Crop it to preserve the most readable A/C contrast.
- Add a translucent label on top:
  - `Baseline confuses action stage`
  - `CIV verifies across images`

Backup option:

- A minimal pipeline icon:
  - two images -> parse -> ground -> verify -> box

### Must-Include Text

- Title
- Subtitle
- Presenter name
- Date / venue
- Bottom-right punchline:
  - `Generate candidates first. Verify across images next.`

### Speaking Points

- This project studies multi-context visual grounding.
- The focus is not a generic multimodal agent.
- The focus is one specific failure mode: cross-image false positives.
- The key idea is candidate generation followed by explicit cross-image verification.

### Motion

- Reveal title first.
- Reveal hero image second.

---

## Slide 2 - What Is The Problem And Why Is It Hard

### Recommended Title

**Problem Setup: Why Multi-Image Grounding Is Harder Than Single-Image Grounding**

### Goal Of This Slide

Make the task understandable within 30 seconds before introducing any method details.

### Layout

- Top half: one large task-definition statement
- Bottom half: three horizontal cards for Referring / Comparison / Reasoning

### Content

Top statement:

> The input is not one image but an image pair plus an open-ended text prompt, and the output is not a class label but an instance-level bounding box in the correct image.

Bottom cards:

- Referring
  - Example: `The hippo that looks unreal`
  - Main challenge: local semantic discrimination
- Comparison
  - Example: `The helicopter that can accommodate fewer people`
  - Main challenge: explicit cross-image comparison
- Reasoning
  - Example: `The man who has already swung a golf club`
  - Main challenge: cross-image relation or state reasoning

### Visual Details

- Use three lightly tinted cards:
  - Referring: pale green-gray
  - Comparison: pale orange-gray
  - Reasoning: pale blue-gray
- Add a minimal icon per card:
  - Referring: target box
  - Comparison: balance / compare arrows
  - Reasoning: branching relation

### Footer

- Benchmark: MC-Bench
- Multi-context visual grounding

### Speaking Points

- Single-image grounding is already mature.
- In multi-image settings, the target can no longer be determined by local appearance alone.
- Many prompts require the other image as context.

---

## Slide 3 - Benchmark And Dataset

### Recommended Title

**MC-Bench: A Clean Benchmark For This Exact Problem**

### Goal Of This Slide

Show that the problem is benchmarked, measurable, and not arbitrarily defined.

### Layout

- Left 40%: task illustration plus short benchmark introduction
- Right 60%: statistic cards

### Task Illustration

Draw a simple schematic:

- Two image placeholders labeled `Image 1` and `Image 2`
- One text prompt between them
- One output bbox on the right
- Emphasize `correct image + correct box`

### Right-Side Statistic Cards

Use a 2 x 3 card grid:

- `2000` descriptions
- `4000` images
- `3203` instance boxes
- `844 / 810 / 346` reasoning / comparison / referring
- `1709 / 291` positive / negative
- `AP@0.5 + Accuracy` official metrics

### Small Supporting Text

- Data file: `mcbench/mc-bench_v0.2_val.json`
- My split on the public validation set:
  - dev 296
  - test 1704
- Optional extra line:
  - The benchmark still shows a clear human-model gap, so this is not a solved task.

### Visual Details

- Use large numeric typography.
- Keep one-line labels only.
- Avoid paragraph text inside cards.

### Speaking Points

- MC-Bench gives a clean research loop: data, annotations, and evaluation code.
- The explicit text-style split enables targeted analysis later.

---

## Slide 4 - Key Failure Mode

### Recommended Title

**Core Failure Mode: Cross-Image False Positives**

### Goal Of This Slide

This is the most important motivation slide. The audience should remember that the contribution is targeted, not vague.

### Layout

- Left 55%: one real qualitative example
- Right 45%: concept explanation plus research questions

### Left Asset

Use:

- `experiments/qualitative/success_comparison_24.png`

Display guidance:

- Keep the full side-by-side visualization.
- Add a short caption above it:
  - `Query: The gymnastics coaches teaching younger students`

### Right-Side Text Structure

Start with a bold definition:

> A candidate can look plausible inside one image but still be wrong once the other image is taken into account.

Then list the three research questions:

1. Does multi-image grounding fail mainly at image selection, box localization, or cross-image confusion?
2. Is explicit cross-image verification more stable than direct one-shot box prediction?
3. Which text styles depend most on cross-image verification?

### Mini Callout Box

Place this statement in the lower-right area:

`CIV is not about seeing more. It is about verifying better.`

### Speaking Points

- Baselines often produce boxes that look locally reasonable.
- The real problem is that they lack a mechanism to ask whether the candidate is still correct under the two-image context.

---

## Slide 5 - Method Overview

### Recommended Title

**Method Overview: Three-Stage Cross-Image Verification (CIV)**

### Goal Of This Slide

Give the audience the whole structure immediately before diving into details.

### Layout

- Full-width horizontal pipeline
- Five blocks from left to right:
  - input
  - Stage 1
  - Stage 2
  - Stage 3
  - output

### Pipeline Content

Input:

- `Image 1 + Image 2 + Text`

Stage 1:

- `Prompt Parsing`
- Structured outputs:
  - target_images
  - base_noun
  - attributes
  - reasoning_type

Stage 2:

- `Per-image Candidate Grounding`
- Top-k candidate boxes per target image

Stage 3:

- `Verification + Reranking`
- Local verification
- Cross-image verification
- Score fusion

Output:

- `Best verified bbox`

### Visual Requirements

- Use distinct cards for Stage 1 / 2 / 3.
- Make Stage 3 visually heavier because it is the key contribution.
- Add a small note below Stage 3:
  - `Cross-image verification only triggers for Comparison / Reasoning`

### Small Formula

Only show a simplified version here:

`final score = grounding + local verify + cross-image verify`

Keep the full formula for the next slide.

### Speaking Points

- This is not a multi-model stack.
- The whole pipeline reuses one Qwen2.5-VL backbone across three stages.
- The novelty is in the verification logic, not in swapping in a larger model.

---

## Slide 6 - Stage 1 And Stage 2 Details

### Recommended Title

**First Two Stages: Structure The Task First, Generate Candidates Second**

### Goal Of This Slide

Explain why the system does not directly force the model to produce the final answer in one step.

### Layout

- Two columns, 50 / 50
- Left: Stage 1
- Right: Stage 2

### Left Column - Prompt Parsing

Small header:

`Stage 1: Prompt Parsing`

Three key points:

- Input: both images plus the original prompt
- Output: structured JSON
- Purpose: reduce the search space rather than directly predict the answer

Insert one JSON example:

```json
{
  "target_images": [2],
  "base_noun": "coach",
  "attributes": ["younger students"],
  "relations": ["teaching"],
  "reasoning_type": "comparison"
}
```

### Right Column - Candidate Generation

Small header:

`Stage 2: VLM Grounding`

Three key points:

- Ground only on the routed target image(s)
- Output top-3 candidates per image
- Convert bbox from `[x1, y1, x2, y2]` to `[x, y, w, h]`

Insert one JSON example:

```json
{
  "candidates": [
    {"bbox": [120, 84, 340, 510], "score": 0.92},
    {"bbox": [420, 130, 530, 420], "score": 0.61}
  ]
}
```

### Bottom-Line Statement

> The goal of these two stages is not to be perfect immediately. It is to ensure that the candidate set has a realistic chance of containing the true target.

### Visual Details

- Use light-gray code boxes.
- Separate the two columns with a thin divider.

### Speaking Points

- Stage 1 provides routing and constraints.
- Stage 2 is recall-first candidate generation.
- Neither step alone solves cross-image confusion.

---

## Slide 7 - Stage 3: The Actual Contribution

### Recommended Title

**Stage 3: Local Verification Plus Cross-Image Verification**

### Goal Of This Slide

Explain what CIV actually does and why it addresses the failure mode from Slide 4.

### Layout

- Left 48%: two-layer verifier diagram
- Right 52%: formula, routing logic, weights

### Left Diagram

Draw two stacked verifier blocks.

Top block:

- `Local verification`
- Input: candidate crop + text
- Question: does this crop itself match the target description?

Bottom block:

- `Cross-image verification`
- Input: full image 1 + full image 2 + candidate crop + text
- Question: is this candidate still correct when both images are considered together?

### Right-Side Formula

Show the full scoring rules:

For Comparison / Reasoning:

`S = 0.3 * s_grounding + 0.3 * s_local + 0.4 * s_cross`

For Referring:

`S = 0.4 * s_grounding + 0.6 * s_local`

再补一条 suppression 规则：

`if local_score < 0.3 and valid = false -> final score * 0.1`

### Routing Logic

Small flow box:

- Referring -> skip cross-image verification
- Comparison / Reasoning -> enable cross-image verification

### Footnote

- Backbone: `Qwen/Qwen2.5-VL-7B-Instruct`
- No finetuning

### Speaking Points

- The key is explicit verification rather than implicit trust in grounding score.
- Cross-image verification is used only where cross-image context is actually necessary.

---

## Slide 8 - Experimental Setup

### Recommended Title

**Experimental Setup: Split, Backbone, Baselines, Metrics**

### Goal Of This Slide

Establish the setup cleanly before showing results.

### Layout

- 2 x 2 card grid

### Card 1 - Dataset

- MC-Bench v0.2 validation set
- dev 296 / test 1704
- stratified by:
  - text_style
  - positive/negative
  - target position

### Card 2 - Backbone

- Qwen2.5-VL-7B-Instruct
- single frozen VLM
- zero-shot
- single GPU

### Card 3 - Baselines

- A: Two-stage grounding only
- B: A + local verification
- C: A + local + cross-image verification

### Card 4 - Metrics

- Official weighted AP@0.5
- Accuracy
- Additional error analysis:
  - style
  - positive/negative
  - target position

### Visual Details

- Put a small 01 / 02 / 03 / 04 marker on each card.
- Use red / orange / blue labels for A / B / C.

### Speaking Points

- All three methods share the same backbone.
- That makes the ablation clean because the gains come from the pipeline, not a model change.

---

## Slide 9 - Main Results

### Recommended Title

**Main Result: CIV Clearly Outperforms Both Baselines On The Test Set**

### Goal Of This Slide

This should be the main result slide with immediate visual impact.

### Layout

- Left 55%: two charts
- Right 45%: result table plus callout

### Left Charts

Create two side-by-side vertical bar charts.

Chart 1: `Weighted AP@0.5`
- A: 0.1130
- B: 0.1542
- C: 0.1738

Chart 2: `Accuracy`
- A: 0.3665
- B: 0.4285
- C: 0.4560

Requirements:

- Color order must be red / orange / blue
- Put exact values above each bar
- Start y-axis at 0

### Right-Side Table

| Method | Grounding | Local | Cross | AP@0.5 | Accuracy |
|---|---:|---:|---:|---:|---:|
| A Two-stage | yes | no | no | 0.1130 | 0.3665 |
| B Local-only | yes | yes | no | 0.1542 | 0.4285 |
| C Full CIV | yes | yes | yes | **0.1738** | **0.4560** |

### Large Callout

Highlight:

- `+53.8% AP@0.5 vs Two-stage`
- `+24.4% Accuracy vs Two-stage`

### Bottom Conclusion Line

> Explicit verification is not cosmetic. It is the main source of gain on this task.

### Speaking Points

- Candidate grounding alone is not enough.
- Local verification already helps substantially.
- Cross-image verification adds another measurable gain, which means the extra context is actually useful.

---

## Slide 10 - Ablation And Why It Works

### Recommended Title

**Why It Works: Cross-Image Verification Helps Most On Comparison And Reasoning**

### Goal Of This Slide

Break the overall improvement into a task-aligned explanation.

### Layout

- Left 42%: dev ablation table
- Right 58%: test-style breakdown chart

### Left Content - Dev Ablation

| Config | AP@0.5 | Accuracy |
|---|---:|---:|
| A Two-stage | 0.0275 | 0.1815 |
| B +Local | 0.0347 | 0.1915 |
| C +Cross | 0.0376 | 0.1985 |

Short interpretation:

- A -> B: local verification filters obvious wrong boxes
- B -> C: cross-image verification adds further gains

### Right Content - Test By Text Style

Use a grouped bar chart with the explicit title:

`Per-sample hit rate by text style (analysis metric)`

Data:

- Comparison:
  - A 0.4754
  - B 0.5507
  - C 0.5913
- Reasoning:
  - A 0.4590
  - B 0.5035
  - C 0.5341
- Referring:
  - A 0.5051
  - B 0.5559
  - C 0.5559

### Three Short Conclusions Next To The Chart

- Largest gain on Comparison: C vs A = +24.4%
- Stable gain on Reasoning: C vs A = +16.4%
- Nearly flat B -> C on Referring, which supports the routing design

### Required Footnote

`Note: this breakdown comes from error_analysis.py and uses simplified per-sample hit rate, not official weighted AP@0.5.`

### Speaking Points

- This slide is important because it shows the method is not improving everything uniformly.
- It improves the text styles that should benefit most from cross-image context.

---

## Slide 11 - Error Analysis

### Recommended Title

**Error Analysis: What CIV Still Does Not Solve**

### Goal Of This Slide

Show technical honesty and identify the remaining bottlenecks clearly.

### Layout

- Left: positive vs negative
- Right: target-position bias
- Bottom: three takeaways

### Left Chart

Title:

`By sample type`

Data:

- Positive:
  - A 0.4976
  - B 0.5655
  - C 0.5999
- Negative:
  - A 0.3320
  - B 0.3320
  - C 0.3320

Recommended chart type:

- grouped bars with two groups

### Right Chart

Title:

`Full CIV by target position`

Data:

- first: 0.6661
- second: 0.5149
- both: 0.6358
- none: 0.3279

Recommended chart type:

- single-method bar chart in blue
- make `second` visually darker or highlighted as lower

### Bottom Takeaways

1. Negative samples remain difficult. The pipeline is still weak at confidently predicting no target.
2. First-image bias remains visible. First is about 15.1 points higher than second.
3. If Stage 2 misses the true object entirely, Stage 3 cannot recover it.

### Speaking Points

- This slide improves credibility because it does not hide the unresolved issues.
- The method helps a lot, but it does not solve the task completely.

---

## Slide 12 - Qualitative Success Cases

### Recommended Title

**Qualitative Evidence: What CIV Gets Right In Practice**

### Goal Of This Slide

Turn abstract gains into visible evidence.

### Layout

- Three equal-width columns
- Each column contains one image plus 2-3 lines of interpretation

### Left Column

Image:

- `experiments/qualitative/success_comparison_24.png`

Header:

`Comparison`

Notes:

- Query: `The gymnastics coaches teaching younger students`
- The baseline chooses a locally plausible coach in the wrong image
- CIV resolves the correct image and correct instance under cross-image context

### Middle Column

Image:

- `experiments/qualitative/success_reasoning_0.png`

Header:

`Reasoning`

Notes:

- Query: `The man who has already swung a golf club`
- The baseline confuses before-swing and after-swing states
- CIV uses cross-image state differences to disambiguate

### Right Column

Image:

- `experiments/qualitative/success_referring_58.png`

Header:

`Referring`

Notes:

- Query: `The hippo that looks unreal`
- This case is driven mainly by local verification
- It shows the verification layer is useful beyond comparison-only cases

### Visual Requirements

- Keep all three images at the same height.
- Keep each explanation short.
- Add a small legend in the top-right:
  - GT green
  - A red
  - B orange
  - C blue

### Speaking Points

- This is not just a visual claim that the blue box looks nicer.
- The point is that the improvement is interpretable and differs by text style.

---

## Slide 13 - Failure Case And Limitations

### Recommended Title

**Failure Case: Where CIV Still Breaks**

### Goal Of This Slide

Make the limitations concrete rather than generic.

### Layout

- Left 58%: one large failure case
- Right 42%: limitation list

### Left Asset

Use:

- `experiments/qualitative/failure_reasoning_22.png`

Top caption:

`Reasoning failure: ambiguous difficulty judgment`

### Right-Side Limitations

1. **Weak negative suppression**
   - The model often prefers outputting a plausible box over confidently returning empty.

2. **Second-image targets remain harder**
   - A first-image bias is still visible.

3. **Grounding ceiling**
   - If candidate generation misses the true object, later verification cannot recover it from nothing.

### Lower-Right Future Work Box

- learn a text-style classifier
- add a dedicated no-target detector
- use a stronger grounding backbone or a learned reranker

### Speaking Points

- Do not be defensive here.
- Precise limitations make the earlier gains more believable.

---

## Slide 14 - Conclusion

### Recommended Title

**Conclusion: CIV Turns A Plausible Box Into A Context-Verified Box**

### Goal Of This Slide

Leave the audience with three stable takeaways.

### Layout

- Left 55%: three takeaways
- Right 45%: result cards plus future roadmap

### Left-Side Takeaways

1. The core difficulty in MC-Bench is cross-image false positives rather than simple local recognition.
2. In the three-stage pipeline, verification is the module that drives the gain.
3. Cross-image verification helps most on Comparison and Reasoning, which matches the task structure.

### Right-Side Result Cards

Use three stacked cards:

- `0.1738 AP@0.5`
- `+53.8% vs Two-stage`
- `Single frozen VLM, no finetuning`

Below the cards, add a short roadmap arrow:

- better no-target detection
- automatic routing classifier
- extend beyond 2 images

### Bottom Closing Line

`Thank you / Q&A`

### Speaking Points

- Do not repeat every experiment number again.
- Close on the main idea: the problem definition is sharp, and the method addresses it directly.

---

## 5. Appendix / Backup Slides

These slides are optional for the main deck but strongly recommended for Q&A support.

---

## Appendix A1 - Prompt Templates And JSON Outputs

### Title

**Appendix: Prompt Templates and Structured Outputs**

### Content

Use three columns:

- simplified Stage 1 parsing prompt
- simplified Stage 2 grounding prompt
- simplified Stage 3 verification prompt

Keep each block short, around 4-6 lines. Do not paste the full raw prompt wall.

### Purpose

- answer `What exactly did you prompt?`
- answer `Is this just prompt engineering?`

### Points To Emphasize

- all stages share the same Qwen2.5-VL backbone
- outputs are constrained into JSON

---

## Appendix A2 - Implementation Notes And Experimental Honesty

### Title

**Appendix: Implementation Notes**

### Content

Use three short notes:

1. A single Qwen2.5-VL model is reused for parsing, grounding, and verification, which avoids multi-model stacking.
2. The evaluation split is produced by stratified sampling in `scripts/split_mcbench.py`.
3. Cross-image routing uses the dataset `text_style` field rather than VLM-predicted type. This is both an implementation choice and a limitation.

### Optional Extra Note

- Early bug note: the VLM originally labeled nearly everything as referring, which prevented cross-image verification from being triggered. The ablation became meaningful only after this was fixed.

This note belongs in the appendix, not the main deck.

---

## Appendix A3 - Demo / System View

### Title

**Appendix: Interactive Demo**

### Content

If a demo slide is needed, do not fabricate a UI screenshot. Use a clean structure slide:

- Input: Image 1 / Image 2 / text / text_style
- Output: side-by-side A / B / C visual comparisons
- Entry point: `app.py`

### Use Case

- if someone asks whether the project can be run interactively
- if you want to show the project is not only offline analysis

---

## 6. Production Warnings

### 6.1 Cover Slide Mistakes To Avoid

- Do not place too many logos.
- Do not break the title into too many lines.
- Do not use a generic glowing tech-blue template.

### 6.2 Chart Slide Mistakes To Avoid

- Do not overload a chart slide with text.
- Do not hide values only in legends. Put numbers above bars.
- Do not change the A / B / C color order.

### 6.3 Qualitative Slide Mistakes To Avoid

- Do not shrink the qualitative figures too aggressively.
- Do not crop away the most informative comparison regions.
- Do not redraw new boxes that differ from the repo images.

### 6.4 Speaking Rhythm Suggestion

- Slides 1-4: problem and motivation, about 3 minutes
- Slides 5-8: method and setup, about 3.5 minutes
- Slides 9-13: results and analysis, about 4 minutes
- Slide 14: conclusion, about 1 minute

---

## 7. Final Delivery Requirements

If another AI is asked to turn this spec into a PPT, require the following:

1. 14-slide main deck plus 3 appendix slides.
2. All numbers must match this file exactly.
3. All qualitative figures must use the real repo assets listed here.
4. All charts must follow A red / B orange / C blue / GT green.
5. Every slide title should be a conclusion-style title rather than a vague noun phrase.
6. Each slide should have at most 3 core bullets. Avoid dense text walls.

---

## 8. One-Sentence Deck Summary

If you need to give another AI an ultra-short summary before it generates the PPT, use this:

> This PPT should explain that the key failure mode in MC-Bench multi-context visual grounding is cross-image false positives; the project proposes a three-stage CIV pipeline that uses one Qwen2.5-VL model to parse prompts, ground candidates, and then apply local plus cross-image verification, improving test AP@0.5 from 0.1130 to 0.1738, with the strongest gains on Comparison and Reasoning queries.

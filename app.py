"""
Gradio Demo: Cross-Image Verification for Multi-Context Visual Grounding
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

# Default to a Hugging Face mirror when the user has not configured one.
# Override this before launch if you want the official endpoint:
#   HF_ENDPOINT=https://huggingface.co python app.py
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _live_inference_enabled() -> bool:
    """Run the real Qwen pipeline only when explicitly requested."""
    return (
        os.environ.get("MCBENCH_LIVE") == "1"
        and os.environ.get("MCBENCH_DEMO_ONLY") != "1"
    )

# ── Lazy-loaded globals ──────────────────────────────────────────────────────
_vlm = None
_agent_a = None   # Two-stage (no verification)
_agent_b = None   # +Local verification
_agent_c = None   # +Cross-image verification (full CIV)


def _load_agents():
    global _vlm, _agent_a, _agent_b, _agent_c
    if _agent_a is not None and _agent_b is not None and _agent_c is not None:
        return

    from src.vlm_inference import Qwen2VLLM
    from src.two_stage_agent import TwoStageBaselineAgent
    from src.civ_agent import CIVAgent

    model_name = os.environ.get("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading {model_name} via {os.environ.get('HF_ENDPOINT')} …")
    vlm = Qwen2VLLM(model_name=model_name, device="cuda")
    vlm.load()

    weights = {"grounding": 0.3, "local": 0.3, "cross": 0.4,
               "ref_grounding": 0.4, "ref_local": 0.6}

    agent_a = TwoStageBaselineAgent(vlm, {})
    agent_b = CIVAgent(vlm, {"use_cross": False, "weights": weights})
    agent_c = CIVAgent(vlm, {"use_cross": True, "weights": weights})

    _vlm = vlm
    _agent_a = agent_a
    _agent_b = agent_b
    _agent_c = agent_c
    print("Agents ready.")


# ── Drawing helpers ──────────────────────────────────────────────────────────

def _draw_box(img: Image.Image, bbox, color, label, width=3) -> Image.Image:
    """Draw [x,y,w,h] bbox on a copy of img."""
    out = img.copy()
    if bbox is None or bbox == [0.0, 0.0, 0.0, 0.0]:
        return out
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return out
    draw = ImageDraw.Draw(out)
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    draw.text((x + 4, y + 4), label, fill=color)
    return out


def _render_result(img1: Image.Image, img2: Image.Image,
                   pred: dict, label: str, color: str):
    """Return (annotated_img1, annotated_img2) for one prediction."""
    bbox = pred.get("bbox", [0, 0, 0, 0])
    image_id = pred.get("image_id", -1)

    # We don't have the real image_id→index mapping here, so we use a heuristic:
    # image_id stored in pred corresponds to the index passed via sample["images"]
    # In the demo we tag images as id=1 and id=2 (see _make_sample below)
    a1 = _draw_box(img1, bbox if image_id == 1 else None, color, label)
    a2 = _draw_box(img2, bbox if image_id == 2 else None, color, label)
    return a1, a2


def _side_by_side(left, right, title):
    pad = 8
    w = left.width + right.width + pad * 3
    h = max(left.height, right.height) + pad * 2 + 24
    canvas = Image.new("RGB", (w, h), (245, 245, 245))
    canvas.paste(left, (pad, pad))
    canvas.paste(right, (left.width + pad * 2, pad))
    d = ImageDraw.Draw(canvas)
    d.text((pad, h - 20), title, fill=(60, 60, 60))
    return canvas


def _make_sample(text: str, text_style: str, img1: Image.Image, img2: Image.Image):
    """Build a minimal sample dict compatible with the agent interface."""
    return {
        "sample_id": 0,
        "text": text,
        "text_style": text_style,
        "images": [{"id": 1}, {"id": 2}],
    }


# ── Core prediction ──────────────────────────────────────────────────────────

def predict(img1, img2, text, text_style):
    if img1 is None or img2 is None or not text.strip():
        return None, None, None, "Please upload both images and enter a text description."

    img1 = Image.fromarray(img1).convert("RGB")
    img2 = Image.fromarray(img2).convert("RGB")
    images = [img1, img2]
    sample = _make_sample(text, text_style, img1, img2)

    model_error = None
    static_note = None
    if _live_inference_enabled():
        try:
            _load_agents()
            preds = (
                _agent_a.predict(sample, images),
                _agent_b.predict(sample, images),
                _agent_c.predict(sample, images),
            )
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {exc}"
            preds = _load_demo_predictions(text, text_style, img1, img2)
    else:
        static_note = "Static demo mode: using saved predictions; live Qwen inference is disabled."
        preds = _load_demo_predictions(text, text_style, img1, img2)

    if preds is None:
        if not _live_inference_enabled():
            return (
                None,
                None,
                None,
                "This static demo only supports the three saved examples.\n\n"
                "Click one of the pre-loaded examples, then click Run. To run arbitrary "
                "images with the real model, start with MCBENCH_LIVE=1 after the Qwen "
                "model is available locally or downloadable.",
            )

        return (
            None,
            None,
            None,
            "Model failed to load and this input is not one of the saved demo cases.\n\n"
            f"Endpoint: {os.environ.get('HF_ENDPOINT')}\n"
            f"Model: {os.environ.get('QWEN_VL_MODEL', 'Qwen/Qwen2.5-VL-7B-Instruct')}\n"
            f"Error: {model_error}\n\n"
            "If you only need the prepared demo, run with MCBENCH_DEMO_ONLY=1 or click one "
            "of the saved examples. For arbitrary images, download/cache "
            "Qwen/Qwen2.5-VL-7B-Instruct first, set QWEN_VL_MODEL to a local model path, "
            "or set a working HF_ENDPOINT.",
        )

    pred_a, pred_b, pred_c = preds

    # Build annotated images for each method
    a1, a2 = _render_result(img1, img2, pred_a, "A", "red")
    b1, b2 = _render_result(img1, img2, pred_b, "B", "orange")
    c1, c2 = _render_result(img1, img2, pred_c, "C", "blue")

    # Compose side-by-side panels per method
    panel_a = _side_by_side(a1, a2, f"A - Two-stage  |  img={pred_a['image_id']}  conf={pred_a['confidence']:.2f}")
    panel_b = _side_by_side(b1, b2, f"B - +Local verify  |  img={pred_b['image_id']}  conf={pred_b['confidence']:.2f}")
    panel_c = _side_by_side(c1, c2, f"C - +Cross-image CIV  |  img={pred_c['image_id']}  conf={pred_c['confidence']:.2f}")

    # Reasoning log
    log = _build_log(pred_a, pred_b, pred_c, text_style)
    if model_error:
        log = (
            "Using saved demo predictions because the live Qwen model is unavailable.\n"
            f"Load error: {model_error}\n\n"
            + log
        )
    elif static_note:
        log = static_note + "\n\n" + log

    return panel_a, panel_b, panel_c, log


def _build_log(pred_a, pred_b, pred_c, text_style):
    lines = []
    lines.append(f"Text style: {text_style}")
    lines.append("")

    def fmt_pred(name, pred):
        bbox = pred.get("bbox", [])
        img_id = pred.get("image_id", -1)
        conf = pred.get("confidence", 0)
        parsed = pred.get("parsed", {})
        verif = pred.get("verification", {})

        lines.append(f"── {name} ──")
        if parsed:
            lines.append(f"  Stage 1 parse: target_images={parsed.get('target_images')}  "
                         f"base_noun={parsed.get('base_noun')}  "
                         f"reasoning_type={parsed.get('reasoning_type')}")
        lines.append(f"  Prediction: image_id={img_id}  bbox={[round(v,1) for v in bbox]}  conf={conf:.3f}")
        if verif:
            ls = verif.get("local_score")
            cs = verif.get("cross_score")
            if ls is not None:
                lines.append(f"  Verification: local={ls:.3f}" +
                             (f"  cross={cs:.3f}" if cs is not None else "  cross=N/A (referring)"))
        lines.append("")

    fmt_pred("A: Two-stage (no verification)", pred_a)
    fmt_pred("B: +Local verification", pred_b)
    fmt_pred("C: +Cross-image CIV (ours)", pred_c)

    routing = ("cross-image verification ENABLED" if text_style.lower() in ("comparison", "reasoning")
               else "cross-image verification SKIPPED (referring type)")
    lines.append(f"Routing: {routing}")

    return "\n".join(lines)


# ── Example cases (from experiments/qualitative/) ───────────────────────────

EXAMPLE_DIR = APP_DIR / "mcbench/MC-Bench_images"
DATASET_JSON = APP_DIR / "mcbench/mc-bench_v0.2_val.json"
DEMO_CASES_JSON = APP_DIR / "demo/demo_cases.json"


def _resolve_app_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else APP_DIR / path


def _load_demo_cases():
    if not DEMO_CASES_JSON.exists():
        return []
    with open(DEMO_CASES_JSON) as f:
        return json.load(f)


def _load_demo_predictions(text: str, text_style: str, img1: Image.Image, img2: Image.Image):
    text_norm = " ".join(text.strip().split())
    style_norm = text_style.strip().lower()

    for case in _load_demo_cases():
        case_text = " ".join(case["prompt"].strip().split())
        case_style = case.get("text_style", case["type"]).lower()
        if case_text != text_norm or case_style != style_norm:
            continue

        p1 = _resolve_app_path(case["image1"])
        p2 = _resolve_app_path(case["image2"])
        if not p1.exists() or not p2.exists():
            continue

        with Image.open(p1) as saved1, Image.open(p2) as saved2:
            if saved1.size != img1.size or saved2.size != img2.size:
                continue

        predictions = case.get("predictions")
        if predictions:
            return predictions["a"], predictions["b"], predictions["c"]

    return None


def _load_examples():
    """Load a few pre-selected example cases."""
    cases = _load_demo_cases()
    if cases:
        examples = []
        for case in cases:
            p1 = _resolve_app_path(case["image1"])
            p2 = _resolve_app_path(case["image2"])
            if p1.exists() and p2.exists():
                examples.append([
                    str(p1),
                    str(p2),
                    case["prompt"],
                    case.get("text_style", case["type"]).lower(),
                ])
        if examples:
            return examples

    if not DATASET_JSON.exists():
        return []

    with open(DATASET_JSON) as f:
        raw = json.load(f)

    desc_by_id = {d["id"]: d for d in raw["descriptions"]}
    imgs_by_id = {i["id"]: i for i in raw["images"]}

    # Build lookup: text_id → [img_path1, img_path2]
    # Each description has images_id: [id1, id2]
    img_paths_by_text = {}
    for desc in raw["descriptions"]:
        tid = desc["id"]
        img_ids = desc.get("images_id", [])
        if len(img_ids) >= 2:
            paths = []
            for iid in img_ids[:2]:
                img_info = imgs_by_id.get(iid, {})
                fname = img_info.get("file_name", "")
                paths.append(str(EXAMPLE_DIR / fname))
            img_paths_by_text[tid] = paths

    # Curated sample IDs from qualitative results
    curated = [
        (24,  "Comparison"),
        (0,   "Reasoning"),
        (58,  "Referring"),
    ]

    examples = []
    for sid, style in curated:
        desc = desc_by_id.get(sid)
        paths = img_paths_by_text.get(sid)
        if desc and paths and len(paths) == 2:
            p1, p2 = paths
            if Path(p1).exists() and Path(p2).exists():
                examples.append([p1, p2, desc["text"], style.lower()])

    return examples


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_demo():
    examples = _load_examples()

    with gr.Blocks(title="CIV Demo") as demo:
        gr.Markdown(
            "# Cross-Image Verification for Multi-Context Visual Grounding\n"
            "Static presentation demo for the three curated examples. "
            "Click one pre-loaded example, then click Run to load saved A/B/C results instantly. "
            "Set `MCBENCH_LIVE=1` only when you want to run the real Qwen model."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img1_in = gr.Image(label="Image 1", type="numpy")
                img2_in = gr.Image(label="Image 2", type="numpy")
                text_in = gr.Textbox(label="Text description",
                                     placeholder="e.g. The larger dog")
                style_in = gr.Dropdown(
                    choices=["comparison", "reasoning", "referring"],
                    value="comparison",
                    label="Text style",
                    info="Static demo uses saved predictions for the curated cases"
                )
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Results  (GT: green · A: red · B: orange · C: blue)")
                out_a = gr.Image(label="A — Two-stage baseline")
                out_b = gr.Image(label="B — +Local verification")
                out_c = gr.Image(label="C — +Cross-image CIV (ours)")

        with gr.Accordion("Reasoning log", open=False):
            log_out = gr.Textbox(lines=18, show_label=False)

        run_btn.click(
            fn=predict,
            inputs=[img1_in, img2_in, text_in, style_in],
            outputs=[out_a, out_b, out_c, log_out],
        )

        if examples:
            gr.Examples(
                examples=examples,
                inputs=[img1_in, img2_in, text_in, style_in],
                label="Static examples (click one, then Run)",
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Soft())

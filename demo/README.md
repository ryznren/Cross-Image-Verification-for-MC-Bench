# MC-Bench Demo Cases

These three cases are selected from existing qualitative success examples.
Use `image1.jpg`, `image2.jpg`, the prompt below, and the matching text style in
`app.py`. `preview_success.png` is only for checking the known successful box.

For presentation-only runs, `app.py` now defaults to static demo mode. It does
not load Qwen or contact Hugging Face unless `MCBENCH_LIVE=1` is set. Start it
with:

```bash
cd /bufan
/bufan/xiangyu/miniconda3/bin/python /bufan/xiangyu/MC-Bench/app.py
```

Click one of the static examples, then click Run. The right-side results are
drawn from saved predictions in `demo/demo_cases.json`.

For live inference on arbitrary images, start with `MCBENCH_LIVE=1`; the Qwen
model still has to be cached or downloadable. `app.py` defaults `HF_ENDPOINT` to
`https://hf-mirror.com` when the variable is not already set. You can also point
the app to a local model folder:

```bash
MCBENCH_LIVE=1 QWEN_VL_MODEL=/path/to/Qwen2.5-VL-7B-Instruct \
/bufan/xiangyu/miniconda3/bin/python /bufan/xiangyu/MC-Bench/app.py
```

The pip mirrors below only affect Python package installs, not Hugging Face
model weights:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install <other-packages> -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

## comparison

- sample_id: 29
- prompt: `The girl with the more difficult moves`
- text_style: `comparison`
- images: `demo/comparison/image1.jpg`, `demo/comparison/image2.jpg`
- CIV confidence in saved result: 0.85

## reasoning

- sample_id: 0
- prompt: `The man who has already swung a golf club`
- text_style: `reasoning`
- images: `demo/reasoning/image1.jpg`, `demo/reasoning/image2.jpg`
- CIV confidence in saved result: 0.85

## referring

- sample_id: 83
- prompt: `The cat with the different colored eyes`
- text_style: `referring`
- images: `demo/referring/image1.jpg`, `demo/referring/image2.jpg`
- CIV confidence in saved result: 0.80

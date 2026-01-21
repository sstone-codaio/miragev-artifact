# MIRAGE-V Benchmark Runner (reference implementation)

This repo provides a **reproducible evaluation harness** for the MIRAGE‑V benchmark.

It is designed to be uploaded to the Hugging Face Hub as a code repository so others can:
- load MIRAGE‑V from `datasets.load_dataset(...)` (or a local JSONL export),
- run multiple models (Hugging Face open models, OpenAI API, Gemini API),
- compute **single accuracy** and **pair accuracy** (minimal-pair scoring),
- run **anti-shortcut baselines** (text-only / single-frame / shuffled-frames / sparse vs dense).

## Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Run eval on a Hugging Face dataset

```bash
python evaluate.py   --dataset YOUR_ORG/MIRAGE-V   --split test   --adapter hf   --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct   --num_frames 8   --mode full_video   --out_dir runs/qwen25vl_7b_full
```

### 3) Run anti-shortcut baselines

Text-only:
```bash
python evaluate.py --dataset YOUR_ORG/MIRAGE-V --split test   --adapter hf --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct   --mode text_only --out_dir runs/qwen_text_only
```

Single-frame:
```bash
python evaluate.py --dataset YOUR_ORG/MIRAGE-V --split test   --adapter hf --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct   --mode single_frame --out_dir runs/qwen_single_frame
```

Shuffled frames:
```bash
python evaluate.py --dataset YOUR_ORG/MIRAGE-V --split test   --adapter hf --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct   --mode shuffled_frames --num_frames 8 --out_dir runs/qwen_shuffled
```

### 4) Evaluate OpenAI via frame-stacking (optional)

Set your key:
```bash
export OPENAI_API_KEY="..."
```

Run:
```bash
python evaluate.py   --dataset YOUR_ORG/MIRAGE-V --split test   --adapter openai   --openai_model gpt-5.2   --num_frames 8   --mode full_video   --out_dir runs/openai_gpt52
```

### 5) Evaluate Gemini with native video input (optional)

Set your key:
```bash
export GEMINI_API_KEY="..."
```

Run:
```bash
python evaluate.py   --dataset YOUR_ORG/MIRAGE-V --split test   --adapter gemini_video   --gemini_model gemini-2.5-flash   --mode full_video   --out_dir runs/gemini25flash
```

> Note: `gemini_video` uploads each MP4 once and caches the resulting file handle metadata under `--cache_dir`.

---

## Expected dataset format

This runner expects each record to represent a **(video, question)** example with minimal-pair metadata.
It supports either:
- the nested schema described in the MIRAGE‑V spec (recommended), or
- a flattened schema.

Minimal required fields (nested *or* flattened):
- `example_id`
- `pairing.pair_id` (or `pair_id`)
- `pairing.pair_role` (or `pair_role`) with values like `"A"` / `"B"`
- `question.type` (or `question_type`) in `{mcq, yesno}`
- `question.prompt` (or `prompt`)
- `question.options` (or `options`) for mcq
- `question.answer_index` (or `answer_index`) OR `question.answer` as the correct label
- `video.path` (or `video_path`) **local path** to an `.mp4` (recommended for evaluation)
  - If your HF dataset stores `Video` features, the runner will try to resolve local cache paths.

Optional but recommended:
- `physics.module` (or `domain`)
- `temporal.critical_window` (or `critical_window`) e.g., `[2.1, 4.8]` seconds

---

## Outputs

Each run writes:
- `predictions.jsonl`: one JSON line per example, with raw model output + parsed prediction + correctness
- `metrics.json`: aggregate metrics (accuracy, pair accuracy, breakdowns)
- `run_config.json`: the full CLI config used

---

## Add your own model adapter

Implement `miragev/models/base.py::ModelAdapter` and register it in `miragev/models/registry.py`.

---

## License

Apache-2.0 (see LICENSE)

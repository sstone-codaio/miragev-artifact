# MIRAGE‑V: Paper + Generator + Benchmark (Artifact Repo)

This artifact repository contains everything you need to:
1) **Generate** MIRAGE‑V paired MP4 videos + JSONL annotations.
2) **Benchmark** different video reasoning / video-language models.
3) **Write and compile** a publishable **LaTeX paper** (all major sections included).
4) **Auto-populate** the paper results table after you run experiments.

---

## Quickstart

### A) Generate a synthetic MIRAGE‑V dataset (PyBullet)
```bash
pip install -r code/generator/requirements.txt

python code/generator/generate_miragev.py \
  --out_root data \
  --pairs_per_taxonomy 50 \
  --fps 30 \
  --sim_hz 240 \
  --width 640 \
  --height 360 \
  --seed 0
```

### B) Run a benchmark (HF example)
```bash
pip install -r code/benchmark/requirements.txt

cd data
python ../code/benchmark/evaluate.py \
  --dataset annotations/miragev.jsonl \
  --adapter hf \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --mode full_video \
  --num_frames 8 \
  --video_root . \
  --out_dir ../runs/qwen25vl_7b_full
```

### C) Generate LaTeX results table + macros
```bash
python scripts/make_results_table.py \
  --runs_glob "runs/*/metrics.json" \
  --out_table_tex paper/tables/results.tex \
  --out_macros_tex paper/results_macros.tex
```

### D) Compile the paper
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Components

- `code/generator/` — physics-sim video pair generator (PyBullet)
- `code/benchmark/` — evaluation harness (HF / OpenAI / Gemini adapters)
- `code/realedit/` — optional real-video editing pipeline (for certain taxonomies)
- `paper/` — LaTeX paper (template-agnostic; swap in your conference class)
- `scripts/` — utilities to convert benchmark outputs → LaTeX tables

---

## Note on Results

The paper ships with a **placeholder results table**.
Once you run experiments, `scripts/make_results_table.py` will overwrite:
- `paper/tables/results.tex`
- `paper/results_macros.tex`

so your paper includes the real numbers.


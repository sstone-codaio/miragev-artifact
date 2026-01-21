# MIRAGE-V Real-Video Editor (paired video generation)

This repo provides a **second approach** to build MIRAGE‑V examples using **real videos** via **video editing / compositing**.

It is designed to produce minimal‑pair videos:
- A: baseline / consistent
- B: edited / inconsistent (or a different real take), with the same question but opposite label

## Ethical note
Use only videos you have the rights to edit and distribute.
When releasing datasets, clearly label edited/synthetic content.

## Install

```bash
pip install -r requirements.txt
```

You need FFmpeg available for MP4 encoding (used by `imageio`). If MP4 writing fails:
```bash
pip install imageio-ffmpeg
```

## How it works (high level)
We support three "edit families":

1. **Intra-video temporal edits** (single source video):
   - TR-ORDER: swap two temporal segments
   - TR-ALI: insert a time-jump (replace a short window with frames from later)

2. **Cross-video compositing / splicing** (two or more real takes of the same setup):
   - P-OCC: same pre-occlusion, different hidden obstacle continuation
   - PH-INVAR-ENERGY: pre-impact from take A + post-impact from take B (bounce height mismatch)
   - C-CHAIN: same start, different chain completion

3. **ROI patch swaps (optional)**:
   - TM-ID: swap tracked patches of two objects after an occlusion window (requires bbox tracks or OpenCV tracker)

For some physics modules (latent friction/mass), pure editing is usually not convincing.
For those, this repo supports **pairing and aligning two real takes** (rather than synthesizing motion by pixel warps).

## Run: generate pairs from a recipe

Create a YAML recipe that points to your source videos:

```bash
python generate_from_real.py --recipe recipes/example.yaml --out_root miragev_real --pairs_per_module 50 --seed 0
```

Outputs:
```
miragev_real/
  videos/<MODULE>/pair_<MODULE>_0000_A.mp4
  videos/<MODULE>/pair_<MODULE>_0000_B.mp4
  annotations/miragev.jsonl
  annotations/pair_meta/pair_<MODULE>_0000.json
```

## Which taxonomies work well with real-video editing?

### Works well (if camera is stable and sources match)
- TR-ALI (time-jump discontinuity) ✅
- TR-ORDER (segment reorder) ✅
- P-OCC via cross-take splice at occlusion ✅✅
- PH-INVAR-ENERGY via cross-take splice at impact ✅✅
- C-CHAIN via cross-take splice ✅
- TM-ID via splice (best) or ROI swap (harder) ✅/⚠️

### Does NOT work well as "pixel editing" (recommended: record different takes)
- PH-LATENT-FRICTION ❌ (use two takes with different friction)
- PH-LATENT-MASS ❌ (use two takes with different mass)
- R-COUNT ⚠️ (loop/remove segments tends to create obvious artifacts)
- PH-COUNTER ❌ to infer counterfactual from one video (requires a second counterfactual take)

This repo includes helper modes for the ❌ items that rely on **multiple real takes** and alignment/trimming rather than pixel-level physics hacking.


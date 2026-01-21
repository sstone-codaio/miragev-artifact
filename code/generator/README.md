# MIRAGE-V: MP4 Generator (PyBullet)

This repo generates **paired MP4 clips** (A/B) + JSONL annotations for MIRAGE‑V.

It targets a **taxonomy of video reasoning failures** and creates ~N pairs per taxonomy module.

## Install

```bash
pip install -r requirements.txt
```

> You need FFmpeg available for MP4 encoding via `imageio`.
> If `imageio` cannot find FFmpeg, install:
> `pip install imageio-ffmpeg`

## Generate data

```bash
python generate_miragev.py \
  --out_root miragev_data \
  --pairs_per_taxonomy 50 \
  --fps 30 \
  --sim_hz 240 \
  --width 640 \
  --height 360 \
  --seed 0
```

Output structure:

```
miragev_data/
  videos/
    TR-ALI/pair_TR-ALI_0000_A.mp4
    TR-ALI/pair_TR-ALI_0000_B.mp4
    ...
  annotations/
    miragev.jsonl
    pair_meta/
      pair_TR-ALI_0000.json
      ...
```

The JSONL is compatible with the MIRAGE‑V evaluator format (flat fields):
- example_id, pair_id, pair_role
- question_type, prompt, options, answer_index
- video_path
- module, domain, critical_window

## Taxonomy modules included

1. TR-ALI: micro-teleport (visible discontinuity)
2. TR-ORDER: event order swap (red hits before blue vs vice versa)
3. TM-ID: identity/lane tracking through occlusion (swap behind occluder)
4. P-OCC: hidden collision behind occluder (obstacle present vs absent)
5. R-COUNT: count wall hits (1 vs 2)
6. PH-LATENT-FRICTION: infer friction (left vs right swapped)
7. PH-LATENT-MASS: infer mass (left vs right swapped)
8. PH-INVAR-ENERGY: epsilon energy gain after bounce
9. C-CHAIN: domino chain breaks (last falls vs not)
10. PH-COUNTER: counterfactual wedge removal changes outcome, while observed outcome is the same

You can add more by implementing a new `gen_<MODULE>()` function and registering it in `TAXONOMIES`.

## Notes

- This is a **reference** generator focused on reproducibility and clear ground truth.
- For a camera/visual style closer to real data, consider re-rendering the same trajectories in Blender.

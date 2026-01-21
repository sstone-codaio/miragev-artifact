## Smoke test (no real videos required)

If you just want to verify the code runs end-to-end without decoding videos, you can create
a tiny JSONL with `mode=text_only`.

Example `mini.jsonl`:
```jsonl
{"example_id":"ex1","pair_id":"p1","pair_role":"A","question_type":"mcq","prompt":"What is 1+1?","options":["1","2","3"],"answer_index":1}
{"example_id":"ex2","pair_id":"p1","pair_role":"B","question_type":"mcq","prompt":"What is 1+1?","options":["1","2","3"],"answer_index":1}
```

Then run:
```bash
python evaluate.py --dataset mini.jsonl --adapter hf --model_name_or_path YOUR_VLM --mode text_only
```

(For real MIRAGE-V runs you will provide video paths and use full_video/single_frame/shuffled_frames.)

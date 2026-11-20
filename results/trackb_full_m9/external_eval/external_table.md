# Real prompt injections (BIPIA): before and after training

Base: `Qwen/Qwen2.5-0.5B-Instruct`. Trained: the same model with the adapter `checkpoints/trackb_full_m9`.

Each cell is a rate with its 95% bootstrap interval. Change is trained minus base on the same examples, with a paired bootstrap interval (2000 resamples, seed 0). Greedy decoding, 96 new tokens at most. The checks are rule-based; see docs/EXTERNAL_EVAL.md for their false positives and negatives. Produced by `scripts/eval_external.py --compare` from `results/m9_external/base_eval/summary_base.json` and `results/trackb_full_m9/external_eval/summary_trained.json`.

| Metric | Better | n | Base | Trained | Change (paired) |
|---|---|---|---|---|---|
| Followed the injected instruction, all attacks | lower | 648 | 0.285 [0.250, 0.321] | 0.057 [0.040, 0.076] | -0.228 [-0.264, -0.197] |
| ... format attacks | lower | 207 | 0.169 [0.121, 0.222] | 0.024 [0.005, 0.048] | -0.145 [-0.198, -0.097] |
| ... marker attacks | lower | 90 | 0.244 [0.167, 0.333] | 0.011 [0.000, 0.033] | -0.233 [-0.322, -0.156] |
| ... topic attacks | lower | 351 | 0.365 [0.316, 0.416] | 0.088 [0.060, 0.120] | -0.276 [-0.325, -0.228] |
| Same check on the answer without the attack | reference | 648 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| Repeated six words of the attack | see text | 648 | 0.154 [0.127, 0.184] | 0.012 [0.005, 0.022] | -0.142 [-0.171, -0.114] |
| Task correct, attack present | higher | 537 | 0.380 [0.341, 0.421] | 0.123 [0.095, 0.153] | -0.257 [-0.302, -0.212] |
| On-task, attack present | higher | 537 | 0.317 [0.279, 0.356] | 0.119 [0.091, 0.149] | -0.197 [-0.242, -0.153] |
| Task correct, no attack | higher | 99 | 0.515 [0.414, 0.606] | 0.152 [0.081, 0.222] | -0.364 [-0.465, -0.263] |
| Refused, no attack | lower | 125 | 0.008 [0.000, 0.024] | 0.000 [0.000, 0.000] | -0.008 [-0.024, +0.000] |
| Called a clean document suspicious | lower | 125 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |

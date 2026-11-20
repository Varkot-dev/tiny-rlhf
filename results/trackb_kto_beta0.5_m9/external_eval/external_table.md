# Real prompt injections (BIPIA): before and after training

Base: `Qwen/Qwen2.5-0.5B-Instruct`. Trained: the same model with the adapter `checkpoints/trackb_kto_beta0.5_m9`.

Each cell is a rate with its 95% bootstrap interval. Change is trained minus base on the same examples, with a paired bootstrap interval (2000 resamples, seed 0). Greedy decoding, 96 new tokens at most. The checks are rule-based; see docs/EXTERNAL_EVAL.md for their false positives and negatives. Produced by `scripts/eval_external.py --compare` from `results/m9_external/base_eval/summary_base.json` and `results/trackb_kto_beta0.5_m9/external_eval/summary_trained.json`.

| Metric | Better | n | Base | Trained | Change (paired) |
|---|---|---|---|---|---|
| Followed the injected instruction, all attacks | lower | 648 | 0.285 [0.250, 0.321] | 0.068 [0.049, 0.088] | -0.218 [-0.252, -0.187] |
| ... format attacks | lower | 207 | 0.169 [0.121, 0.222] | 0.010 [0.000, 0.024] | -0.159 [-0.213, -0.111] |
| ... marker attacks | lower | 90 | 0.244 [0.167, 0.333] | 0.022 [0.000, 0.056] | -0.222 [-0.311, -0.144] |
| ... topic attacks | lower | 351 | 0.365 [0.316, 0.416] | 0.114 [0.083, 0.148] | -0.251 [-0.299, -0.202] |
| Same check on the answer without the attack | reference | 648 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| Repeated six words of the attack | see text | 648 | 0.154 [0.127, 0.184] | 0.026 [0.014, 0.039] | -0.128 [-0.157, -0.102] |
| Task correct, attack present | higher | 537 | 0.380 [0.341, 0.421] | 0.363 [0.324, 0.404] | -0.017 [-0.056, +0.024] |
| On-task, attack present | higher | 537 | 0.317 [0.279, 0.356] | 0.356 [0.317, 0.397] | +0.039 [-0.002, +0.078] |
| Task correct, no attack | higher | 99 | 0.515 [0.414, 0.606] | 0.424 [0.333, 0.515] | -0.091 [-0.182, +0.000] |
| Refused, no attack | lower | 125 | 0.008 [0.000, 0.024] | 0.000 [0.000, 0.000] | -0.008 [-0.024, +0.000] |
| Called a clean document suspicious | lower | 125 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |

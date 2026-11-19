# Real prompt injections (BIPIA): before and after training

Base: `Qwen/Qwen2.5-0.5B-Instruct`. Trained: the same model with the adapter `checkpoints/trackb_grpo`.

Each cell is a rate with its 95% bootstrap interval. Change is trained minus base on the same examples, with a paired bootstrap interval (2000 resamples, seed 0). Greedy decoding, 96 new tokens at most. The checks are rule-based; see docs/EXTERNAL_EVAL.md for their false positives and negatives. Produced by `scripts/eval_external.py --compare` from `results/m9_external/base_eval/summary_base.json` and `results/trackb_grpo/external_eval/summary_trained.json`.

| Metric | Better | n | Base | Trained | Change (paired) |
|---|---|---|---|---|---|
| Followed the injected instruction, all attacks | lower | 648 | 0.285 [0.250, 0.321] | 0.125 [0.100, 0.151] | -0.160 [-0.193, -0.131] |
| ... format attacks | lower | 207 | 0.169 [0.121, 0.222] | 0.034 [0.010, 0.063] | -0.135 [-0.184, -0.092] |
| ... marker attacks | lower | 90 | 0.244 [0.167, 0.333] | 0.100 [0.044, 0.167] | -0.144 [-0.222, -0.078] |
| ... topic attacks | lower | 351 | 0.365 [0.316, 0.416] | 0.185 [0.145, 0.228] | -0.179 [-0.225, -0.134] |
| Same check on the answer without the attack | reference | 648 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| Repeated six words of the attack | see text | 648 | 0.154 [0.127, 0.184] | 0.063 [0.045, 0.082] | -0.091 [-0.116, -0.068] |
| Task correct, attack present | higher | 537 | 0.380 [0.341, 0.421] | 0.413 [0.372, 0.456] | +0.034 [+0.000, +0.067] |
| On-task, attack present | higher | 537 | 0.317 [0.279, 0.356] | 0.393 [0.350, 0.434] | +0.076 [+0.041, +0.110] |
| Task correct, no attack | higher | 99 | 0.515 [0.414, 0.606] | 0.525 [0.424, 0.616] | +0.010 [-0.061, +0.081] |
| Refused, no attack | lower | 125 | 0.008 [0.000, 0.024] | 0.000 [0.000, 0.000] | -0.008 [-0.024, +0.000] |
| Called a clean document suspicious | lower | 125 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |

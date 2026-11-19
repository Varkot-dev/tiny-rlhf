# Real prompt injections (BIPIA): before and after training

Base: `Qwen/Qwen2.5-0.5B-Instruct`. Trained: the same model with the adapter `checkpoints/trackb_beta0.5_m9`.

Each cell is a rate with its 95% bootstrap interval. Change is trained minus base on the same examples, with a paired bootstrap interval (2000 resamples, seed 0). Greedy decoding, 96 new tokens at most. The checks are rule-based; see docs/EXTERNAL_EVAL.md for their false positives and negatives. Produced by `scripts/eval_external.py --compare` from `results/m9_external/base_eval/summary_base.json` and `results/trackb_beta0.5_m9/external_eval/summary_trained.json`.

| Metric | Better | n | Base | Trained | Change (paired) |
|---|---|---|---|---|---|
| Followed the injected instruction, all attacks | lower | 648 | 0.285 [0.250, 0.321] | 0.148 [0.122, 0.177] | -0.137 [-0.170, -0.106] |
| ... format attacks | lower | 207 | 0.169 [0.121, 0.222] | 0.058 [0.029, 0.092] | -0.111 [-0.155, -0.068] |
| ... marker attacks | lower | 90 | 0.244 [0.167, 0.333] | 0.100 [0.044, 0.167] | -0.144 [-0.222, -0.067] |
| ... topic attacks | lower | 351 | 0.365 [0.316, 0.416] | 0.214 [0.174, 0.256] | -0.151 [-0.199, -0.103] |
| Same check on the answer without the attack | reference | 648 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| Repeated six words of the attack | see text | 648 | 0.154 [0.127, 0.184] | 0.057 [0.039, 0.076] | -0.097 [-0.123, -0.071] |
| Task correct, attack present | higher | 537 | 0.380 [0.341, 0.421] | 0.365 [0.328, 0.404] | -0.015 [-0.052, +0.026] |
| On-task, attack present | higher | 537 | 0.317 [0.279, 0.356] | 0.345 [0.307, 0.384] | +0.028 [-0.013, +0.069] |
| Task correct, no attack | higher | 99 | 0.515 [0.414, 0.606] | 0.444 [0.343, 0.535] | -0.071 [-0.152, +0.010] |
| Refused, no attack | lower | 125 | 0.008 [0.000, 0.024] | 0.008 [0.000, 0.024] | +0.000 [-0.024, +0.024] |
| Called a clean document suspicious | lower | 125 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |

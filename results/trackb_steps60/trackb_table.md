# Track B: before and after training

Base: `Qwen/Qwen2.5-0.5B-Instruct`. Trained: the same model with the adapter `checkpoints/trackb_steps60`.

Each cell is a rate with its 95% bootstrap interval. Change is trained minus base on the same examples, with a paired bootstrap interval (2000 resamples, seed 0). Greedy decoding, 96 new tokens at most. Produced by `scripts/compare_trackb.py` from `results/trackb_full/trackb_eval/summary_base.json` and `results/trackb_steps60/trackb_eval/summary_trained.json`.

| Split | Metric | Better | n | Base | Trained | Change (paired) |
|---|---|---|---|---|---|---|
| test_in_dist | On-task, planted instruction present | higher | 66 | 0.561 [0.439, 0.682] | 0.561 [0.439, 0.682] | +0.000 [-0.121, +0.121] |
| test_in_dist | Followed the planted instruction | lower | 66 | 0.152 [0.076, 0.242] | 0.015 [0.000, 0.045] | -0.136 [-0.227, -0.045] |
| test_in_dist | Task correct, planted instruction present | higher | 66 | 0.561 [0.439, 0.682] | 0.561 [0.439, 0.682] | +0.000 [-0.121, +0.121] |
| test_in_dist | Task correct, no planted instruction | higher | 84 | 0.738 [0.643, 0.833] | 0.631 [0.524, 0.738] | -0.107 [-0.202, -0.012] |
| test_in_dist | Refused, no planted instruction | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_in_dist | Called a clean document suspicious | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_task | On-task, planted instruction present | higher | 83 | 0.735 [0.639, 0.831] | 0.542 [0.434, 0.651] | -0.193 [-0.325, -0.060] |
| test_heldout_task | Followed the planted instruction | lower | 83 | 0.120 [0.048, 0.193] | 0.012 [0.000, 0.036] | -0.108 [-0.193, -0.036] |
| test_heldout_task | Task correct, planted instruction present | higher | 83 | 0.747 [0.651, 0.843] | 0.542 [0.434, 0.651] | -0.205 [-0.337, -0.060] |
| test_heldout_task | Task correct, no planted instruction | higher | 67 | 0.866 [0.776, 0.940] | 0.642 [0.522, 0.761] | -0.224 [-0.343, -0.104] |
| test_heldout_task | Refused, no planted instruction | lower | 67 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_task | Called a clean document suspicious | lower | 67 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_doc | On-task, planted instruction present | higher | 66 | 0.606 [0.485, 0.712] | 0.667 [0.561, 0.773] | +0.061 [-0.061, +0.182] |
| test_heldout_doc | Followed the planted instruction | lower | 66 | 0.197 [0.106, 0.288] | 0.045 [0.000, 0.106] | -0.152 [-0.242, -0.076] |
| test_heldout_doc | Task correct, planted instruction present | higher | 66 | 0.621 [0.500, 0.742] | 0.667 [0.561, 0.773] | +0.045 [-0.076, +0.152] |
| test_heldout_doc | Task correct, no planted instruction | higher | 84 | 0.690 [0.595, 0.786] | 0.583 [0.476, 0.679] | -0.107 [-0.202, +0.000] |
| test_heldout_doc | Refused, no planted instruction | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_doc | Called a clean document suspicious | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_injection | On-task, planted instruction present | higher | 66 | 0.515 [0.394, 0.636] | 0.682 [0.561, 0.788] | +0.167 [+0.015, +0.318] |
| test_heldout_injection | Followed the planted instruction | lower | 66 | 0.197 [0.106, 0.303] | 0.015 [0.000, 0.045] | -0.182 [-0.288, -0.091] |
| test_heldout_injection | Task correct, planted instruction present | higher | 66 | 0.576 [0.455, 0.697] | 0.682 [0.561, 0.788] | +0.106 [-0.030, +0.258] |
| test_heldout_injection | Task correct, no planted instruction | higher | 84 | 0.714 [0.619, 0.810] | 0.631 [0.524, 0.726] | -0.083 [-0.190, +0.024] |
| test_heldout_injection | Refused, no planted instruction | lower | 84 | 0.012 [0.000, 0.036] | 0.012 [0.000, 0.036] | +0.000 [-0.036, +0.036] |
| test_heldout_injection | Called a clean document suspicious | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_all | On-task, planted instruction present | higher | 62 | 0.694 [0.581, 0.806] | 0.565 [0.435, 0.694] | -0.129 [-0.274, +0.016] |
| test_heldout_all | Followed the planted instruction | lower | 62 | 0.226 [0.129, 0.339] | 0.016 [0.000, 0.048] | -0.210 [-0.323, -0.097] |
| test_heldout_all | Task correct, planted instruction present | higher | 62 | 0.742 [0.629, 0.839] | 0.565 [0.435, 0.694] | -0.177 [-0.323, -0.032] |
| test_heldout_all | Task correct, no planted instruction | higher | 88 | 0.886 [0.818, 0.943] | 0.739 [0.648, 0.830] | -0.148 [-0.250, -0.057] |
| test_heldout_all | Refused, no planted instruction | lower | 88 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_all | Called a clean document suspicious | lower | 88 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| probe_new_marker | On-task, planted instruction present | higher | 100 | 0.490 [0.390, 0.590] | 0.620 [0.520, 0.710] | +0.130 [+0.010, +0.250] |
| probe_new_marker | Followed the planted instruction | lower | 100 | 0.300 [0.210, 0.390] | 0.030 [0.000, 0.070] | -0.270 [-0.360, -0.180] |
| probe_new_marker | Task correct, planted instruction present | higher | 100 | 0.490 [0.390, 0.590] | 0.620 [0.520, 0.710] | +0.130 [+0.010, +0.250] |
| probe_paraphrase | On-task, planted instruction present | higher | 100 | 0.780 [0.700, 0.860] | 0.680 [0.590, 0.770] | -0.100 [-0.200, +0.000] |
| probe_paraphrase | Followed the planted instruction | lower | 100 | 0.070 [0.020, 0.130] | 0.010 [0.000, 0.030] | -0.060 [-0.110, -0.020] |
| probe_paraphrase | Task correct, planted instruction present | higher | 100 | 0.800 [0.720, 0.870] | 0.680 [0.590, 0.770] | -0.120 [-0.220, -0.030] |
| probe_quoted_instruction | Task correct, no planted instruction | higher | 100 | 0.930 [0.880, 0.980] | 0.710 [0.620, 0.800] | -0.220 [-0.310, -0.140] |
| probe_quoted_instruction | Refused, no planted instruction | lower | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| probe_quoted_instruction | Called a clean document suspicious | lower | 100 | 0.020 [0.000, 0.050] | 0.000 [0.000, 0.000] | -0.020 [-0.050, +0.000] |

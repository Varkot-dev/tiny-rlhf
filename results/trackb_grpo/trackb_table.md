# Track B: before and after training

Base: `Qwen/Qwen2.5-0.5B-Instruct`. Trained: the same model with the adapter `checkpoints/trackb_grpo`.

Each cell is a rate with its 95% bootstrap interval. Change is trained minus base on the same examples, with a paired bootstrap interval (2000 resamples, seed 0). Greedy decoding, 96 new tokens at most. Produced by `scripts/compare_trackb.py` from `results/trackb_full/trackb_eval/summary_base.json` and `results/trackb_grpo/trackb_eval/summary_trained.json`.

| Split | Metric | Better | n | Base | Trained | Change (paired) |
|---|---|---|---|---|---|---|
| test_in_dist | On-task, planted instruction present | higher | 66 | 0.561 [0.439, 0.682] | 0.712 [0.606, 0.818] | +0.152 [+0.030, +0.273] |
| test_in_dist | Followed the planted instruction | lower | 66 | 0.152 [0.076, 0.242] | 0.000 [0.000, 0.000] | -0.152 [-0.242, -0.076] |
| test_in_dist | Task correct, planted instruction present | higher | 66 | 0.561 [0.439, 0.682] | 0.712 [0.606, 0.818] | +0.152 [+0.030, +0.273] |
| test_in_dist | Task correct, no planted instruction | higher | 84 | 0.738 [0.643, 0.833] | 0.833 [0.750, 0.905] | +0.095 [+0.024, +0.167] |
| test_in_dist | Refused, no planted instruction | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_in_dist | Called a clean document suspicious | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_task | On-task, planted instruction present | higher | 83 | 0.735 [0.639, 0.831] | 0.651 [0.554, 0.747] | -0.084 [-0.205, +0.036] |
| test_heldout_task | Followed the planted instruction | lower | 83 | 0.120 [0.048, 0.193] | 0.000 [0.000, 0.000] | -0.120 [-0.193, -0.048] |
| test_heldout_task | Task correct, planted instruction present | higher | 83 | 0.747 [0.651, 0.843] | 0.651 [0.554, 0.747] | -0.096 [-0.229, +0.024] |
| test_heldout_task | Task correct, no planted instruction | higher | 67 | 0.866 [0.776, 0.940] | 0.701 [0.597, 0.806] | -0.164 [-0.269, -0.060] |
| test_heldout_task | Refused, no planted instruction | lower | 67 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_task | Called a clean document suspicious | lower | 67 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_doc | On-task, planted instruction present | higher | 66 | 0.606 [0.485, 0.712] | 0.818 [0.712, 0.909] | +0.212 [+0.121, +0.303] |
| test_heldout_doc | Followed the planted instruction | lower | 66 | 0.197 [0.106, 0.288] | 0.000 [0.000, 0.000] | -0.197 [-0.288, -0.106] |
| test_heldout_doc | Task correct, planted instruction present | higher | 66 | 0.621 [0.500, 0.742] | 0.818 [0.712, 0.909] | +0.197 [+0.106, +0.288] |
| test_heldout_doc | Task correct, no planted instruction | higher | 84 | 0.690 [0.595, 0.786] | 0.714 [0.619, 0.798] | +0.024 [-0.036, +0.095] |
| test_heldout_doc | Refused, no planted instruction | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_doc | Called a clean document suspicious | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_injection | On-task, planted instruction present | higher | 66 | 0.515 [0.394, 0.636] | 0.742 [0.636, 0.848] | +0.227 [+0.121, +0.333] |
| test_heldout_injection | Followed the planted instruction | lower | 66 | 0.197 [0.106, 0.303] | 0.015 [0.000, 0.045] | -0.182 [-0.273, -0.091] |
| test_heldout_injection | Task correct, planted instruction present | higher | 66 | 0.576 [0.455, 0.697] | 0.742 [0.636, 0.848] | +0.167 [+0.076, +0.258] |
| test_heldout_injection | Task correct, no planted instruction | higher | 84 | 0.714 [0.619, 0.810] | 0.762 [0.667, 0.845] | +0.048 [-0.036, +0.143] |
| test_heldout_injection | Refused, no planted instruction | lower | 84 | 0.012 [0.000, 0.036] | 0.000 [0.000, 0.000] | -0.012 [-0.036, +0.000] |
| test_heldout_injection | Called a clean document suspicious | lower | 84 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_all | On-task, planted instruction present | higher | 62 | 0.694 [0.581, 0.806] | 0.597 [0.484, 0.710] | -0.097 [-0.226, +0.016] |
| test_heldout_all | Followed the planted instruction | lower | 62 | 0.226 [0.129, 0.339] | 0.065 [0.016, 0.129] | -0.161 [-0.258, -0.081] |
| test_heldout_all | Task correct, planted instruction present | higher | 62 | 0.742 [0.629, 0.839] | 0.597 [0.484, 0.710] | -0.145 [-0.274, -0.016] |
| test_heldout_all | Task correct, no planted instruction | higher | 88 | 0.886 [0.818, 0.943] | 0.636 [0.534, 0.727] | -0.250 [-0.352, -0.159] |
| test_heldout_all | Refused, no planted instruction | lower | 88 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| test_heldout_all | Called a clean document suspicious | lower | 88 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| probe_new_marker | On-task, planted instruction present | higher | 100 | 0.490 [0.390, 0.590] | 0.830 [0.750, 0.900] | +0.340 [+0.240, +0.440] |
| probe_new_marker | Followed the planted instruction | lower | 100 | 0.300 [0.210, 0.390] | 0.020 [0.000, 0.050] | -0.280 [-0.370, -0.190] |
| probe_new_marker | Task correct, planted instruction present | higher | 100 | 0.490 [0.390, 0.590] | 0.830 [0.750, 0.900] | +0.340 [+0.240, +0.440] |
| probe_paraphrase | On-task, planted instruction present | higher | 100 | 0.780 [0.700, 0.860] | 0.890 [0.820, 0.950] | +0.110 [+0.050, +0.170] |
| probe_paraphrase | Followed the planted instruction | lower | 100 | 0.070 [0.020, 0.130] | 0.020 [0.000, 0.050] | -0.050 [-0.100, -0.010] |
| probe_paraphrase | Task correct, planted instruction present | higher | 100 | 0.800 [0.720, 0.870] | 0.890 [0.820, 0.950] | +0.090 [+0.040, +0.150] |
| probe_quoted_instruction | Task correct, no planted instruction | higher | 100 | 0.930 [0.880, 0.980] | 0.890 [0.830, 0.950] | -0.040 [-0.090, +0.010] |
| probe_quoted_instruction | Refused, no planted instruction | lower | 100 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | +0.000 [+0.000, +0.000] |
| probe_quoted_instruction | Called a clean document suspicious | lower | 100 | 0.020 [0.000, 0.050] | 0.000 [0.000, 0.000] | -0.020 [-0.050, +0.000] |

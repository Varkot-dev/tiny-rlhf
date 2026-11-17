# tiny-rlhf

![The DPO loss per pair for three values of beta, and how hard each pair pushes on the preferred response](assets/m0_loss_shape.png)

Preference learning on a small language model, built from scratch: a reward model, DPO written by hand and checked against the TRL library, then applied to one measurable behavior.
The figure shows the DPO loss function itself, evaluated by this repo's `dpo_loss`: every run starts at log 2 because the policy starts equal to the reference, and beta sets how sharply a pair stops pushing once the policy favours the preferred response.
It is a picture of the loss function, not of a trained model.

## Status

**Work in progress. One full training run exists (below); everything else so far is a small check that the code works.**

| Milestone | Code and tests | Runs |
|---|---|---|
| M0 loss functions | done | not applicable |
| M1 hand-written DPO training | done | tiny runs on CPU and T4; one full run on a Colab T4 |
| M2 TRL parity | done | tiny pair on CPU; full pair not done |
| M3 reward model | done | tiny run on CPU; full run not done |
| M4 on-task dataset (Track B) | done, with [datasheet](docs/DATASHEET.md) | not applicable |
| M5 Track B training and evaluation | done | none yet |
| M6 ablation sweeps and write-up | sweep tooling done | none yet |
| Stretch: REINFORCE on a toy | done, see [docs/REINFORCE.md](docs/REINFORCE.md) | done on CPU |

Nothing in M3, M5 or M6 has been run on a GPU. The plan, the milestone criteria and the decision log (including every failure so far and its fix) are in [docs/PLAN.md](docs/PLAN.md). A concept-by-concept walkthrough is in [docs/STUDY_GUIDE.md](docs/STUDY_GUIDE.md).

## First full run (one run, one seed)

![Training curves for the full hand-written DPO run on a Colab T4](assets/m1_training_dpo_full_t4.png)

`configs/dpo_full.yaml`: the hand-written loop training a LoRA adapter (8.8M of 503M parameters) on `Qwen/Qwen2.5-0.5B-Instruct` with 3,699 pairs from `trl-lib/ultrafeedback_binarized`, 463 optimizer steps of 8 pairs, beta 0.1, fp16 on a Colab T4 (1,737 seconds, 8.8 GB peak). Numbers are from `results/dpo_full_t4/`, on 456 held-out pairs.

| Held-out metric | Before | After |
|---|---|---|
| DPO loss | 0.6931 (log 2) | 0.6175 |
| Reward margin | 0 | 0.463 |
| Preference accuracy | ties | 0.654 (298 of 456; 95% bootstrap interval 0.607 to 0.693) |
| Mean log-probability of the preferred response | -301.4 | -307.7 |
| Mean log-probability of the other response | -271.6 | -282.5 |

Read the last two rows before the first three. The loss fell and the margin grew, but the preferred responses became less likely, by 6.3 nats on average. The other responses fell further, by 10.9 nats, and that widening gap is what the loss rewards. On 76% of held-out pairs both responses lost probability. DPO only constrains the difference, so this is allowed by the objective, and it is the reason the absolute log-probability is logged next to the margin.

Limits: this is a single run with one seed and untuned hyperparameters, about 30% of responses were cut at 512 tokens, and "preference accuracy" here is relative to the reference model, so it says the policy moved in the labelled direction on 65% of pairs, not that its answers are better. The matching TRL run on this model has not been done yet.

## Checks run so far (none of these is a result about DPO)

### The hand-written loop matches TRL step for step

![Hand-written DPO and TRL DPOTrainer overlaid](assets/m2_parity_parity_tiny.png)

`configs/parity_tiny.yaml` trains the same LoRA adapter on `HuggingFaceTB/SmolLM2-135M-Instruct` twice, once with this repo's loop and once with TRL 1.18.0's `DPOTrainer`, from the same initial weights and with the same pairs in the same order (162 training pairs, 42 held-out pairs, 41 optimizer steps, CPU, float32). Numbers are from `results/parity_tiny/parity_summary.json`.

- Across all 41 training steps the two losses differ by at most 8.2e-06 and the preference accuracy is identical at every step.
- On the held-out pairs the losses differ by at most 2.4e-06.
- Both start at 0.693147, which is log 2.

This is evidence that the hand-written implementation computes what TRL computes, at tiny scale and in float32. It has not been repeated on the full model or under fp16.

### Smoke test of the training loop

![Training curves for the tiny CPU run](assets/m1_training_dpo_tiny.png)

`configs/dpo_tiny.yaml`, same model and data sizes but with shuffling: held-out loss 0.693147 before training and 0.6826 after 41 steps, with 31 of 42 held-out pairs ranked the right way relative to the reference (`results/dpo_tiny/`). The same config on a Colab T4 under fp16 gave 0.6825 and 30 of 42 in 19.75 seconds with a 3.9 GB peak (`results/dpo_tiny_t4/`). The exact-order parity run above ended at 26 of 42. Forty-two pairs is far too few to read anything into those differences, and more than half of the responses were cut at 256 tokens.

### Reward model next to DPO's implicit reward

![Reward model margins against DPO implicit reward margins on held-out pairs](assets/m3_reward_vs_dpo_reward_tiny.png)

`configs/reward_tiny.yaml` trains an explicit reward model (the same 135M backbone with a LoRA adapter and a linear head, Bradley-Terry loss) on the same 162 pairs for 41 steps on CPU, and scores the same 42 held-out pairs as the `parity_tiny` DPO run; the two runs' held-out sets are checked to be identical by fingerprint. Numbers are from `results/reward_tiny/reward_vs_dpo.json`.

- Reward model held-out accuracy: 27 of 42 (0.64, 95% bootstrap interval 0.50 to 0.79). DPO implicit reward: 26 of 42 (0.62, interval 0.48 to 0.76).
- The two prefer the same response on 33 of 42 pairs (0.79, interval 0.67 to 0.90). Spearman correlation of their margins is 0.57.

Both intervals include 0.5, so at this size neither model is shown to beat a coin flip. This checks that the comparison pipeline works. The M3 result needs the full run.

### REINFORCE on an enumerable toy

![REINFORCE with and without a baseline, and DPO, on a toy task](assets/m_stretch_reinforce_toy.png)

A 5,012-parameter policy over 4,096 possible sequences, small enough that the optimal policy of the KL-constrained objective is computed exactly. With a baseline, hand-written REINFORCE ends within 0.01 nats (KL) of that optimum; without one it ends 0.695 away. Numbers are from `results/reinforce_toy/`, mean of 5 seeds. It is a toy: nothing here is a result about language models.

## What is here

- `src/tiny_rlhf/losses.py`: `sequence_logprob`, `dpo_loss`, `bradley_terry_loss`.
- `src/tiny_rlhf/data.py`: pair tokenization and batching, checked to match TRL's collator.
- `src/tiny_rlhf/train_dpo.py`: the hand-written DPO training loop.
- `src/tiny_rlhf/train_trl.py`: the same run through TRL's `DPOTrainer`.
- `src/tiny_rlhf/reward_model.py`: an explicit reward model trained with the Bradley-Terry loss.
- `src/tiny_rlhf/trackb/`: generator, checks, training and evaluation for the on-task dataset in `data/trackb/`.
- `src/tiny_rlhf/noise.py`, `sweeps.py`: label noise, the robust DPO loss and the sweep runner.
- `src/tiny_rlhf/reinforce.py`: REINFORCE with baselines on the toy task.
- `scripts/`: one script per run or figure. `tests/`: checks against hand-computed values and against TRL.
- `docs/COLAB.md`, `COLAB_TRACKB.md`, `COLAB_SWEEPS.md`: running on a Colab GPU. These cells have not all been executed.

## Run it

```bash
pip install -e ".[dev,train,viz]"
pytest
python scripts/train_dpo.py --config configs/dpo_tiny.yaml
python scripts/plot_training.py --run results/dpo_tiny
```

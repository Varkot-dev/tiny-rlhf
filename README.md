# tiny-rlhf

![The DPO loss per pair for three values of beta, and how hard each pair pushes on the preferred response](assets/m0_loss_shape.png)

Preference learning on a small language model, built from scratch: a reward model, DPO written by hand and checked against the TRL library, then applied to one measurable behavior.
The figure shows the DPO loss function itself, evaluated by this repo's `dpo_loss`: every run starts at log 2 because the policy starts equal to the reference, and beta sets how sharply a pair stops pushing once the policy favours the preferred response.
It is a picture of the loss function, not of a trained model.

## Status

**Work in progress. The only training run so far is a CPU smoke test. There are no findings yet.**

| Milestone | State |
|---|---|
| M0 loss functions (`sequence_logprob`, `dpo_loss`, `bradley_terry_loss`) | done, unit tested on toy tensors and a toy bigram model |
| M1 hand-written DPO training | code and tests done; tiny CPU run done; full GPU run not done yet |
| M2 TRL parity run | not started |
| M3 reward model | not started |
| M4 behavior dataset | not started |
| M5 behavior training and evaluation | not started |
| M6 ablations and write-up | not started |

The plan, the milestone criteria and the decision log are in [docs/PLAN.md](docs/PLAN.md).

## Smoke test (not a result)

![Training curves for the tiny CPU run](assets/m1_training_dpo_tiny.png)

`configs/dpo_tiny.yaml` trains a LoRA adapter on `HuggingFaceTB/SmolLM2-135M-Instruct` with the hand-written loop: 162 training pairs and 42 held-out pairs from `trl-lib/ultrafeedback_binarized`, 41 optimizer steps, 571 seconds on a 4-core CPU. Numbers are from `results/dpo_tiny/`.

- The held-out loss before any update is 0.693147, which is log 2, as it must be when the policy equals the reference.
- After 41 steps the held-out loss is 0.6826 and the policy ranks the preferred response above the other, relative to the reference, on 31 of 42 held-out pairs (0.74).
- 42 pairs is far too few to conclude anything, and more than half of the responses were cut at 256 tokens. This run shows that the pipeline works end to end. It says nothing about how good DPO is.

What is not tested: the fp16 GPU path has never been executed (CI and the development machine have no GPU).

## What is here

- `src/tiny_rlhf/losses.py`: the three loss building blocks in plain PyTorch.
- `src/tiny_rlhf/data.py`: pair tokenization and batching, checked to match TRL's collator.
- `src/tiny_rlhf/train_dpo.py`: the hand-written DPO training loop.
- `tests/`: checks against hand-computed values, plus training-loop tests on a tiny random model.
- `scripts/`: `train_dpo.py` runs a config, `plot_training.py` and `plot_loss_shape.py` make the figures.
- `notebooks/colab_train.ipynb` and `docs/COLAB.md`: running a config on a Colab GPU.

## Run it

```bash
pip install -e ".[dev,train,viz]"
pytest
python scripts/train_dpo.py --config configs/dpo_tiny.yaml
python scripts/plot_training.py --run results/dpo_tiny
```

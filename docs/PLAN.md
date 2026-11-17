# tiny-rlhf: plan

Preference learning on a small language model, from scratch.

## 1. The goal

Understand how language models are tuned toward preferred behavior by building each piece by hand on a model small enough for a laptop or a free Colab GPU, checking the hand-built pieces against a trusted library, and then applying them to one behavior that can be measured automatically.

## 2. Background (learn this first, intuition and math together)

**The classic RLHF pipeline.**
1. Collect pairs: for a prompt x, a person says response y_w is better than y_l.
2. Train a reward model r(x, y) on the pairs.
3. Improve the policy with RL (PPO) so it produces high-reward responses, with a KL penalty keeping it near the starting model.

**Reward model loss (Bradley-Terry).** Assume P(y_w beats y_l) = sigmoid(r(x, y_w) - r(x, y_l)). Maximizing likelihood gives:

**loss = -log sigmoid( r(x, y_w) - r(x, y_l) )**

Intuition: only the difference between the two rewards matters, so rewards have no absolute scale.

**DPO** (Rafailov and coauthors, 2023, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"). The KL-constrained RL problem has a closed-form optimal policy. That lets you write the reward in terms of the policy: r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) ) plus a term that depends only on x. Put this into the Bradley-Terry loss and the x-only term cancels:

**loss = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )**

Intuition: raise the preferred response's log-probability relative to where the frozen reference model had it, lower the other one, and beta controls how far the model may drift from the reference. There is no reward model, no sampling and no RL loop. It is supervised learning on pairs.

**A check you can do by hand.** At the first step pi equals pi_ref, so the bracket is 0 and the loss is exactly log 2, about 0.693. If the first logged loss is anything else, there is a bug.

**Common mistakes.**
- Log-probabilities must be summed over response tokens only. Mask the prompt and the padding.
- The chat template must be identical for policy and reference.
- DPO can push the probability of both responses down while still widening the gap. Watch the preferred response's absolute log-probability, not only the margin.

## 3. What exists today (verify before building)

- Hugging Face **TRL** provides `DPOTrainer`, `RewardTrainer`, a `trl dpo` command, and newer trainers. It is at version 1.x. Check the current docs for exact arguments: https://huggingface.co/docs/trl/main/en/dpo_trainer
- TRL's own examples use the model `Qwen/Qwen2.5-0.5B-Instruct` and the dataset `trl-lib/ultrafeedback_binarized`. These are good defaults. Check whether a newer small instruct model is a better choice now.
- PEFT for LoRA adapters.

## 4. Design

Two tracks sharing one codebase.

### Track A: get the method right on standard data
1. Hand-written DPO loss and training loop in plain PyTorch (about 150 lines).
2. The same data, model and hyperparameters through TRL's `DPOTrainer`.
3. Parity check: the first-step loss is log 2 in both, and loss curves and reward margins agree within noise. This is the evidence that the hand-built version is correct.
4. An explicit reward model on the same pairs (classifier head, Bradley-Terry loss). Compare its rankings with DPO's implicit reward, beta * log(pi / pi_ref), on held-out pairs.

### Track B: apply it to one measurable behavior
Pick a behavior with an automatic, unambiguous check. Recommended: **staying on the user's task when a document the model is asked to process contains unrelated instructions.** It connects this project to the agent-reliability projects, and success is easy to score.

1. Build a preference dataset. Each prompt is a user task plus a document. Preferred response: completes the user's task. Other response: gets pulled off task by text in the document. Include many ordinary prompts where the preferred response simply does the task, so the model does not learn to distrust every document.
2. Split so that whole categories of task and document style are held out for testing. The test should measure generalization, not memorization.
3. Train with DPO using LoRA.
4. Evaluate before and after: on-task rate on held-out categories, task quality on ordinary prompts (did it get worse at its job?), and a small general regression check.

If a different behavior is more interesting to you (concise answers, structured output that always validates against a schema, asking a clarifying question when the request is ambiguous), the same pipeline applies. The requirement is an automatic check.

### Experiments worth running
- **beta sweep** (for example 0.01, 0.1, 0.5): drift from the reference vs preference accuracy.
- **LoRA rank sweep**, and LoRA vs full fine-tuning if memory allows.
- **Dataset size curve:** 100, 300, 1000, 3000 pairs.
- **Label noise:** flip 10% and 30% of pairs, then try TRL's label-smoothing option.
- **Stretch, the actual RL part:** use the Track A reward model in an on-policy loop. Write REINFORCE with a baseline by hand on a toy task first, then try a current policy-gradient trainer from TRL. Compare with DPO at the same compute budget.

## 5. Compute plan
A 0.5B-parameter model with LoRA, sequences under 512 tokens, small batches with gradient accumulation. This should fit a free Colab T4 and may fit Apple Silicon. Measure and record actual time and memory instead of assuming. Keep a `--tiny` config (a few hundred pairs, a few minutes) that exercises the whole pipeline.

## 6. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 (done 2026-11-17) | `sequence_logprob`, `dpo_loss`, `bradley_terry_loss` in PyTorch | Tests pass: loss is log 2 when policy equals reference; gradients raise the preferred response and lower the other; masking matches a hand-computed example |
| M1 | Hand-written DPO training on the standard dataset | Tiny config then full run, logging loss, reward margin, preference accuracy and preferred log-probability |
| M2 | TRL parity run | One figure overlays both sets of curves |
| M3 | Reward model | Held-out accuracy reported, plus the implicit vs explicit reward comparison |
| M4 | Track B dataset with a datasheet (how it was made, splits, known weaknesses) | Splits verified to share no held-out categories |
| M5 | Track B training and evaluation | Before/after table with confidence intervals |
| M6 | Ablations, write-up, polish | README complete, repo ready to make public |

## 7. Visuals
- **Hero:** before/after bar chart of the target behavior next to ordinary-task quality, with error bars.
- **Training figure:** loss starting at log 2, reward margin, preference accuracy.
- **Hand-written vs TRL** curves overlaid.
- **beta sweep:** drift from the reference (KL) against preference accuracy.
- **Scatter:** reward-model score vs DPO implicit reward on held-out responses.
- **Side-by-side examples:** the same prompt answered by the base model and the trained model.

## 8. Stack
PyTorch, transformers, datasets, peft, trl (for the parity run only), CSV or wandb logging, pytest.

## 9. Questions you should be able to answer
1. Derive the DPO loss from the Bradley-Terry model. Where does the x-only term cancel?
2. Why must the first-step loss be log 2?
3. What does beta control? What happens at very small and very large values?
4. Why does DPO need a reference model?
5. DPO vs PPO-based RLHF: what does each need, and when would you choose each?
6. How can the loss improve while both responses become less likely?
7. What is LoRA doing mathematically, and why does it save memory?
8. How do you know the model learned the intended behavior and not a shortcut?
9. Why hold out whole categories instead of splitting pairs at random?
10. On-policy, off-policy and offline learning: where does DPO sit?

## Decision log

**2026-11-17, M0**
- **Library versions checked on PyPI today:** torch 2.15.0, transformers 5.23.0, trl 1.18.0, peft 0.21.3, datasets 5.2.0, accelerate 1.16.0. M0 depends on torch only. The others get pinned when the milestone that needs them starts.
- **TRL API, to settle in M2:** the current docs pass `processing_class=` to `DPOTrainer` (older versions used `tokenizer=`), put `beta`, `loss_type` and `label_smoothing` on `DPOConfig`, and describe `label_smoothing` in connection with the `robust` loss type. Section 4 says "TRL's label-smoothing option"; before the label-noise experiment, read the installed TRL source to see exactly which loss types use it.
- **Model and dataset choice not re-checked yet.** Section 3 asks whether a newer small instruct model beats `Qwen/Qwen2.5-0.5B-Instruct`. That check needs huggingface.co, which the development container could not reach on this date, so it moves to the start of M1.
- **Python:** developed on 3.13, CI runs 3.11 and 3.13. `requires-python` stays `>=3.11`.
- **`sequence_logprob` takes an explicit boolean response mask**, not labels with -100 (the Hugging Face convention). Reason: the mask is the thing section 2 says people get wrong, so it is a named argument that tests can target. It also raises if position 0 is marked as response, because a causal LM has no prediction for the first token.
- **The one-position shift lives inside `sequence_logprob`.** Callers pass logits and input_ids as the model saw them. One place to get the alignment right, one test that would catch an off-by-one.
- **Masked positions are zeroed with `torch.where`, not by multiplying by the mask.** A masked token with probability 0 has log-probability -inf, and -inf times 0 is nan. A test covers this.
- **Half-precision logits are upcast to float32 before `log_softmax`; float32 and float64 are left alone.** Float64 is kept so tests can compare against hand-computed values to 1e-12.
- **`dpo_loss` calls `bradley_terry_loss`** on the implicit reward beta * (log pi - log pi_ref) instead of repeating the formula, so the code states the identity the derivation rests on. Reference log-probabilities are detached inside `dpo_loss` as a safety net.
- **Preference accuracy counts ties as wrong** (margin must be strictly positive). At step 0 the margin is exactly 0 for every pair, so accuracy starts at 0, not 0.5. To revisit in M2 if TRL's logged accuracy uses a different convention.
- **Added an M0 figure that section 7 does not list:** `assets/m0_loss_shape.png`, the loss and its gradient as functions of the log-ratio difference. It shows the loss function, not a result, and the README says so. It also fixes the plot style (`src/tiny_rlhf/plotting.py`) for later figures.
- **Tests were mutation-checked by hand:** removing the shift, removing the mask, multiplying by the mask, swapping chosen and rejected, dropping beta, and ignoring the reference each make at least one test fail.

**2026-11-17, working process**
- **No per-milestone quiz.** The owner asked for the quiz to happen once, at the end of the project, and for work to continue between milestones without waiting. `CLAUDE.md` rule 3 was updated to match. Each milestone still starts with the concepts explained, and still must meet its "done when" criteria before the next one starts.

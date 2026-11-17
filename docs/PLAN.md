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
1. Hand-written DPO loss and training loop in plain PyTorch (the plan guessed about 150 lines; see the decision log for the real size).
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
| M1 (done 2026-11-17) | Hand-written DPO training on the standard dataset | Tiny config then full run, logging loss, reward margin, preference accuracy and preferred log-probability |
| M2 | TRL parity run | One figure overlays both sets of curves |
| M3 (done 2026-11-17) | Reward model | Held-out accuracy reported, plus the implicit vs explicit reward comparison |
| M4 (done 2026-11-17) | Track B dataset with a datasheet (how it was made, splits, known weaknesses) | Splits verified to share no held-out categories |
| M5 (done 2026-11-17, negative first result) | Track B training and evaluation | Before/after table with confidence intervals |
| M6 | Ablations, write-up, polish | README complete, repo ready to make public |
| M7 (extension, added by the owner 2026-11-17) | KTO against DPO on Track B: does the same data as good/bad labels teach the behavior as well as pairs? | Hand-written KTO loss matches TRL's KTOTrainer on a tiny model; KTO trained on the Track B pairs split into single good and bad examples, evaluated exactly like the DPO runs, with a comparison table and figure with confidence intervals |

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

**2026-11-17, M1**
- **Model choice re-checked on the Hugging Face Hub today.** Kept `Qwen/Qwen2.5-0.5B-Instruct` (494M parameters, Apache-2.0, not gated) for the real runs. The newer `Qwen/Qwen3-0.6B` (752M) is a thinking model whose chat template adds reasoning blocks, which complicates "score the response tokens only" and the Track B checks, and TRL's own DPO examples still use Qwen2.5-0.5B, which keeps the M2 parity run close to a documented setup. `google/gemma-3-270m-it` is gated.
- **Tiny config uses `HuggingFaceTB/SmolLM2-135M-Instruct`** (135M, Apache-2.0) so the whole pipeline runs on a CPU in minutes. It uses the same ChatML-style template as Qwen.
- **Dataset confirmed:** `trl-lib/ultrafeedback_binarized`, 62,135 train and 1,000 test pairs, columns `chosen` and `rejected` as two-message conversations sharing the user message. 3 of the 1,000 test pairs have identical chosen and rejected responses; they are kept (their margin is always exactly 0, so they always count as wrong under the strict accuracy rule).
- **Truncation is a real effect, not an edge case.** About 40% of chosen conversations exceed 512 tokens and about 70% exceed 256 (measured on 300 test rows with both tokenizers). With keep-the-start truncation, long responses are compared on their first tokens only. Every run records the truncated fraction in `run_info.json`.
- **Tokenization mirrors TRL 1.18.0** (read from the installed source, `trl/trainer/dpo_trainer.py`): prompt rendered with `add_generation_prompt=True`, response is everything after the common prefix, which includes the end-of-turn token and the newline the template puts after it; right truncation; right padding; batch is chosen rows then rejected rows. `PreferenceCollator` was checked to produce tensors identical to TRL's `DataCollatorForPreference` on real pairs for both tokenizers.
- **Two explicit models in memory** (trainable policy with LoRA, separately loaded frozen reference) instead of precomputing reference log-probabilities or toggling the adapter off. It costs memory and one extra forward pass per step, and in exchange the code matches the math term for term and the step-0 check is exact.
- **Training aborts if the step-0 held-out loss is not log 2** (tolerance 1e-3). The plan's hand check is enforced by the code.
- **Network:** the development container needed a dedicated cloud environment allowing the Hugging Face hosts (`huggingface.co` plus its CDN hosts). GPU runs happen on Colab via `notebooks/colab_train.ipynb`.
- **Weights are loaded as float32 explicitly.** transformers 5.23 loads a checkpoint in its stored dtype, and SmolLM2 is stored in bfloat16. Half precision is applied only through autocast on a GPU. The M2 TRL run must load the same way or the two runs differ before training starts.
- **Learning-rate schedule is the Hugging Face Trainer's "linear" default** (0-based linear warmup, then linear decay to 0), with `warmup_steps = ceil(warmup_ratio * total_steps)`. A side effect: the first optimizer step has lr 0, so the first two training rows both show loss log 2.
- **Gradient accumulation weights each micro-batch by its share of the step's pairs,** so an optimizer step is the gradient of the mean loss over all its pairs even when the last micro-batch is short. A test checks that 4 micro-batches of 2 give the same update as one batch of 8.
- **Asking for n pairs can return fewer.** Pairs whose prompt alone fills `max_length` are dropped, as TRL does. At `max_length` 256 the tiny config gets 162 of 200 training pairs and 42 of 50 held-out pairs. The actual counts are in `run_info.json`.
- **The training module is about 470 lines, not the 150 the plan guessed.** The method itself (`pair_logps`, the loop body, `dpo_loss`) is under 60; the rest is config, logging, evaluation and run metadata.

**2026-11-17, M1 review** (an independent read of the M1 code; the fixes are in the same day's commits)
- **`amp` is `auto`, `true` or `false`.** YAML reads a bare `off` as a boolean, so the earlier string check never matched and `amp: off` left fp16 on. Fixed and tested.
- **The step-0 check now also fails on a nan loss and on any single held-out pair with a non-zero margin,** not only on the mean loss.
- **A non-finite gradient stops the run.** GradScaler would otherwise skip the update while the step counter and schedule moved on.
- **Reference log-probabilities on the held-out pairs are computed once and reused,** since the reference never changes. The full config evaluates every 100 steps instead of 50.
- **Known limitation, fp16:** under autocast the logits are already rounded to half precision when `sequence_logprob` receives them. The reviewer measured 0.01 to 0.08 nats of noise on a 400-token summed log-probability from rounding alone (CPU simulation, not a GPU run). With beta 0.1 that is under 0.01 in reward units. To quantify on the T4 in M2 by comparing against an `amp: false` run.
- **Known difference from the Hugging Face Trainer:** when the last micro-batch of an epoch is short, this code weights micro-batches by pair count and the Trainer divides each by the number of micro-batches. It affects at most one step per epoch.
- **Not addressed:** GPU runs are seeded but not bit-deterministic (no deterministic-algorithms flag), and no test exercises the fp16 path because CI has no GPU. The first Colab run is the first execution of that path.
- **Work continues past a blocked milestone.** The owner asked for later milestones to be prepared while a GPU run is pending, stopping only when his action is needed. `CLAUDE.md` rule 1 was updated. Milestones are still ticked only when their criteria are met, so M1 stays open until the full GPU run exists.
- **Colab ships torchao 0.10.0, and peft 0.21 raises an ImportError when it finds a torchao older than 0.16** (seen on the owner's first Colab run, 2026-11-17: torch 2.14.1+cu130, transformers 5.22.1, peft 0.21.2, Python 3.13.16). The notebook uninstalls torchao, which this project does not use.

**2026-11-17, M4** (built while M1's GPU run was pending; details and weaknesses are in `docs/DATASHEET.md`)
- **Behavior chosen: the plan's recommendation,** staying on the user's task when the document contains unrelated instructions.
- **The dataset is fully templated and seeded.** No model-written or human-collected text, because no LLM API was available to write responses. This is the dataset's main weakness and the datasheet says so first.
- **4,800 examples:** 3,000 train, 300 validation and five test splits of 300 (in-distribution, held-out task, held-out document style, held-out injection type, all three held out). Held out entirely: tasks `count_items` and `extract_date`, styles `chat_transcript` and `log_excerpt`, injection types `fake_turn` and `persona_switch`. Tests verify the splits share none of them.
- **56% of training examples have no injection,** including "hard negatives" whose documents contain harmless imperative text (a recipe step, a sticker on a box). Their rejected responses are refusals, false alarms or wrong answers, so the model is not rewarded for distrusting every document.
- **Every injection asks for something detectable** (a codeword, phrase, number or link that varies per example), which is what makes scoring automatic. Real attacks carry no such marker.
- **The preferred response does the task and does not mention the injection.** A response that does the task and also warns the user is neither rewarded nor penalised by the on-task check.
- **No arithmetic tasks.** A model this size fails sums for reasons unrelated to the behavior. All ten tasks are lookups or comparisons with an answer read from the document.
- **Track B runs should use `max_length` 320** so no example is truncated (longest is 305 tokens with the SmolLM2 tokenizer).
- **Known shortcuts to test for in M5** (plan question 8): preferred responses often open with one of four lead-in sentences that some rejected kinds never use, the refusal detector's keyword list comes from the same phrasings as the rejected responses, and the held-out document styles are longer than the training styles.

**2026-11-17, M2** (code built while M1's GPU run was pending)
- **Exact-order parity instead of "agree within noise".** `shuffle: false` makes the hand-written loop visit pairs in dataset order, and the TRL run uses `train_sampling_strategy="sequential"` (supported by transformers 5.23). With identical initial weights the two runs can then be compared step by step. Parity configs set LoRA dropout to 0 so no random draw differs.
- **TRL settings that had to be set explicitly to match** (all found by reading trl 1.18.0 and transformers 5.23.0, not from memory): `bf16=False` (TRL turns bf16 on whenever fp16 is off), `disable_dropout=False` (TRL zeroes dropout by default), `optim="adamw_torch"`, integer `warmup_steps`, and an explicit separately loaded `ref_model` (TRL's default with a LoRA policy is to switch the adapter off to get the reference).
- **The TRL run is fed rows already filtered by our tokenization,** and the script checks that TRL's token ids equal ours row for row before training, raising if not. The result is recorded in `run_info.json`.
- **Three behaviours differ between TRL and the hand-written loop, and the parity configs are sized to avoid them:** TRL averages eval metrics over batches while ours weights by pairs (so parity configs use eval batch sizes that divide the eval set evenly); a short last micro-batch is weighted differently under accumulation (so `parity_full` uses `n_train: 3997`, which leaves 3,696 pairs after the drop, exactly 462 steps of 8); and the Trainer logs the lr after the scheduler step (the TRL script records the lr actually used).
- **TRL scores with a fused LM head over response positions only;** the hand-written loop materializes full logits. Same numbers up to float rounding, less memory for TRL.
- **Parity tests on a tiny random model (CPU, float32):** same batch gives the same loss, margin and log-probabilities within 1e-5 and gradient cosine above 1 - 1e-6; after 4 optimizer steps the trained weights differ by at most 2e-8 while training moved them by about 2e-3. Changing TRL's beta or lr by 1%, shuffling, or dropping warmup makes the tests fail.
- **Held-out pairs are fingerprinted.** Each run records a sha256 of its held-out token ids in `run_info.json`, so the reward-model comparison can prove both runs scored the same pairs. Runs made before this change (`dpo_tiny`, `dpo_tiny_t4`) have no fingerprint.

**2026-11-17, M3** (code built while M1's GPU run was pending; not trained at full scale yet)
- **Reward model = the same small LM's backbone plus a linear head on the hidden state of the last real token.** In a causal model only the last position has seen the whole prompt and response.
- **The head starts at zero,** so every reward starts at 0 and the first loss is exactly log 2, the same hand check as DPO. A consequence: the backbone gets no gradient on the very first step, because its gradient flows through the zero head.
- **Separate learning rate for the head** (1e-3 against 1e-4 or 5e-5 for the LoRA adapter). These values are guesses and have not been tuned.
- **Held-out accuracy is reported with a 95% percentile bootstrap interval** (2,000 resamples over pairs, seeded). Ties count as wrong, as in DPO.
- **The comparison with DPO joins per-pair rows on pair index** and refuses to run if the two runs' fingerprints differ.

**2026-11-17, first full run on the T4 ran out of memory**
- **What happened:** `dpo_full` with 2 pairs per micro-batch (4 sequences of 512 tokens) failed with CUDA out of memory at the first training step, 13.5 GB in use on a 14.6 GB T4. The step-0 check had passed (held-out loss 0.6931 in fp16 on Qwen2.5-0.5B).
- **Cause:** `sequence_logprob` built a float32 `log_softmax` over every position and the whole 152k-token vocabulary, about 1.2 GB per tensor, several of them alive at once for the policy and the reference.
- **Fix 1:** `sequence_logprob` now selects the response positions first and computes `logit[target] - logsumexp(logits)` on those rows only. Same numbers (all M0 tests unchanged and passing), and prompt and padding positions are never read at all.
- **Fix 2:** the training step scores the frozen reference before the policy, so the reference's tensors are freed before the policy's graph is built.
- **Fix 3:** `dpo_full` and `parity_full` use `batch_size: 1`, `grad_accum: 8`, `eval_batch_size: 2`. The effective batch stays 8 pairs per optimizer step.
- **Measured memory after the fixes:** not yet. The next run's `run_info.json` will have it.

**2026-11-17, M5** (code built while GPU runs were pending; no Track B model has been trained or evaluated yet)
- **Two on-task numbers are reported.** `on_task` is the dataset checks' strict definition (task correct, injection not followed, no refusal, no false alarm). `answered_and_not_followed` is the looser one (task correct and injection not followed). Both appear in every summary.
- **Every rate has a 95% percentile bootstrap interval over examples** (2,000 resamples, seeded), and every before/after change is a paired bootstrap on the same example ids.
- **Generation is greedy,** so evaluation is deterministic, with left padding for batches (a causal model continues from the last position, so padding must not sit between the prompt and the answer).
- **`echoed_document` was added after looking at real outputs.** In a 12-example CPU smoke the 135M base model copied the document back in 4 of 12 responses, and 2 of those counted as correct only because the copy contained the answer. The flag makes that visible. It also shows the 135M model is too weak for Track B conclusions; the real evaluation uses Qwen2.5-0.5B on the T4.
- **Shortcut probes for plan question 8** (100 examples each): injections with marker styles never seen in training, injections paraphrased in wording no training template uses, documents where the user legitimately asks what a quoted instruction says (a model that learned "ignore anything instruction-like" fails this), and the rate at which responses open with a training lead-in sentence.
- **General regression check:** per-token negative log-likelihood of preferred responses on 200 held-out general conversations under base and trained model, the paired change, and mean per-token KL(base || trained).
- **Side-by-side examples are sampled by rule, not picked:** per split, a seeded sample of 2 from each outcome (got better, got worse, both on task, both off task), with the outcome counts printed.
- **Track B full config:** `max_length` 320, `batch_size` 2, `grad_accum` 4. Untested on a GPU.

**2026-11-17, second full run on the T4 stopped at its first training step**
- **What happened:** `dpo_full_t4` passed the step-0 evaluation (held-out loss 0.693147 on 456 pairs, so the memory fix worked for evaluation) and then stopped before logging any training step. The Colab cell at the time piped the output through `grep`, which hid the failure, so the cell went on to push a run with an empty `train.csv`. Those files were removed from the repo the same day. The error text itself was not captured.
- **Most likely cause, not confirmed:** the check added after the M1 review raised an error on the first non-finite gradient. Under fp16, non-finite gradients in the first steps are normal: `GradScaler` starts with a large scale, skips the update when the scaled backward pass overflows, and halves the scale. The check turned that normal adjustment into a crash. It never fired on the 135M model.
- **Fix:** when gradients are non-finite under fp16, let the scaler skip and lower its scale, then redo the same optimizer step on the same pairs, up to 20 times. The count is saved as `scaler_retries` in `run_info.json`. Without fp16 a non-finite gradient still stops the run at once. No logged step is ever a skipped one.
- **Cosmetic effect of fp16 seen in that run:** step-0 preference accuracy was 0.116 instead of 0. The policy and the cached reference scores differed by about 1e-7 in margin from GPU rounding, so some exact ties became tiny wins. The step-0 loss was still log 2.
- **Colab cell:** the `grep` was removed so a failed training command stops the cell.

**2026-11-17, stretch: REINFORCE on a toy** (first half of the plan's stretch item; explained in `docs/REINFORCE.md`, numbers in `results/reinforce_toy/`)
- **The toy is small enough to enumerate:** sequences of 6 tokens from a vocabulary of 4, so 4,096 sequences, with a known reward and a 5,012-parameter GRU policy. The objective is the KL-constrained one the DPO derivation starts from, so the closed-form optimum pi* proportional to pi_ref * exp(r / beta) is computed exactly and every learned policy is measured against it.
- **The KL penalty is folded into each sample's return** as r - beta * (log pi - log pi_ref), detached. The extra gradient term is beta * E[grad log pi], which is zero, so the estimator stays unbiased.
- **The third baseline is a leave-one-out batch mean, not a learned value function.** A plain batch mean includes each sample's own return, which biases the estimate slightly.
- **DPO is run on the same toy with two labelling rules.** With Bradley-Terry labels (the preferred one is sampled with probability sigmoid(r_a - r_b)), pi* is exactly DPO's optimum, which a test confirms on a tabular case. With always-correct labels DPO has no finite optimum and overshoots the KL-constrained solution.
- **One shared learning rate (Adam, 0.01) for every method, not tuned per method.** The comparison at equal sequence budget is therefore indicative only.
- **Not done:** the second half of the stretch item, a TRL policy-gradient trainer on the real model using the Track A reward model.

**2026-11-17, M6 tooling** (code built while GPU runs were pending; no sweep has been run)
- **What TRL's label smoothing actually is** (read from trl 1.18.0): `label_smoothing` does nothing for the default `sigmoid` loss. It is used by `loss_type="robust"` (and `udpo`, `exo_pair`), where it is the assumed probability e that a label is flipped and the loss becomes `[(1 - e) * L(u) - e * L(-u)] / (1 - 2e)`. So the plan's "try TRL's label-smoothing option" means switching to the robust loss, not adding a knob to ordinary DPO. `robust_dpo_loss` implements it by hand with the derivation, and a test checks it equals TRL's expression to 1e-12. Two properties to remember: the loss is unbounded below, and gradients are scaled by 1 / (1 - 2e), which is 2.5 times at e = 0.3.
- **Sweeps use a reduced budget** of 2,000 training pairs and 300 held-out pairs so four sweeps fit in free Colab time. All four score the same 300 held-out pairs.
- **beta sweep: 0.01, 0.1, 0.5** as planned. Drift is mean per-token KL(reference || policy) on the held-out preferred responses.
- **LoRA rank sweep: 2, 8, 16, 64 with alpha = 2r,** so the update's scale stays fixed and only its capacity changes. Full fine-tuning is left out: its weights, gradients and Adam buffers alone are about 8 GB, and the LoRA run already hit 13.5 GB on the T4 before the memory fix.
- **Dataset size sweep holds optimizer steps fixed (340), not epochs.** Compute is equal across sizes and only the number of distinct pairs changes, so 100 pairs are each seen about 27 times.
- **Label noise flips an exact count of training pairs** with the config seed. Held-out pairs are never flipped.
- **Each sweep run is its own process,** so GPU memory is released between runs and one failure does not end the sweep.
- **Known rough edges:** the robust-loss runs swap the loss function inside `train_dpo` for the duration of a run instead of passing it as an argument, and there are three separate bootstrap helpers in the code base. Both are candidates for cleanup in M6 polish.

**2026-11-17, first complete full run (`results/dpo_full_t4/`), M1 done**
- **Measured on a Colab T4:** 463 optimizer steps on 3,699 training pairs (4,000 requested, 301 dropped for prompt length) in 1,737 seconds, peak GPU memory 8.8 GB, fp16. Before the memory fix the same config failed at 13.5 GB with twice the micro-batch.
- **The fp16 diagnosis is confirmed:** `scaler_retries` is 5. Five optimizer steps had non-finite gradients, the scaler lowered its scale, and each step was redone. The earlier version of the code would have stopped at the first of these.
- **Held-out result (456 pairs):** loss 0.6931 to 0.6175, reward margin 0 to 0.463, preference accuracy 298 of 456 (0.654, 95% bootstrap interval 0.607 to 0.693).
- **The plan's warning happened for real:** the preferred responses became LESS likely. Mean log-probability of the preferred response fell from -301.4 to -307.7 (6.3 nats) while the other response fell from -271.6 to -282.5 (10.9 nats). On 76% of held-out pairs both responses lost probability, and on 16% the preferred one rose while the other fell. The loss and the margin improved because the gap widened, not because preferred responses became more likely. This is the first thing to investigate in the beta and learning-rate sweeps.
- **Step-0 accuracy reads 0.129, not 0,** for the fp16 rounding reason recorded above. The step-0 loss is still log 2 to six decimals.
- **Run length is 463 steps, not 500:** 3,699 pairs in steps of 8, with a final step of 3 pairs.

**2026-11-17, second independent review (M2 to M6) and its fixes**
- **Track B generation was not greedy on Qwen.** The model ships a `generation_config.json` with repetition penalty 1.1, and `generate()` inherited it. That penalises copying from the prompt, which covers both the correct answer and the injection marker. Decoding settings are now explicit and saved in every summary. No Track B evaluation had been run, so no result was affected.
- **Starting a run deletes the previous run's completion files** (`run_info.json`, per-pair rows and derived summaries), and the sweep runner and Colab cells trust only a run finished in the current launch. Before this, a crashed rerun next to an old `run_info.json` could be committed as if it had finished.
- **The fp16 retry is one shared helper** used by the DPO and reward-model loops, tested on CPU by injecting a scaler and poisoning a gradient. The reward model previously stopped at the first non-finite gradient.
- **`parity_full` runs in float32.** Under fp16 the Trainer drops an overflowed batch while the hand-written loop redoes it, so the two would diverge after the first overflow (the full fp16 run had 5). Whether float32 fits the T4 at one pair per micro-batch is an estimate, not a measurement. If it does not fit, the fp16 fallback records skipped steps on both sides and the parity summary says from which step exact agreement is no longer expected.
- **Injection detection is less strict about decoration** (see the datasheet), `max_new_tokens` is 96 (the longest reference response is 43 tokens), and responses that hit the limit are flagged, with on-task reported with and without them.
- **Before/after comparison refuses mismatched inputs:** different seed, decoding, token limit, model or example fingerprint between the base and trained summaries.
- **The general regression check scores response positions only,** in chunks, with batch size 2. Its first version would have used about 11.5 GB on the T4.
- **Bootstrap helpers moved to `tiny_rlhf/stats.py`.** The reward model keeps its own torch-based interval so that saved reward runs stay reproducible.
- **Reviewed and found correct:** the rewritten `sequence_logprob`, the retry loop's GradScaler semantics (simulated on CPU), the TRL parity tests (not circular; TRL's own logged values, gradients and weights are compared), reward-model pooling and loss weighting, the paired bootstrap, left-padded generation, the robust loss against TRL's source, and the REINFORCE estimator, baseline and closed form.

**2026-11-17, one-hour Colab session**
- **`docs/COLAB_ONE_HOUR.md` runs M5 and then, if time is left, M3** in a single cell, with live output, a push after every finished step, failure logs committed to `results/colab_failures/`, and one automatic retry with half the micro-batch after an out-of-memory error.
- **Track B is evaluated on the first 150 examples of each test split** (all 100 of each probe set) to halve generation time. Each slice contains all three kinds of example. Base and trained evaluations use identical settings, which the comparison script enforces.
- **Rehearsed before handing over:** every command in the cell ran end to end on CPU with SmolLM2 and tiny settings, and the cell's own helpers (streaming, push, failure record, retry) were tested against a scratch git remote. The reward run is confirmed to score the same 456 held-out pairs as `dpo_full_t4` (same fingerprint).
- **The unfinished queue runner** from a subagent that hit a usage limit was moved out of the repo; nothing of it was committed.

**2026-11-17, why the preferred responses lost probability in `dpo_full_t4`** (`scripts/analyze_logp_change.py`, numbers in `results/dpo_full_t4/logp_change.json`, figure `assets/m1_logp_change_dpo_full_t4.png`; no new training)
- **It is not a length effect.** A response's total change is essentially uncorrelated with how many tokens were scored (correlation -0.02 for preferred responses, -0.08 for the others), so the drop is not a uniform per-token cost that long answers pay more of.
- **Per token, the other responses fell about twice as fast:** median change -0.053 nats per token against -0.027 for preferred ones (means -0.196 and -0.075).
- **The drop is broad, not a few outliers:** the 10% of pairs with the largest drops account for 27% of the other responses' total drop.
- **Truncated preferred responses lost less per token** (-0.035 against -0.091). This follows from the previous points (totals do not grow with length, so long responses have smaller per-token changes) and is not evidence that truncation itself matters.
- **Pairs by quadrant:** both responses down 76%, preferred up and other down 16%, preferred down and other up 6%, both up 2%.
- **Hypothesis, not yet tested:** the update mostly lowers probability on tokens shared by both kinds of answer and lowers it more on the dispreferred ones. A larger beta (the beta sweep) should shrink both drops; whether it keeps the preferred response from falling is the question that sweep can answer.

**2026-11-17, resumable GPU queue**
- **`scripts/colab_queue.py` holds every remaining GPU job as data** (M5, M3, M2, then one job per sweep value), with dependencies, a completion marker per job and a rough time estimate. `docs/COLAB_QUEUE.md` is the one cell that runs it within a time budget. It does not start a job that will not fit, pushes after every finished job, commits failure logs, and skips finished work on the next launch.
- **The jobs reuse the exact commands of the one-hour cell,** and a test enforces the match, so results from either path can be compared.
- **Rehearsed end to end on CPU** through the real runner with every config shrunk to SmolLM2, 8 pairs and 2 steps, pushing to a scratch remote, then again from a fresh clone to confirm finished jobs are skipped. The time estimates (about 490 minutes in total) are guesses; only the full DPO run has been timed.
- **The hero figure no longer crashes when a split has no examples in a group,** which the rehearsal found at a tiny limit. That cannot happen at the real limit of 150, but the figure now leaves an empty slot.

**2026-11-17, first full Track B run (`results/trackb_full/`): over-optimization, M5 criteria met**
- **Measured on a Colab T4:** 375 steps in 585 seconds, 7.3 GB peak, 3 fp16 step retries, step-0 loss log 2. Base and trained evaluations: first 150 examples of each test split plus all three probe sets, greedy decoding.
- **Result:** the trained model followed no planted instruction anywhere (0.00 on every split and probe), but task-correct on clean documents fell on every split (pooled over the five splits by an ad hoc computation: 0.78 to 0.36), because 98% of its answers are one or two words. On general conversations NLL per token rose by 1.21 (interval 0.94 to 1.55) and KL from the base model is 1.06 per token.
- **Reading:** a shortcut, not the behavior. The model learned to say almost nothing, which defeats every injection and also the task. Which cue drove the model toward ending answers early is not established. Response length in the training pairs does not explain it directly: preferred responses are slightly longer than rejected ones (median 9 words against 7, per the datasheet). The probes and the clean-document metric, built for plan question 8, are what exposed it.
- **Cause, as far as the logs show:** validation accuracy was 0.993 at step 125 and flat after; the preferred responses' log-probability then fell from -35.2 to -59.5 by step 250. The pairs were separated early and the remaining steps only pushed probabilities down, as in the M1 run but far stronger.
- **Follow-up runs prepared, not yet run:** beta 0.5 (stronger pull to the reference); stopping at step 60; and an added negative log-likelihood term on the preferred responses (`sft_weight` 1.0, the same as TRL's `loss_type=["sigmoid", "sft"]`), which directly penalises the preferred answers becoming less likely.
- **The M5 "done when" criterion (a before/after table with confidence intervals) is met.** The behavior is not achieved. M5 is ticked on the criterion, and the follow-ups decide what the write-up can claim.
- **The comparison warned that base and trained evaluations ran at different commits** (a5bb6d7 and 5a492b6). `git diff` between them shows no change to `src/tiny_rlhf/trackb/`, `scripts/eval_trackb.py`, `src/tiny_rlhf/stats.py` or `data/trackb/`; the commits in between added an analysis script and documentation only. The comparison is valid.

**2026-11-17, Track B follow-up prepared (stage `m5_followup` in the queue; not yet run)**
- **Three variants of `trackb_full`, each changing one thing:** `trackb_beta0.5` (beta 0.5, a stronger pull toward the reference); `trackb_steps60` (60 steps instead of 375, stopping before the plateau seen at step 125; the cap also shortens the learning-rate schedule, so it is a shorter run rather than an early stop of the same run); `trackb_sft1` (`sft_weight` 1.0).
- **`sft_weight` adds the mean negative log-likelihood per token of the preferred responses** to the DPO loss, matching TRL 1.18.0's `loss_type=["sigmoid", "sft"]` with `loss_weights=[1, sft_weight]` (a token-weighted mean over the micro-batch, not a mean of per-response means). Tests match TRL's loss and gradients at weight 0.5 and confirm that weight 0 is bit-identical to the plain loop. `train.csv` keeps `loss` as the DPO loss, so it still starts at log 2; `sft_nll` and `total_loss` columns are added only when the weight is positive.
- **All variants are evaluated with exactly the M5 settings and compared against the same base-model evaluation** (`results/trackb_full/trackb_eval/summary_base.json`), and `scripts/plot_trackb_variants.py` puts base, `trackb_full` and the variants side by side: injection followed, task correct on clean documents, median answer length, and the general-text NLL change.
- **Bug caught before it ran:** the comparison, figure and examples scripts wrote their outputs next to the base summary's run. With one shared base, every variant comparison would have overwritten `trackb_full`'s table, figure and examples. They now write next to the trained run, and a test guards it.
- **Queue estimates for M5 now use the measured times** from `results/trackb_full` instead of guesses.

**2026-11-17, full reward model run (`results/reward_full/`), M3 done**
- **Measured on a Colab T4:** 463 steps in 1,057 seconds, 6.4 GB peak, 4 fp16 step retries, step-0 loss log 2. Same 3,699 training pairs and same 456 held-out pairs as `dpo_full_t4` (fingerprint verified).
- **Result:** held-out accuracy 0.638 (0.594 to 0.682) against DPO's implicit reward 0.654 (0.607 to 0.693). They prefer the same response on 0.713 of pairs (0.673 to 0.754), Spearman of margins 0.49; when they disagree, each is right about as often (62 and 69 pairs).
- **Training was unstable in a way to fix before relying on this model for RL:** the gradient norm before clipping was 20 to 90 at every logged step, so every update was clipped to norm 1; training loss spiked to 2.4 near step 50; and the held-out loss stayed above log 2 from step 100 until step 400 (0.889 at step 100), ending at 0.679. Accuracy was about 0.6 throughout, so the model was ranking pairs but was overconfident on the ones it got wrong. The likely cause is the head learning rate of 1e-3, a guess, 20 times the adapter's. A rerun with a lower head learning rate is the obvious next check; it is not in the queue yet.
- **CI failed on two pushes because of a test-order race in the queue runner, now fixed.** A Python child process writing to a pipe buffers stdout but not stderr, so a line printed to stderr could arrive before an earlier stdout line (reproduced: `['two', 'one']`). The runner now sets `PYTHONUNBUFFERED=1` for every job it starts, which also makes progress appear live. No result was affected.
- **One unexplained local test failure.** Right after that fix, one full local run reported 1 failed test out of 300 without showing which. Thirteen reruns, locally and on CI, all passed. If a test fails intermittently again, record its name here before rerunning.

**2026-11-17, Track B follow-up results and a scoring flaw**
- **The three follow-up runs finished on a Colab T4** (`results/trackb_beta0.5`, `results/trackb_steps60`, `results/trackb_sft1`; every step-0 loss log 2; evaluation code and data identical to the base evaluation's commit, checked with `git diff`). Write-up follows once the rescoring below is done.
- **Scoring flaw found while reading the answers:** the `number` rule accepted the expected count if it appeared anywhere in the answer, so a numbered list ("1. ... 2. ... 3.") counted as correct for 3. On `count_items`, the base model scored 0.74 under that rule but stated the right count first only 0.15 of the time. `count_items` and `extract_date` are the held-out tasks, so this affects `test_heldout_task` and `test_heldout_all` for every model. A strict rescoring of the saved answers (no GPU needed) is being added; the lenient numbers stay reproducible, and the README will quote the strict ones.

**2026-11-17, M2 parity run shortened to 100 steps**
- `configs/parity_full.yaml` now stops at step 100 and evaluates at steps 0, 50 and 100. Parity needs enough steps to compare the two runs step by step, not a full epoch, and the full 462-step float32 pair was estimated at well over two hours of free Colab time. Both implementations get the same shortened schedule.

**2026-11-17, M7 added: KTO against DPO** (the owner's request)
- **Question:** RLHF and DPO learn from pairs ("A is better than B"). KTO (Ethayarajh et al., 2024) learns from single answers labelled good or bad. With the same Track B data and the same evaluation, does one teach the behavior better, and which damages the model less?
- **Fair comparison:** KTO gets the same examples DPO saw, with each pair split into one good and one bad answer, so the information differs only in that KTO never sees which two answers belong together.
- **Built after the current GPU runs,** in new files only, so the running Colab session is not affected.

**2026-11-17, M7 built (KTO), not yet run on a GPU**
- **What TRL 1.18.0's KTOTrainer does** (read from its source): each training row is a prompt, one answer and a good/bad label. The reference point z0 is the mean of log pi(y'|x) - log pi_ref(y'|x) over mismatched answers (each prompt paired with the answer of the previous row in its micro-batch), clamped at 0 and detached. A good answer's loss is 1 - sigmoid(beta * (r - z0)) and a bad answer's is 1 - sigmoid(beta * (z0 - r)), each times its weight, averaged over the batch. Log-probabilities are summed over answer tokens. With both weights 1 the step-0 loss is exactly 0.5.
- **The loss is bounded** between 0 and the weight, so answers already far past the reference point get almost no gradient; DPO's loss has no upper bound for a pair ranked the wrong way.
- **Fair comparison with DPO:** the same 3,000 Track B pairs split into 6,000 single answers, 16 answers per optimizer step (DPO used 8 pairs, also 16 answers), so both make 375 steps on the same schedule, same model, LoRA, learning rate and seed. Two runs, at beta 0.1 and 0.5, mirror the two main DPO runs. Beta does not mean exactly the same thing in the two losses.
- **Halves of a pair are kept out of the same micro-batch,** so a prompt's mismatched answer is never its own pair-mate.
- **Parity with TRL's KTOTrainer** on a tiny model in float32: same-batch loss within 1e-6, training curves within 1.5e-7 and weights within 6e-8 over two epochs. Six injected bugs (wrong rotation of mismatched answers, flipped sign, no clamp, z0 ignored, sum instead of mean, max instead of mean) each fail the tests.
- **TRL defaults that differ** and are set explicitly where they matter: learning rate 1e-6 (we use 5e-5 for LoRA, as in the DPO runs), gradient checkpointing on, bf16 on whenever fp16 is off, dropout disabled.

**2026-11-17, third Colab session**
- **The session cloned the code four minutes before the shortened parity run's new time estimate was pushed,** so the queue still budgeted 75 minutes for M2 and skipped it. It ran the beta sweep's first value instead (`results/sweep_beta_0.01`: 907 seconds, 8.8 GB peak). No job failed.
- **With the current estimates M2's two runs will usually land in different sessions** (the queue does not start a job that does not fit the time left). That is fine: each run is pushed when it finishes and the parity figure is drawn once both exist.

**2026-11-17, an old cell reran the tiny T4 run**
- An earlier one-run Colab cell (`configs/dpo_tiny.yaml`, run name `dpo_tiny_t4`) was run again at 21:37 UTC and overwrote `results/dpo_tiny_t4`, which the README quotes. The original files were restored from git, and the rerun was kept as `results/dpo_tiny_t4_rerun`.
- **The rerun is a useful reproducibility check:** it used code from after the `sequence_logprob` rewrite and the fp16 retry change, and ended at held-out loss 0.6829 against the original 0.6825 (19.21 s against 19.75 s, step-0 loss log 2 in both). The scoring rewrite did not change results beyond fp16 noise.

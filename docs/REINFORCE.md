# REINFORCE on a toy task, next to DPO

This is the first half of the plan's stretch item ("the actual RL part"): REINFORCE with a
baseline written by hand on a toy task, compared with DPO on the same task. The code is
`src/tiny_rlhf/reinforce.py`, the experiment is `scripts/train_reinforce.py`, and every
number below comes from `results/reinforce_toy/`.

The toy is small enough that every possible sequence can be listed, so the best possible
policy is known exactly and each method can be measured against it.

## 1. The policy-gradient identity

We want to raise J = E over y sampled from pi of R(y), by changing the policy's parameters.
Write the expectation as a sum:

    J = sum over y of pi(y) * R(y)
    grad J = sum over y of grad pi(y) * R(y)

The sum over all sequences is impossible for a language model, so it has to become an
average over samples. The log-derivative trick does that. Because
grad log pi(y) = grad pi(y) / pi(y), we have grad pi(y) = pi(y) * grad log pi(y), and so

    grad J = sum over y of pi(y) * R(y) * grad log pi(y)
           = E over y sampled from pi of [ R(y) * grad log pi(y) ]

That is REINFORCE: sample from the current policy, and push up the log-probability of
each sample in proportion to its return. In code the surrogate loss is

    loss = -mean( R(y) * log pi(y) )      with R held constant

whose gradient is the estimate above.

## 2. Why a baseline is free, and why it helps

Subtract any number b that does not depend on the sample:

    E[ (R(y) - b) * grad log pi(y) ] = grad J - b * E[ grad log pi(y) ]

and E[grad log pi(y)] = sum over y of grad pi(y) = grad (sum over y of pi(y)) = grad 1 = 0.
So the expected gradient is unchanged for every b. Only the noise changes.

Worked example. Two outcomes A and B, one parameter theta with pi(A) = sigmoid(theta),
currently pi(A) = 0.5. Returns R(A) = 11 and R(B) = 9.

- grad log pi(A) = 1 - pi(A) = 0.5, and grad log pi(B) = -pi(A) = -0.5.
- True gradient: pi(A) * (1 - pi(A)) * (11 - 9) = 0.5.
- Without a baseline, one sample gives either 11 * 0.5 = 5.5 or 9 * (-0.5) = -4.5.
  The average is 0.5, which is correct, but each sample is off by 5: variance 25.
- With b = 10, one sample gives either 1 * 0.5 = 0.5 or (-1) * (-0.5) = 0.5.
  The average is still 0.5 and the variance is 0.

The returns 11 and 9 share a large common part (10) that says nothing about which
outcome is better. Without a baseline that common part multiplies the noisy direction
grad log pi. The baseline removes it.

The code compares three settings:

- **none:** b = 0.
- **running mean:** an exponential moving average of the mean return of earlier batches.
  It never includes the current batch, so it is independent of the samples it is applied to.
- **leave-one-out mean:** for each sample, the mean return of the other samples in the
  batch. Using the plain batch mean would include a sample's own return in its own
  baseline, which biases the estimate slightly; leaving it out keeps it unbiased.

## 3. The KL-penalised objective and its closed form

The objective the DPO derivation starts from is

    J(pi) = E over y from pi of [ r(y) ] - beta * KL(pi || pi_ref)
          = E over y from pi of [ r(y) - beta * (log pi(y) - log pi_ref(y)) ]

**How REINFORCE handles the KL term.** Define the per-sample return
R(y) = r(y) - beta * (log pi(y) - log pi_ref(y)). Now R depends on the parameters through
log pi(y), so differentiating gives an extra term:

    grad J = E[ R(y) * grad log pi(y) ] - beta * E[ grad log pi(y) ]

The extra term is beta times the same zero as in section 2. So R can be computed, held
constant, and used as the return. This is what `kl_penalised_return` does.

**The closed form.** Maximise J over all distributions with a Lagrange multiplier lam for
sum over y of pi(y) = 1. Setting the derivative with respect to pi(y) to zero:

    r(y) - beta * (log pi(y) - log pi_ref(y)) - beta - lam = 0
    log pi(y) = log pi_ref(y) + r(y) / beta + constant
    pi*(y) = pi_ref(y) * exp(r(y) / beta) / Z,     Z = sum over y of pi_ref(y) * exp(r(y) / beta)

Two consequences that the code checks numerically:

- J(pi*) = beta * log Z.
- For any policy, J(pi) = J(pi*) - beta * KL(pi || pi*). So "KL to the optimum" is the
  gap in the objective, measured in units of beta. A test asserts this identity.

**This is the formula DPO inverts.** Solve the closed form for the reward:

    r(y) = beta * (log pi*(y) - log pi_ref(y)) + beta * log Z

Put that into the Bradley-Terry model P(a preferred over b) = sigmoid(r(a) - r(b)). The
beta * log Z term is the same for a and b and cancels, which leaves the DPO loss. So if
the labels really follow the Bradley-Terry model with reward r, the policy that best fits
the DPO loss with the same beta is exactly pi*. A test minimises the exact (infinite
data) DPO loss on a 9-sequence case and lands on pi* to 6 decimal places.

## 4. On-policy, off-policy, offline

- **On-policy:** learn from samples drawn from the policy being trained, and throw them
  away after each update because the policy has changed. REINFORCE is on-policy. So is
  PPO, which reuses each batch for a few updates and corrects for the mismatch with an
  importance ratio that it clips.
- **Off-policy:** learn from samples drawn from a different policy while still collecting
  new ones along the way.
- **Offline:** learn from a fixed dataset collected beforehand, with no new sampling at
  all. DPO is offline: its pairs here are sampled once from the reference and labelled
  once. It never generates from the policy it trains and never sees a reward value, only
  which of two sequences was preferred.

The practical difference: REINFORCE and PPO need a reward function that can score fresh
samples (in RLHF, a trained reward model). DPO needs only the logged pairs, and in
exchange it can only learn about sequences the data covers.

## 5. The toy and what was measured

**Task.** Sequences of 6 tokens from a vocabulary of 4 (tokens 0, 1, 2 and a stop token),
so 4^6 = 4096 possible sequences. Reward: +1 for each adjacent pair that counts upward
cyclically (0 then 1, 1 then 2, 2 then 0) and +2 if the last token is the stop token.
The maximum is 6. The policy is a GRU with 5,012 parameters. The reference is its random
initialisation, frozen. beta = 0.5. All metrics are exact sums over the 4096 sequences,
so the curves contain no evaluation noise.

**Budget.** 400 steps of 32 sequences = 12,800 sequences per run, 5 sampling seeds
(0 to 4), Adam with learning rate 0.01 for every method (not tuned per method). A DPO
step uses 16 pairs, which is also 32 sequences. The whole experiment took 75.7 seconds
on one CPU thread on a shared machine (`results/reinforce_toy/run_info.json`).

**The exact optimum** (`summary.json`, "task"): the reference earns an expected reward of
1.425. The closed-form optimum pi* earns 4.668 at a KL of 3.114 from the reference.

**Final values after 12,800 sequences** (`summary.json`, "final"; mean over 5 seeds, with
the lowest and highest seed in brackets):

| Method | Expected reward | KL from reference | KL to optimum |
|---|---|---|---|
| Closed-form optimum | 4.668 | 3.114 | 0 |
| REINFORCE, no baseline | 4.528 [4.074, 4.780] | 3.529 [3.183, 3.900] | 0.695 [0.366, 1.471] |
| REINFORCE, running mean | 4.659 [4.652, 4.664] | 3.103 [3.083, 3.130] | 0.0073 [0.0012, 0.0299] |
| REINFORCE, leave-one-out | 4.665 [4.661, 4.669] | 3.110 [3.102, 3.116] | 0.0016 [0.0012, 0.0031] |
| DPO, Bradley-Terry labels | 3.767 [3.660, 3.982] | 2.310 [1.874, 2.578] | 0.999 [0.592, 1.465] |
| DPO, always-correct labels | 5.413 [5.153, 5.649] | 5.889 [5.738, 6.045] | 1.285 [0.905, 1.962] |

**Gradient noise** (`grad_variance.json`): trace of the covariance of the gradient
estimate across 200 batches of 32, at fixed parameters taken from one leave-one-out run
(seed 0).

| Policy taken at | No baseline | Running mean | Leave-one-out |
|---|---|---|---|
| step 0 | 2.20 | 0.91 | 0.99 |
| step 100 | 16.1 | 0.024 | 0.024 |
| step 400 | 13.6 | 0.0016 | 0.0016 |

What the numbers show, on this toy:

- **REINFORCE with a baseline reaches the closed form.** With either baseline the learned
  policy ends within 0.01 nats of pi* on average, and its expected reward and its KL from
  the reference both match the optimum's to two decimal places.
- **The baseline matters more as training goes on.** At the start it cuts the gradient
  noise by about half. Near the optimum the returns of different samples are almost
  equal (that is what being at the optimum of this objective means), so with a baseline
  the noise falls to 0.0016 while without one it stays above 13, roughly 8,000 times
  larger. Without a baseline the policy keeps wandering around the optimum.
- **The two baselines are about equally good here.**
- **At the same number of sequences, DPO is much further from the optimum than
  REINFORCE.** With Bradley-Terry labels it is moving in the right direction: its
  expected reward has covered about 70% of the way from the reference (1.425) to the
  optimum (4.668), but its KL to the optimum is still 1.0, against 0.0016 for REINFORCE
  with a baseline.
- **With ten times more pairs DPO with Bradley-Terry labels gets close** (`summary.json`,
  "extended_final", one run, seed 0, 128,000 sequences): expected reward 4.507, KL from
  reference 2.880, KL to optimum 0.088. That is the prediction of section 3.
- **With always-correct labels DPO overshoots.** After 128,000 sequences its expected
  reward is 5.999 out of 6 and its KL from the reference is 7.18, more than twice the
  optimum's 3.11. Labels that are never wrong are not explained by any finite reward
  difference, so the loss keeps widening the margins and beta no longer limits the drift
  in the way the derivation promises. Real preference data sits between these two
  labelling rules.

The figure is `assets/m_stretch_reinforce_toy.png`, made by `scripts/plot_reinforce.py`.

## 6. What this does not show

- **It is a toy.** 4096 sequences, a 5,012-parameter model, no prompt, and a reference
  that is close to uniform, so samples from the reference cover every sequence. None of
  that holds for a language model. Nothing here transfers quantitatively.
- **The sample-efficiency comparison is specific to this setup.** REINFORCE was given the
  true reward function for free. In RLHF that reward comes from a trained reward model,
  which costs preference data too and can be wrong. DPO's pairs came only from the
  reference; pairs from better policies would help it. One learning rate was used for
  everything and was not tuned.
- **Only the simplest estimator is implemented.** No reward-to-go, no learned value
  function, no PPO clipping, no generalised advantage estimation.
- **The extended DPO runs are one seed each.**
- **The second half of the stretch item is not done:** running a current policy-gradient
  trainer from TRL on the real model with the Track A reward model, and comparing it
  with DPO at the same compute budget.

## Reproduce

```bash
python scripts/train_reinforce.py --config configs/reinforce_toy.yaml
python scripts/plot_reinforce.py --run results/reinforce_toy
```

# One-hour Colab session

One cell that runs, in priority order, as much of the remaining GPU work as fits in about an hour on a free Colab T4, and pushes after every finished step:

1. **M5, Track B:** evaluate the base model, train on the Track B pairs, evaluate the trained adapter, run the general regression check, then write the before/after table, the hero figure and side-by-side examples.
2. **M3, reward model:** started only if at least `REWARD_NEEDS_MIN` minutes are left in the budget, then compared with the DPO run `results/dpo_full_t4`.

How long each step takes on a T4 has not been measured. The only measured number is the full DPO run: 463 optimizer steps in 1,737 seconds with an 8.8 GB peak.

Rules the cell follows:

- **Live output:** every command's output is shown as it is printed.
- **Pushes:** only steps that finished are pushed. A partial run is never committed.
- **Failures:** a failed step stops its own stage only. Its last 80 lines of output are committed to `results/colab_failures/`, so they can be read from the repo without copying anything.
- **Out of memory:** a training run that runs out of GPU memory is retried once with half the micro-batch. The effective batch stays the same.
- **Resuming:** rerunning the cell starts a fresh clone. Finished steps are already in the repo; the base evaluation is reused when its settings match, and everything else runs again.

The five test splits are evaluated on their first 150 examples each, and every probe set on all 100. This halves the generation time. Each of those first-150 slices contains all three kinds of example (planted instruction, ordinary, and harmless instructions), checked on 2026-11-17.

```python
# tiny-rlhf: one-hour session on a Colab T4. Runs M5 (Track B) then, if time allows, M3.
BUDGET_MIN = 55  # stop starting new stages after this many minutes
REWARD_NEEDS_MIN = 25  # start the reward model only if at least this much budget is left
EVAL_ARGS = "--with-probes --batch-size 16 --limit 150"
BRANCH = "claude/charming-cray-20zxv9"
REPO = "Varkot-dev/tiny-rlhf"

import base64, collections, os, shutil, subprocess, time
import torch
from google.colab import userdata

assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then run again."
START = time.time()
GPU = torch.cuda.get_device_name(0)


def minutes_used():
    return (time.time() - START) / 60


# Give git the token through environment variables, so it is never written to disk or printed.
auth = base64.b64encode(f"x-access-token:{userdata.get('GH_TOKEN')}".encode()).decode()
os.environ.update(
    {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONUNBUFFERED": "1",
    }
)
TAIL = collections.deque(maxlen=80)  # the last lines of output, saved if a step fails


def run(cmd, show=True):
    """Run a shell command, print its output live, raise if it fails."""
    print(f"\n$ {cmd}   [{minutes_used():.1f} min]", flush=True)
    TAIL.clear()
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        TAIL.append(line)
        if show and "Warning" not in line and "warnings.warn" not in line:
            print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"exit {proc.returncode}: {cmd}")


def push(message, *paths):
    """Commit the given paths and push, merging anything pushed in the meantime."""
    run("git add " + " ".join(paths), show=False)
    if subprocess.run("git diff --cached --quiet", shell=True).returncode == 0:
        print("nothing new to commit")
        return
    run(f'git commit -q -m "{message}"')
    for attempt in range(4):
        try:
            run(f"git fetch -q origin {BRANCH} && git merge -q --no-edit FETCH_HEAD")
            run(f"git push -q origin HEAD:{BRANCH}")
            print(f"pushed: {message}")
            return
        except RuntimeError:
            time.sleep(2 ** (attempt + 1))
    print("PUSH FAILED. Do not close this tab; tell Claude 'push failed'.")


def record_failure(stage, err):
    """Commit the tail of the failed step's output so it can be read from the repo."""
    tail = "".join(TAIL)
    print(f"\n*** {stage} FAILED: {err}\n{tail}")
    os.makedirs("results/colab_failures", exist_ok=True)
    path = f"results/colab_failures/{stage}.log"
    with open(path, "w") as f:
        f.write(f"{err}\nGPU: {GPU}\nminutes used: {minutes_used():.1f}\n\n{tail}")
    push(f"Record failure of {stage} (Colab {GPU})", path)


def train_with_oom_retry(cmd, small_batch_args):
    """Train; if the GPU runs out of memory, retry once with half the micro-batch."""
    try:
        run(cmd)
    except RuntimeError:
        if "out of memory" not in "".join(TAIL).lower():
            raise
        print("\nOut of GPU memory. Retrying with a smaller micro-batch (same effective batch).")
        torch.cuda.empty_cache()
        run(f"{cmd} {small_batch_args}")


# Setup: fresh clone, install, tests.
os.chdir("/content")
shutil.rmtree("/content/tiny-rlhf", ignore_errors=True)
run(f"git clone -q --branch {BRANCH} https://github.com/{REPO}.git /content/tiny-rlhf")
os.chdir("/content/tiny-rlhf")
run('pip install -q -e ".[train,viz,dev]"')
run("pip uninstall -y -q torchao")  # Colab's old torchao breaks the LoRA library; unused here
run('git config user.name "Varshith Kotagiri"')
run('git config user.email "Varkot-dev@users.noreply.github.com"')
try:
    run("python -m pytest -q -x", show=False)
    print("tests passed")
except RuntimeError as err:
    record_failure("tests", err)  # recorded, but the runs go ahead: they have their own checks

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
status = {}

# Stage 1: M5, Track B.
RUN = "trackb_full"
EVAL = f"results/{RUN}/trackb_eval"
ADAPTER = f"checkpoints/{RUN}"
SUMMARIES = f"--base {EVAL}/summary_base.json --trained {EVAL}/summary_trained.json"
try:
    run(
        f"python scripts/eval_trackb.py --model {MODEL} --label base --out {EVAL} {EVAL_ARGS} --reuse-if-current"
    )
    push(f"Add Track B base-model evaluation (Colab {GPU})", EVAL)

    train_with_oom_retry(
        "python scripts/train_trackb.py --config configs/trackb_full.yaml --set eval_every=125",
        "--set batch_size=1 --set grad_accum=8 --set eval_batch_size=2",
    )
    assert os.path.exists(f"results/{RUN}/run_info.json"), "training did not finish"
    run(f"python scripts/plot_training.py --run results/{RUN} --out assets/m5_training_{RUN}.png")
    push(f"Add Track B training run {RUN} (Colab {GPU})", f"results/{RUN}", "assets")

    run(
        f"python scripts/eval_trackb.py --model {MODEL} --adapter {ADAPTER} --label trained --out {EVAL} {EVAL_ARGS}"
    )
    push(f"Add Track B trained-model evaluation (Colab {GPU})", EVAL)

    run(
        f"python scripts/eval_general.py --model {MODEL} --adapter {ADAPTER} --out results/{RUN}/general_regression.json"
    )
    run(f"python scripts/compare_trackb.py {SUMMARIES}")
    run(f"python scripts/plot_trackb.py {SUMMARIES}")
    run(f"python scripts/make_examples.py {SUMMARIES}")
    push(
        f"Add Track B before/after table, figure and examples (Colab {GPU})",
        f"results/{RUN}",
        "assets",
    )
    status["M5 Track B"] = "done"
except Exception as err:
    status["M5 Track B"] = f"failed: {err}"
    record_failure("m5_trackb", err)

# Stage 2: M3, reward model, only if enough of the budget is left.
left = BUDGET_MIN - minutes_used()
if left < REWARD_NEEDS_MIN:
    status["M3 reward model"] = f"not started: {left:.0f} min of budget left"
else:
    try:
        train_with_oom_retry(
            "python scripts/train_reward.py --config configs/reward_full.yaml",
            "--set batch_size=1 --set grad_accum=8 --set eval_batch_size=2",
        )
        assert os.path.exists("results/reward_full/run_info.json"), "training did not finish"
        run(
            "python scripts/plot_reward_vs_dpo.py --dpo results/dpo_full_t4 --reward results/reward_full"
        )
        push(
            f"Add full reward model run and its comparison with DPO (Colab {GPU})",
            "results/reward_full",
            "assets",
        )
        status["M3 reward model"] = "done"
    except Exception as err:
        status["M3 reward model"] = f"failed: {err}"
        record_failure("m3_reward", err)

print(f"\n===== finished in {minutes_used():.1f} min =====")
for stage, result in status.items():
    print(f"{stage}: {result}")

from IPython.display import Image, Markdown, display

for png in [f"assets/m5_trackb_{RUN}.png", "assets/m3_reward_vs_dpo_reward_full.png"]:
    if os.path.exists(png):
        display(Image(png))
if os.path.exists(f"results/{RUN}/trackb_table.md"):
    display(Markdown(open(f"results/{RUN}/trackb_table.md").read()))
```

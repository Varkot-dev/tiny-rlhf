# Track B on Colab: one cell

> **Recommended:** to run all remaining GPU work in one resumable cell, use [docs/COLAB_QUEUE.md](COLAB_QUEUE.md).

This runs the whole Track B experiment on a free Colab T4 and pushes every output to
the repo. Setup is the same as in `docs/COLAB.md`: pick the T4 runtime and add a
`GH_TOKEN` secret (fine-grained token, this repository only, Contents: Read and write)
with notebook access switched on.

Paste the cell below into a new Colab cell and run it. It works in three stages and
commits after each one, so an earlier stage is saved even if a later one fails:

1. Score the base model on the five test files and the three probe sets.
2. Train `configs/trackb_full.yaml` with the hand-written DPO loop.
3. Score the trained adapter, run the general regression check, and build the
   before/after table, the figure and the side-by-side examples.

Time and memory are not known in advance for this model. The training run records
both in `results/trackb_full/run_info.json`. If stage 2 stops with an out-of-memory
error, set `TRAIN_ARGS = "--set batch_size=1 --set grad_accum=8"` and run the cell
again: stage 1 is skipped when its output is already in the repo and was made with the
same data and settings (the evaluation script checks this itself).

Every command stops the cell if it fails, and each stage is pushed only after its
commands finished. Starting a training run deletes the previous run's completion files
in the same directory, so a run that crashes cannot be pushed as if it had finished.

```python
# tiny-rlhf, Track B: evaluate base, train, evaluate trained, compare, push.
CONFIG = "configs/trackb_full.yaml"
TRAIN_ARGS = ""  # for example "--set batch_size=1 --set grad_accum=8" after an out-of-memory error
EVAL_ARGS = "--with-probes --batch-size 8"
BRANCH = "claude/charming-cray-20zxv9"
REPO = "Varkot-dev/tiny-rlhf"

import base64, os, shutil, subprocess, torch, yaml
from google.colab import userdata

assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then run again."

# Pass the GitHub token to git through environment variables, so it is never written to disk or printed.
auth = base64.b64encode(f"x-access-token:{userdata.get('GH_TOKEN')}".encode()).decode()
os.environ.update(
    {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
        "GIT_TERMINAL_PROMPT": "0",
    }
)


def sh(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def save(message, *paths):
    """Commit the given paths and push, merging anything pushed in the meantime."""
    sh("git add " + " ".join(paths))
    if subprocess.run("git diff --cached --quiet", shell=True).returncode == 0:
        print("nothing new to commit")
        return
    sh(f'git commit -q -m "{message}"')
    sh(f"git fetch -q origin {BRANCH} && git merge -q --no-edit FETCH_HEAD")
    sh(f"git push -q origin HEAD:{BRANCH}")


os.chdir("/content")
shutil.rmtree("/content/tiny-rlhf", ignore_errors=True)
sh(f"git clone --branch {BRANCH} https://github.com/{REPO}.git /content/tiny-rlhf")
os.chdir("/content/tiny-rlhf")
sh('pip install -q -e ".[train,viz,dev]"')
sh("pip uninstall -y -q torchao")  # Colab's old torchao breaks the LoRA library; we do not use it
sh("python -m pytest -q")
sh('git config user.name "Varshith Kotagiri"')
sh('git config user.email "Varkot-dev@users.noreply.github.com"')

cfg = yaml.safe_load(open(CONFIG))
RUN, MODEL = cfg["run_name"], cfg["model_name"]
EVAL = f"results/{RUN}/trackb_eval"
GPU = torch.cuda.get_device_name(0)

# Stage 1: the base model, before any training. Committed first. An evaluation from an
# earlier session is reused only if it scored the same prompts with the same settings;
# the script checks that itself and evaluates again otherwise.
sh(
    f"python scripts/eval_trackb.py --model {MODEL} --label base --out {EVAL} {EVAL_ARGS} --reuse-if-current"
)
save(f"Add Track B base-model evaluation for {RUN} (Colab {GPU})", EVAL)

# Stage 2: train.
sh(f"python scripts/train_trackb.py --config {CONFIG} {TRAIN_ARGS}")
assert os.path.exists(f"results/{RUN}/run_info.json"), "Training did not finish; not pushed."
sh(f"python scripts/plot_training.py --run results/{RUN} --out assets/m5_training_{RUN}.png")
save(f"Add Track B training run {RUN} (Colab {GPU})", f"results/{RUN}", "assets")

# Stage 3: the trained adapter, the regression check, and the comparison.
ADAPTER = f"checkpoints/{RUN}"
sh(
    f"python scripts/eval_trackb.py --model {MODEL} --adapter {ADAPTER} --label trained --out {EVAL} {EVAL_ARGS}"
)
save(f"Add Track B trained-model evaluation for {RUN} (Colab {GPU})", EVAL)
sh(
    f"python scripts/eval_general.py --model {MODEL} --adapter {ADAPTER} --out results/{RUN}/general_regression.json"
)
summaries = f"--base {EVAL}/summary_base.json --trained {EVAL}/summary_trained.json"
sh(f"python scripts/compare_trackb.py {summaries}")
sh(f"python scripts/plot_trackb.py {summaries}")
sh(f"python scripts/make_examples.py {summaries}")
save(f"Add Track B before/after table, figure and examples for {RUN}", f"results/{RUN}", "assets")

from IPython.display import Image, Markdown, display

display(Image(f"assets/m5_trackb_{RUN}.png"))
display(Markdown(open(f"results/{RUN}/trackb_table.md").read()))
print(open(f"results/{RUN}/general_regression.json").read())
```

## What gets written

| Path | What it is |
|---|---|
| `results/trackb_full/trackb_eval/summary_base.json`, `summary_trained.json` | rates with 95% bootstrap intervals, per test file and probe |
| `results/trackb_full/trackb_eval/<split>_<label>.jsonl` | every prompt, the generated answer and its checks |
| `results/trackb_full/train.csv`, `eval.csv`, `run_info.json` | the training run, as for Track A |
| `results/trackb_full/general_regression.json` | drift on ordinary chat data |
| `results/trackb_full/trackb_table.md`, `trackb_compare.json` | the before/after table with paired intervals |
| `results/trackb_full/examples.md` | the same prompts answered by both models |
| `assets/m5_trackb_trackb_full.png`, `assets/m5_training_trackb_full.png` | the two figures |

The adapter itself stays in `checkpoints/trackb_full/` on the Colab machine and is not
committed. Copy it to Google Drive before the runtime is recycled if you want to keep it.

## Not yet verified

This cell has not been executed on Colab. The scripts it calls were tested on a CPU
with a tiny random model, and the base-model evaluation was run for 12 examples with
the 135M model. Nothing here has run with the Qwen model or on a GPU.

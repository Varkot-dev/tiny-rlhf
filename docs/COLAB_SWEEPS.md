# Running the ablation sweeps on Google Colab

> **Recommended:** to run all remaining GPU work in one resumable cell, use [docs/COLAB_QUEUE.md](COLAB_QUEUE.md).

Each sweep is several ordinary training runs in a row. One paste-ready cell per sweep is below. Setup is the same as for a single run and is described in `docs/COLAB.md`: a T4 GPU runtime and a `GH_TOKEN` secret with notebook access switched on.

## What each cell does

1. Clones the branch, installs the project, removes Colab's outdated `torchao`, and runs the tests.
2. For each run of the sweep: trains it, then commits and pushes that run's results straight away. If Colab disconnects, at most the run in progress is lost.
3. Skips runs whose results are already in the repo, so running the same cell again continues where the last session stopped. Only a run the sweep runner marks as done is committed; a run that crashes leaves nothing in the repo.
4. Makes the sweep's figure, commits it with a summary of the plotted numbers, and pushes.
5. A run that fails is recorded as failed in `results/sweeps/<name>/index.json` and the cell carries on with the next run.

## How long it takes

Not known yet, and this document does not guess. What has been measured so far:

- The tiny config (135M-parameter model, 41 optimizer steps) took 19.75 seconds on a T4 with a peak of 3.8 GiB of GPU memory (`results/dpo_tiny_t4/run_info.json`).
- Nothing has been measured for the 494M-parameter Qwen model that the sweeps use. The first full run failed with out of memory and its settings were changed afterwards (decision log, 2026-11-17).

Every run records its own `wall_time_s` and `peak_gpu_memory_mb` in `run_info.json`. After the first run of any sweep finishes, that number tells you how long the rest will take, since the runs in a sweep are the same size (the dataset size sweep holds optimizer steps fixed for exactly this reason). The sweeps use a reduced budget, 2,000 training pairs and 300 held-out pairs, stated at the top of each file under `configs/sweeps/`.

Free Colab sessions can disconnect. Because every run is pushed as soon as it finishes, the fix is to run the same cell again.

## To see what a sweep will run without running it

```bash
python scripts/run_sweep.py --sweep configs/sweeps/beta.yaml --dry-run
```

## Beta sweep

Three runs (beta 0.01, 0.1, 0.5). After each run the cell also measures how far the trained model moved from the reference (`scripts/eval_drift.py`), which is the x axis of the figure.

```python
# tiny-rlhf: run the beta sweep on the Colab GPU, pushing after every run.
SWEEP = "configs/sweeps/beta.yaml"
NEEDS_DRIFT = True  # the beta figure needs each run's drift from the reference
BRANCH = "claude/charming-cray-20zxv9"
REPO = "Varkot-dev/tiny-rlhf"

import base64, os, shutil, subprocess, torch
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


def sh(cmd, check=True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check).returncode


def push(message):
    """Commit whatever is staged, merge anything new on the branch, push."""
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

import yaml

name = yaml.safe_load(open(SWEEP))["name"]
gpu = torch.cuda.get_device_name(0)
# The run names of this sweep, as the sweep runner itself computes them.
runs = subprocess.run(
    [
        "python",
        "-c",
        "import sys; from tiny_rlhf.sweeps import load_sweep; "
        "print('\\n'.join(r.run_name for r in load_sweep(sys.argv[1]).runs))",
        SWEEP,
    ],
    capture_output=True,
    text=True,
    check=True,
).stdout.split()

import json


def run_status(run):
    """The sweep runner's own verdict on a run it just launched: "done" or "failed"."""
    index = json.load(open(f"results/sweeps/{name}/index.json"))
    return next(r["status"] for r in index["runs"] if r["run_name"] == run)


for run in runs:
    finished = os.path.exists(f"results/{run}/run_info.json")
    have_drift = os.path.exists(f"results/{run}/drift.json")
    if finished and (have_drift or not NEEDS_DRIFT):
        print(f"{run}: already in the repo, skipped")
        continue
    # A finished run without drift.json is trained again: the drift measurement needs the
    # trained adapter, which is not stored in git and is gone with the previous runtime.
    # Training deletes the old run_info.json first, so a crash cannot look like a finished run.
    sh(f"python scripts/run_sweep.py --sweep {SWEEP} --only {run}", check=False)
    status = run_status(run)
    sh(f"git add results/sweeps/{name}")
    if status != "done":
        print(
            f"{run} FAILED (status {status}). Its error is printed above. Nothing of this run is committed."
        )
        # Put back the committed copy of this run's files, if an earlier session pushed one.
        subprocess.run(
            f"git ls-files --error-unmatch results/{run} >/dev/null 2>&1 && git checkout -q -- results/{run}",
            shell=True,
        )
        push(f"Record failed sweep run {run} (Colab {gpu})")
        continue
    if NEEDS_DRIFT and sh(f"python scripts/eval_drift.py --run results/{run}", check=False) != 0:
        print(
            f"{run}: training finished but the drift measurement FAILED (error above). The run is "
            "committed without drift.json, and this cell will train it again next time."
        )
    sh(f"git add results/{run}")
    push(f"Add sweep run {run} (Colab {gpu})")

sh("python scripts/plot_beta_sweep.py")
sh(f"git add results/sweeps/{name} assets")
push(f"Add {name} sweep figure and summary")

from IPython.display import Image, display

display(Image("assets/m6_beta_sweep.png"))
print(open(f"results/sweeps/{name}/index.json").read())
```

## LoRA rank sweep

Four runs (rank 2, 8, 16, 64, with alpha kept at twice the rank). Full fine-tuning is not included; `configs/sweeps/lora_rank.yaml` explains why it does not fit a T4.

```python
# tiny-rlhf: run the LoRA rank sweep on the Colab GPU, pushing after every run.
SWEEP = "configs/sweeps/lora_rank.yaml"
NEEDS_DRIFT = False  # the beta figure needs each run's drift from the reference
BRANCH = "claude/charming-cray-20zxv9"
REPO = "Varkot-dev/tiny-rlhf"

import base64, os, shutil, subprocess, torch
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


def sh(cmd, check=True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check).returncode


def push(message):
    """Commit whatever is staged, merge anything new on the branch, push."""
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

import yaml

name = yaml.safe_load(open(SWEEP))["name"]
gpu = torch.cuda.get_device_name(0)
# The run names of this sweep, as the sweep runner itself computes them.
runs = subprocess.run(
    [
        "python",
        "-c",
        "import sys; from tiny_rlhf.sweeps import load_sweep; "
        "print('\\n'.join(r.run_name for r in load_sweep(sys.argv[1]).runs))",
        SWEEP,
    ],
    capture_output=True,
    text=True,
    check=True,
).stdout.split()

import json


def run_status(run):
    """The sweep runner's own verdict on a run it just launched: "done" or "failed"."""
    index = json.load(open(f"results/sweeps/{name}/index.json"))
    return next(r["status"] for r in index["runs"] if r["run_name"] == run)


for run in runs:
    finished = os.path.exists(f"results/{run}/run_info.json")
    have_drift = os.path.exists(f"results/{run}/drift.json")
    if finished and (have_drift or not NEEDS_DRIFT):
        print(f"{run}: already in the repo, skipped")
        continue
    # A finished run without drift.json is trained again: the drift measurement needs the
    # trained adapter, which is not stored in git and is gone with the previous runtime.
    # Training deletes the old run_info.json first, so a crash cannot look like a finished run.
    sh(f"python scripts/run_sweep.py --sweep {SWEEP} --only {run}", check=False)
    status = run_status(run)
    sh(f"git add results/sweeps/{name}")
    if status != "done":
        print(
            f"{run} FAILED (status {status}). Its error is printed above. Nothing of this run is committed."
        )
        # Put back the committed copy of this run's files, if an earlier session pushed one.
        subprocess.run(
            f"git ls-files --error-unmatch results/{run} >/dev/null 2>&1 && git checkout -q -- results/{run}",
            shell=True,
        )
        push(f"Record failed sweep run {run} (Colab {gpu})")
        continue
    if NEEDS_DRIFT and sh(f"python scripts/eval_drift.py --run results/{run}", check=False) != 0:
        print(
            f"{run}: training finished but the drift measurement FAILED (error above). The run is "
            "committed without drift.json, and this cell will train it again next time."
        )
    sh(f"git add results/{run}")
    push(f"Add sweep run {run} (Colab {gpu})")

sh("python scripts/plot_sweep.py --sweep lora_rank")
sh(f"git add results/sweeps/{name} assets")
push(f"Add {name} sweep figure and summary")

from IPython.display import Image, display

display(Image("assets/m6_sweep_lora_rank.png"))
print(open(f"results/sweeps/{name}/index.json").read())
```

## Dataset size sweep

Four runs (100, 300, 1000, 3000 training pairs), each capped at the same 340 optimizer steps so the comparison is at equal compute.

```python
# tiny-rlhf: run the dataset size sweep on the Colab GPU, pushing after every run.
SWEEP = "configs/sweeps/data_size.yaml"
NEEDS_DRIFT = False  # the beta figure needs each run's drift from the reference
BRANCH = "claude/charming-cray-20zxv9"
REPO = "Varkot-dev/tiny-rlhf"

import base64, os, shutil, subprocess, torch
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


def sh(cmd, check=True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check).returncode


def push(message):
    """Commit whatever is staged, merge anything new on the branch, push."""
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

import yaml

name = yaml.safe_load(open(SWEEP))["name"]
gpu = torch.cuda.get_device_name(0)
# The run names of this sweep, as the sweep runner itself computes them.
runs = subprocess.run(
    [
        "python",
        "-c",
        "import sys; from tiny_rlhf.sweeps import load_sweep; "
        "print('\\n'.join(r.run_name for r in load_sweep(sys.argv[1]).runs))",
        SWEEP,
    ],
    capture_output=True,
    text=True,
    check=True,
).stdout.split()

import json


def run_status(run):
    """The sweep runner's own verdict on a run it just launched: "done" or "failed"."""
    index = json.load(open(f"results/sweeps/{name}/index.json"))
    return next(r["status"] for r in index["runs"] if r["run_name"] == run)


for run in runs:
    finished = os.path.exists(f"results/{run}/run_info.json")
    have_drift = os.path.exists(f"results/{run}/drift.json")
    if finished and (have_drift or not NEEDS_DRIFT):
        print(f"{run}: already in the repo, skipped")
        continue
    # A finished run without drift.json is trained again: the drift measurement needs the
    # trained adapter, which is not stored in git and is gone with the previous runtime.
    # Training deletes the old run_info.json first, so a crash cannot look like a finished run.
    sh(f"python scripts/run_sweep.py --sweep {SWEEP} --only {run}", check=False)
    status = run_status(run)
    sh(f"git add results/sweeps/{name}")
    if status != "done":
        print(
            f"{run} FAILED (status {status}). Its error is printed above. Nothing of this run is committed."
        )
        # Put back the committed copy of this run's files, if an earlier session pushed one.
        subprocess.run(
            f"git ls-files --error-unmatch results/{run} >/dev/null 2>&1 && git checkout -q -- results/{run}",
            shell=True,
        )
        push(f"Record failed sweep run {run} (Colab {gpu})")
        continue
    if NEEDS_DRIFT and sh(f"python scripts/eval_drift.py --run results/{run}", check=False) != 0:
        print(
            f"{run}: training finished but the drift measurement FAILED (error above). The run is "
            "committed without drift.json, and this cell will train it again next time."
        )
    sh(f"git add results/{run}")
    push(f"Add sweep run {run} (Colab {gpu})")

sh("python scripts/plot_sweep.py --sweep data_size")
sh(f"git add results/sweeps/{name} assets")
push(f"Add {name} sweep figure and summary")

from IPython.display import Image, display

display(Image("assets/m6_sweep_data_size.png"))
print(open(f"results/sweeps/{name}/index.json").read())
```

## Label noise sweep

Four runs: 0%, 10% and 30% of the training pairs flipped with the ordinary loss, then the 30% data again with the robust loss (what TRL's `label_smoothing` does with `loss_type="robust"`).

```python
# tiny-rlhf: run the label noise sweep on the Colab GPU, pushing after every run.
SWEEP = "configs/sweeps/label_noise.yaml"
NEEDS_DRIFT = False  # the beta figure needs each run's drift from the reference
BRANCH = "claude/charming-cray-20zxv9"
REPO = "Varkot-dev/tiny-rlhf"

import base64, os, shutil, subprocess, torch
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


def sh(cmd, check=True):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check).returncode


def push(message):
    """Commit whatever is staged, merge anything new on the branch, push."""
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

import yaml

name = yaml.safe_load(open(SWEEP))["name"]
gpu = torch.cuda.get_device_name(0)
# The run names of this sweep, as the sweep runner itself computes them.
runs = subprocess.run(
    [
        "python",
        "-c",
        "import sys; from tiny_rlhf.sweeps import load_sweep; "
        "print('\\n'.join(r.run_name for r in load_sweep(sys.argv[1]).runs))",
        SWEEP,
    ],
    capture_output=True,
    text=True,
    check=True,
).stdout.split()

import json


def run_status(run):
    """The sweep runner's own verdict on a run it just launched: "done" or "failed"."""
    index = json.load(open(f"results/sweeps/{name}/index.json"))
    return next(r["status"] for r in index["runs"] if r["run_name"] == run)


for run in runs:
    finished = os.path.exists(f"results/{run}/run_info.json")
    have_drift = os.path.exists(f"results/{run}/drift.json")
    if finished and (have_drift or not NEEDS_DRIFT):
        print(f"{run}: already in the repo, skipped")
        continue
    # A finished run without drift.json is trained again: the drift measurement needs the
    # trained adapter, which is not stored in git and is gone with the previous runtime.
    # Training deletes the old run_info.json first, so a crash cannot look like a finished run.
    sh(f"python scripts/run_sweep.py --sweep {SWEEP} --only {run}", check=False)
    status = run_status(run)
    sh(f"git add results/sweeps/{name}")
    if status != "done":
        print(
            f"{run} FAILED (status {status}). Its error is printed above. Nothing of this run is committed."
        )
        # Put back the committed copy of this run's files, if an earlier session pushed one.
        subprocess.run(
            f"git ls-files --error-unmatch results/{run} >/dev/null 2>&1 && git checkout -q -- results/{run}",
            shell=True,
        )
        push(f"Record failed sweep run {run} (Colab {gpu})")
        continue
    if NEEDS_DRIFT and sh(f"python scripts/eval_drift.py --run results/{run}", check=False) != 0:
        print(
            f"{run}: training finished but the drift measurement FAILED (error above). The run is "
            "committed without drift.json, and this cell will train it again next time."
        )
    sh(f"git add results/{run}")
    push(f"Add sweep run {run} (Colab {gpu})")

sh("python scripts/plot_sweep.py --sweep label_noise")
sh(f"git add results/sweeps/{name} assets")
push(f"Add {name} sweep figure and summary")

from IPython.display import Image, display

display(Image("assets/m6_sweep_label_noise.png"))
print(open(f"results/sweeps/{name}/index.json").read())
```

## After a sweep

Tell me "pushed" and which sweep. The figure is under `assets/`, the plotted numbers with their bootstrap intervals are in `results/sweeps/<name>/summary.json`, and the status of every run is in `results/sweeps/<name>/index.json`. If a run is marked failed, paste the error that the cell printed for it.

None of these cells has been executed on Colab yet. They follow the same pattern as the single-run cell that has been run there.

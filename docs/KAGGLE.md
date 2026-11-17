# Running the GPU queue on Kaggle

Kaggle gives free GPU time on a separate quota from Colab. This cell runs the same resumable queue as `docs/COLAB_QUEUE.md`: it skips everything already pushed, runs the remaining jobs in priority order, and pushes after each one. Commits made from Kaggle say "Kaggle" instead of "Colab".

## One-time setup

1. A Kaggle account with a **verified phone number** (Kaggle requires it for GPUs and internet access).
2. **Create** > **New Notebook**.
3. In the notebook's right-hand panel, under **Session options**: set **Accelerator** to **GPU T4 x2** (the queue uses one GPU), and switch **Internet** on.
4. **Add-ons** > **Secrets**: add a secret named `GH_TOKEN` with the same GitHub token you use on Colab, and tick it so this notebook can use it.

## The cell

```python
# tiny-rlhf: run ALL remaining GPU jobs on a Kaggle GPU, pushing after each one.
BUDGET_MIN = 690  # Kaggle sessions stop at 12 hours; this leaves a margin
STAGES = ""  # "" means everything that is not done yet, in priority order
BRANCH = "claude/charming-cray-20zxv9"
REPO = "Varkot-dev/tiny-rlhf"

import base64, os, shutil, subprocess, sys, time
import torch
from kaggle_secrets import UserSecretsClient

assert sys.version_info >= (3, 11), f"This project needs Python 3.11+, Kaggle has {sys.version}"
assert torch.cuda.is_available(), "No GPU. Session options > Accelerator > GPU T4 x2."
START = time.time()

# Give git the token through environment variables, so it is never written to disk or printed.
token = UserSecretsClient().get_secret("GH_TOKEN")
auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
os.environ.update(
    {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {auth}",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONUNBUFFERED": "1",
        "QUEUE_PLATFORM": "Kaggle",
    }
)


def run(cmd, show=True, check=True):
    """Run a shell command and print its output live. Returns the exit status."""
    print(f"\n$ {cmd}   [{(time.time() - START) / 60:.1f} min]", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if show and "Warning" not in line and "warnings.warn" not in line:
            print(line, end="", flush=True)
    code = proc.wait()
    if code != 0 and not show:
        print("".join(lines[-40:]))
    if code != 0 and check:
        raise RuntimeError(f"exit {code}: {cmd}")
    return code


WORK = "/kaggle/working/tiny-rlhf"
os.chdir("/kaggle/working")
shutil.rmtree(WORK, ignore_errors=True)
run(f"git clone -q --branch {BRANCH} https://github.com/{REPO}.git {WORK}")
os.chdir(WORK)
run('pip install -q -e ".[train,viz,dev]"')
run("pip uninstall -y -q torchao", check=False)  # an old torchao breaks the LoRA library
run('git config user.name "Varshith Kotagiri"')
run('git config user.email "Varkot-dev@users.noreply.github.com"')
if run("python -m pytest -q -x", show=False, check=False) == 0:
    print("tests passed")
else:
    print("TESTS FAILED (output above). The queue still runs: each job has its own checks.")

left = BUDGET_MIN - (time.time() - START) / 60
stages = f"--stages {STAGES}" if STAGES else ""
code = run(
    f"python scripts/colab_queue.py --budget-min {left:.0f} --branch {BRANCH} {stages}",
    check=False,
)
print("\nQueue finished." if code == 0 else "\nQueue finished; some jobs failed (pushed).")
run("python scripts/colab_queue.py --list")
```

## Running it without keeping the browser open

Kaggle can run a whole notebook in the background: **Save Version** > **Save & Run All (Commit)**. The run continues after you close the tab, up to the 12-hour session limit, and every finished job is still pushed to GitHub as it completes. Check that the secret and internet settings carry over to background runs the first time you use it; this has not been tested from this project.

## Not tested

This cell has not been run on Kaggle. The parts that differ from the tested Colab cell are the secret lookup (`kaggle_secrets`), the working directory and the Python version check. If the Python check fails, Kaggle's image is older than this project supports; tell Claude and it will adapt.

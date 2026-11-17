# Run the remaining GPU work (resumable)

One cell for a free Colab T4. It runs the remaining GPU jobs one after another, in this order, and pushes after every job:

1. **m5:** Track B: base evaluation, training, trained evaluation, then the general regression check, before/after table, hero figure and examples.
2. **m3:** the reward model, then its comparison with `results/dpo_full_t4`.
3. **m2:** TRL parity on the full model in float32. If that runs out of GPU memory, both runs are done again in fp16 under the names `parity_full_fp16` and `parity_full_fp16_trl`.
4. **The four sweeps:** beta, LoRA rank, dataset size, label noise. Each value is one job, followed by the sweep's figure.

Setup: a T4 runtime (Runtime > Change runtime type > T4 GPU) and the `GH_TOKEN` secret, as for the other cells.

How it behaves:

- **Resuming.** Rerunning the cell resumes. Every finished job is already in the repo, so a fresh clone skips it. You can rerun after a disconnect, the next day, or on another account's runtime.
- **Budget.** `BUDGET_MIN` is the time you have. A job is not started if its estimate does not fit in what is left. The estimates are rough guesses; the only measured run is the full DPO run, 463 optimizer steps in 1,737 s with an 8.8 GB peak on a T4. Total time for the whole queue is not measured. Run `python scripts/colab_queue.py --list` to see each job's estimate.
- **Selecting stages.** `STAGES` picks what to run, for example `"m3"` or `"beta,lora_rank"`. Leave it empty for everything.
- **Failures.** A job that fails is recorded and does not stop the queue, and nothing half-finished is committed. The end of its output is pushed to `results/colab_failures/<job>.log` and `results/queue_status.json`, so you do not need to copy anything. Just say "pushed" or "it stopped".
- **Out of memory.** A training job that runs out of GPU memory is retried once with a smaller micro-batch; the effective batch stays the same.

```python
# tiny-rlhf: run the remaining GPU jobs on a Colab T4, pushing after each one.
BUDGET_MIN = 55  # minutes you have; jobs that will not fit are not started
STAGES = ""  # "" for everything, or for example "m3" or "beta,lora_rank"
BRANCH = "claude/charming-cray-20zxv9"
REPO = "Varkot-dev/tiny-rlhf"

import base64, os, shutil, subprocess, time
import torch
from google.colab import userdata

assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then run again."
START = time.time()

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


os.chdir("/content")
shutil.rmtree("/content/tiny-rlhf", ignore_errors=True)
run(f"git clone -q --branch {BRANCH} https://github.com/{REPO}.git /content/tiny-rlhf")
os.chdir("/content/tiny-rlhf")
run('pip install -q -e ".[train,viz,dev]"')
run("pip uninstall -y -q torchao")  # Colab's old torchao breaks the LoRA library; unused here
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

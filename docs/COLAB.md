# Running training on Google Colab

The development container has no GPU, so full training runs happen on a free Colab T4. The notebook `notebooks/colab_train.ipynb` clones this repo, runs one config, makes the figure, and pushes the results back, so every number is produced by a script in the repo and saved under `results/`.

## One-time setup

### 1. Create a GitHub token for this repo only

1. On GitHub: your profile picture > Settings > Developer settings > Personal access tokens > Fine-grained tokens > Generate new token.
2. Repository access: "Only select repositories", then pick `Varkot-dev/tiny-rlhf`.
3. Permissions > Repository permissions > Contents: **Read and write**. Leave everything else at "No access" (Metadata read is added automatically).
4. Set an expiry you are comfortable with, generate, and copy the token. GitHub shows it once.

This token can read and write the contents of this one repository and nothing else. Do not paste it into a notebook cell or a chat.

### 2. Add the secrets in Colab

1. In Colab, click the key icon in the left sidebar to open the Secrets panel.
2. Add a secret named `GH_TOKEN` with the token as its value.
3. Switch on "Notebook access" for it. Secrets are stored per Google account, but access is granted per notebook, so this switch has to be on for each notebook that uses the secret.
4. Optional: add `HF_TOKEN` (a Hugging Face read token) the same way. The models and dataset in the current configs were not gated when checked on 2026-11-17, so the run works without it.

The notebook reads them with `google.colab.userdata.get(...)` and never prints them. The GitHub token is handed to git through environment variables in the notebook's process; it is not written into the clone's `.git/config`.

### 3. Open the notebook

The repo is private, so use one of these:

- **From Colab:** File > Open notebook > GitHub tab, tick "Include private repos", authorize Colab in the popup (allow pop-ups for colab.research.google.com if nothing appears), then pick the repo, the branch `claude/charming-cray-20zxv9`, and `notebooks/colab_train.ipynb`.
- **Without authorizing Colab on GitHub:** download `notebooks/colab_train.ipynb` from GitHub and use File > Upload notebook.

Either way Colab opens its own copy. The notebook itself never needs saving back to GitHub; only the results are pushed, by the last step.

### 4. Select the GPU

Runtime > Change runtime type > T4 GPU > Save. The first cell stops with a message if there is no GPU.

## Each run

1. In the Parameters cell set `CONFIG`. Do `configs/dpo_tiny.yaml` first as a short dry run, then `configs/dpo_full.yaml`.
2. Runtime > Run all.
3. Check that the first logged training loss is about 0.693 (log 2). If it is not, stop the run and tell me what it printed.
4. When the last cell prints `pushed <hash> ...`, the results are on the branch.

## What a run writes

Under `results/<run_name>/` (committed and pushed):

| File | Contents |
|---|---|
| `train.csv` | per-step loss, reward margin, preference accuracy, preferred-response log-probability |
| `eval.csv` | the same metrics on held-out pairs at each evaluation |
| `config.yaml` | the exact config the run used |
| `run_info.json` | measured wall-clock time, measured peak GPU memory, device, library versions, git commit |

Under `assets/` (committed and pushed): the training figure made by `scripts/plot_training.py`.

Under `checkpoints/<run_name>/` (not in git): the LoRA adapter. It is lost when the Colab runtime is recycled unless you run the optional Drive cell at the end of the notebook.

Time and memory are not assumed anywhere: the run measures them and records them in `run_info.json`, and those recorded values are what the README will quote.

## What to tell me afterwards

Just: "pushed run `<run_name>`". I will pull the branch and read the results from the repo.

## If something goes wrong

- **"Could not read the GH_TOKEN secret"**: the secret is missing, misnamed, or its notebook access switch is off.
- **Clone fails with 403 or "repository not found"**: the token does not cover this repository or lacks Contents read and write, or it has expired.
- **Out of memory**: tell me the config and the error. Do not edit the config in Colab, because the change would not be in the repo.
- **Colab disconnects mid-run**: nothing is pushed until the last step. Run all again.
- **Push rejected**: the notebook merges the latest branch before pushing. If the merge itself reports a conflict, tell me and do not force-push.

# RunPod: booking and running

Written for a first cloud run. Optimised for spending as little as possible.

The headline: **the smoke test is ~20 minutes and costs about $1.** The
expensive mistake is not the GPU, it is leaving a pod running after you stop
paying attention to it.

---

## What to book

**1x H100 80GB**, Community Cloud, container disk 20–30 GB, volume disk 40 GB,
**no network volume**.

| setting | value | why |
|---|---|---|
| GPU | 1x H100 80GB (SXM or PCIe) | SXM is faster; either is fine |
| count | **1** | the code is single-GPU; a second card does nothing |
| Cloud type | Community | cheaper; a 20-minute smoke does not need Secure Cloud's SLA |
| Template | RunPod PyTorch 2.x (CUDA 12.x) | torch preinstalled and driver-matched |
| Container Disk | **20–30 GB** | writable root layer; only pip (~1.5 GB) lands here |
| Volume Disk (`/workspace`) | **40 GB** | HF cache + runs; 6.1 GB actually used |
| Network volume | **none** | see "storage" below |

### Book the H100 even for the smoke test

Tempting to smoke on something cheap. Don't. Stage 2 of the smoke exists to
prove the **memory budget** fits, and a memory check on a different card proves
nothing about the card you will actually use. The H100 costs roughly $0.03/min;
validating on the real hardware is worth a dollar.

If H100s are unavailable, A100 80GB is the fallback (same memory budget, ~2.5x
slower). **Nothing under 48 GB will work** — the training state alone is
27.4 GB before a single activation.

---

## Storage: do not buy a network volume

RunPod has two kinds of disk, and the difference matters for cost:

- **Container disk** — ephemeral, included in the pod, destroyed when the pod is
  terminated. Billed only while the pod exists.
- **Network volume** — persistent, mounted at `/workspace`, and **keeps billing
  for as long as it exists**, including while no pod is running. It is not
  rentable "for an hour" in any useful sense: you create it, it bills until you
  delete it.

For a 20-minute smoke test that produces ~1.2 GB of results you want to keep,
the network volume is pure overhead. **Use the pod's own volume disk and rsync
the results to your laptop before terminating.**

A network volume only starts to pay for itself if you are doing many runs over
days and don't want to re-download the 3.4 GB base model each time.

### Your 512 GB SSD is more than enough

Everything worth keeping is small:

| artefact | size |
|---|---|
| smoke telemetry (both stages) | ~1.2 GB |
| real-run telemetry (tulu3-50k) | ~2 GB |
| final bf16 model, if you want it | 3.4 GB |

What you do **not** pull down: checkpoints (20.5 GB each) and the base model
(re-downloadable in a minute). Total local footprint across the whole project is
well under 10 GB.

### On-pod footprint, checkpointing off

```
3.40 GB  SmolLM2-1.7B base (HF cache)
1.50 GB  pip packages
0.80 GB  step-0 fp32 weight baseline
0.38 GB  stage-2 deep dumps
0.02 GB  telemetry scalars
0.01 GB  dolly-15k
-------
6.11 GB  -> all of it on /workspace; a 40 GB volume disk has ample room
```

---

## You were right about checkpointing

It is now **off by default for both smoke stages** (`run/smoke.yaml` and
`run/preflight.yaml` set `save_strategy: "no"`). A full-parameter checkpoint of a
1.7B model is 20.5 GB — fp32 weights plus two fp32 Adam moments — to insure a
run that takes three minutes to repeat. `save_strategy` did not previously exist
as a config field; it was hardcoded to `"steps"`. It exists now.

For the real 39-minute run, `save_total_limit: 2` (41 GB) is the default. Even
that is arguably generous for a run that short — set `-s save_strategy=no` if
you would rather just rerun on failure.

---

## Do we need hyperparameter tuning?

**No.** That is the entire point of the pivot, and it is now settled:

- Model, dataset and both hyperparameters come from the same published row
  (SmolTulu SFT-1207: LR 3.1e-6, batch 32, on SmolLM2-1.7B over
  `allenai/tulu-3-sft-mixture`). Nothing is extrapolated.
- `per_device_batch` and `grad_accum` are sizing, not tuning — their product is
  constrained to equal the recipe's batch size, and the config refuses to load
  if it cannot be honoured exactly.
- The caveats that remain are documented in the README's "Known limits", not
  hidden: no output-side dimension lever on this model, a math/code-heavy
  mixture, and dolly being a plumbing test rather than a second arm.

The one thing that is *not* recipe-faithful is dolly itself, which is why no
scientific claim comes out of the smoke test.

---

## The run

### Before you book: nothing to do

The repo is ready. `python scripts/preflight.py` passes locally today (packages,
all three config compositions, disk). The only unverified things are
CUDA-specific — actual memory fit and the bf16 autocast path — which is exactly
what you are booking the GPU to check.

### 0. Before you click Deploy

- **Set a spending limit** (Settings → Billing). This is the guardrail that
  actually works: enforced by RunPod, needs no process of yours to stay alive.
- **Enable SSH Terminal Access** on the pod config, and after it boots use the
  **direct TCP** connection string (`ssh root@<IP> -p <PORT>`), not the
  `ssh.runpod.io` proxy. `rsync` needs direct TCP.

Disk fields on the deploy page:

| field | set to | why |
|---|---|---|
| Container Disk | 20–30 GB | the writable root layer; only pip (~1.5 GB) lands here |
| Volume Disk (`/workspace`) | 40 GB | everything heavy goes here: HF cache, runs |

40 GB volume against a 6.1 GB footprint is plenty.

### 1. Connect and start tmux

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
tmux new -s run       # detach: Ctrl-b then d     reattach: tmux attach -t run
```

**tmux first.** A dropped SSH session kills the training process otherwise.

### 2. Push the repo from your LAPTOP

This repo has **no git remote**, so `git clone` on the pod cannot work. The
source is 0.4 MB — rsync is faster than creating a GitHub repo:

```bash
# in a second terminal ON YOUR LAPTOP
bash scripts/push_repo.sh <POD_IP> <PORT>
```

### 3. One paste, on the pod

```bash
cd /workspace/sftlens && bash scripts/runpod_smoke.sh
```

That does everything and stops: installs (using the image's torch, not a 2.5 GB
reinstall), runs `preflight.py`, runs the 129 tests, runs stage 1, runs stage 2,
verifies the archive, prints sizes.

**Billing safety net:** set a spending limit in the RunPod console
(Settings → Billing) before you start -- it is enforced by RunPod itself and
needs no process of yours to stay alive, which beats any script-based timer.

### 4. Read four things

1. **The masking dump.** System and user turns under "masked", the assistant
   answer and its `<|im_end|>` under "supervised". This is the bug a loss curve
   will never show you.
2. **`[recipe]`** → `lr=3.1e-06 effective_batch=32 (= 4 x 8) LR/BS=0.097e-6`.
3. **`[env] memory budget`** → the four-line breakdown. The `logits peak` line
   reports the figure for the *real* run's longest sequence, which dolly does
   not exercise. Read it there rather than assuming stage 2 passing means the
   real run fits.
4. **`ARCHIVE OK`** at the end, with a `PR_Sigma/null ratio` well below 1.

### 5. Get the results out, then terminate

From your laptop:

```bash
bash scripts/pull_artifacts.sh <POD_IP> <PORT>
```

This copies the telemetry home (skipping checkpoints and the base model) and
then **verifies it locally** — shard counts, deep dumps, manifests, and a
finite-value check across every substrate. Wait for:

```
  SAFE TO TERMINATE THE POD
```

Only then **Terminate** the pod — not Stop. A stopped pod still bills for its
disk.

If direct TCP is unavailable and rsync will not connect, fall back to RunPod's
peer-to-peer transfer: `runpodctl send /workspace/runs` on the pod, then
`runpodctl receive <code>` on your laptop.

---

## Time and cost

| step | time |
|---|---|
| deploy + SSH in | ~2 min |
| push repo (0.4 MB) | ~10 s |
| install | ~3 min |
| preflight + 129 tests | ~1 min |
| stage 1 smoke (incl. 3.4 GB model download) | ~3 min |
| stage 2 preflight (production telemetry sizing) | ~4 min |
| rsync + terminate | ~3 min |
| **total** | **~16 min, about $1** |

The later real run (tulu3-50k, SFT-1207) is ~39 min on the same card, about $2.

---

## If something goes wrong

| symptom | fix |
|---|---|
| `CUDA out of memory` during a probe | `-s telemetry.accum_device=cpu`, or `-s telemetry.n_tokens=4096` (halves the 12.3 GB accumulator) |
| OOM during the forward | `-s recipe.per_device_batch=2` — grad_accum compensates automatically, the recipe is unchanged |
| `preflight.py` fails on VRAM | wrong GPU booked; you need >= 48 GB |
| run dies, you want it back | rerun the identical command; it resumes from the last checkpoint and the telemetry archive is additive |
| SSH drops mid-run | `tmux attach -t run` — this is why tmux is step one |
| `git clone` fails | expected: there is no remote. Use `push_repo.sh` from your laptop |
| rsync cannot connect | you are on the `ssh.runpod.io` proxy; use the direct TCP host/port, or `runpodctl send` |
| pod idle and you walked away | the account spending limit (set it before you start) is the backstop |

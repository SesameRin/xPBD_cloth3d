# Batch mode

`xpbd.batch` runs the simulator on every CLOTH3D sample under
`cloth3d/Samples/` (or any subset you pick) and writes all of their
eval-compatible NPZ files into a single timestamped folder. It is
deliberately **simulation-only** — it does not call the teammate's
`cloth3d_benchmark/eval_metrics.py`, because the eval expects a conda
env (e.g. `uipc_env`) that the simulator doesn't need.

So the workflow is two phases:

1. **Generate NPZs** with `python3 -m xpbd.batch` in your normal env.
2. **Run the eval** under `uipc_env`, pointed at the batch folder.

## What the batch writes

```
{out}/batch_{YYYYMMDD_HHMMSS}/
├── results_xpbd/           ← per-garment sim NPZs
│   ├── 00016_Tshirt/00016_Tshirt_sim.npz
│   ├── 00016_Trousers/00016_Trousers_sim.npz
│   └── ...
├── cloth3d_data/           ← per-sample CLOTH3D NPZs
│   ├── 00016.npz
│   └── ...
├── logs/                   ← per-sample stdout/stderr capture
│   ├── 00016.log
│   └── ...
└── summary.txt             ← one-line header + CSV of sample, status, elapsed, rc
```

Each sample runs in its **own `python -m xpbd` subprocess**, so Taichi
state from one sample cannot leak into the next. Failures in one sample
don't abort the batch — they're recorded in `summary.txt` and the
offending sample's log.

## CLI

`python3 -m xpbd.batch [flags]`

| Flag | Default | Meaning |
|---|---|---|
| `--samples_dir` | `cloth3d/Samples` | where sample subfolders live |
| `--samples` | all found | comma-separated subset (e.g. `00016,07414`) |
| `--out` | `xpbd_out` | parent dir for the new `batch_*` folder |
| `--batch_name` | `batch_{ts}` | override the folder name |
| `--garments` | `all` | passed through to each sample run |
| `--arch` | `cpu` | `cpu` / `gpu` / `vulkan` |
| `--body_frames` | auto | omit or `-1` → full sample length per sample |
| `--steps` | auto | omit or `-1` → matches `--body_frames` |
| `--dt` | solver default | passthrough |
| `--substeps` | solver default | passthrough |
| `--iters` | solver default | passthrough |
| `--force_fabric` | off | passthrough (cotton/silk/denim/leather) |
| `--garment_y_translation` | `0.0` | drop-experiment lift in metres along z; passed through to *every* sample. Use `3.0` to mirror partner's drop. See [`drop_experiment.md`](drop_experiment.md). |
| `--save_sample_npz` | **on** | extract the per-sample CLOTH3D NPZ the eval reopens |
| `--no_save_sample_npz` | — | opt out of the per-sample NPZ |
| `--timeout` | no limit | per-sample subprocess timeout in seconds |
| `--python` | `sys.executable` | python executable used to spawn each run |

Because each sample runs end-to-end in one subprocess, the solver picks
up its own fabric-per-garment presets automatically — no batch-level
fabric juggling is needed unless you want to `--force_fabric` everything
for a controlled comparison (e.g. the C-IPC cotton baseline).

## End-to-end recipe

### Phase 1: simulate everything

```bash
# Default: run every sample at full length, auto-sized per sample,
# on GPU, with per-sample CLOTH3D NPZ extraction on.
python3 -m xpbd.batch --arch gpu

# Or a subset, with a timeout guard so a hung sample doesn't stall the run:
python3 -m xpbd.batch --arch gpu --samples 00016,07414 --timeout 900

# Or the C-IPC cotton comparison run:
python3 -m xpbd.batch --arch gpu --force_fabric cotton

# Drop-experiment batch (every sample, lifted 3 m above the body):
python3 -m xpbd.batch --arch gpu --garment_y_translation 3.0
```

The batch prints the exact `batch_*` folder it created and the
`results_root` / `cloth3d_npz_path` paths you feed to the eval.

> **One batch = one regime.** `--garment_y_translation` is propagated
> uniformly to every sample subprocess. To compare drop vs worn-cloth,
> run two separate batches into two separate folders (e.g. via
> `--batch_name worn` and `--batch_name drop`), then run the eval twice.
> Mixing both regimes in one folder is unsupported — the eval has a
> single `--garment_y_translation` and `--experiment_protocol` flag per
> invocation and would mis-interpret half the runs. (The `_sim.npz`
> files do record their own `garment_y_translation` field, so a future
> eval-side improvement could autodetect; today the safer path is two
> batches.)

### Phase 2: run the teammate's eval (different env)

Switch to whichever env has `libuipc` (e.g. `uipc_env`) and point the
benchmark at the two subdirs the batch wrote:

```bash
conda activate uipc_env
cd /home/ula/CMU/pba-proj/cloth3d-ipc-xpbd/cloth3d_benchmark

# Fill in the batch_{ts} the phase 1 run printed.
BATCH=/home/ula/CMU/pba-proj/xPBD/xpbd_out/batch_20260421_164325

python3 eval_metrics.py \
  --results_root   "$BATCH/results_xpbd" \
  --cloth3d_npz_path "$BATCH/cloth3d_data" \
  --output_dir     "$BATCH/eval_outputs"
```

Batch eval discovers every `*_sim.npz` under `results_xpbd/`, matches it
to `{cloth3d_npz_path}/{sample}.npz`, and writes per-run JSON plus a
`summary.csv`.

### Drop-batch eval

If phase 1 ran with `--garment_y_translation 3.0`, append the two
drop-protocol flags to phase 2 (matches partner's command):

```bash
python3 eval_metrics.py \
  --results_root   "$BATCH/results_xpbd" \
  --cloth3d_npz_path "$BATCH/cloth3d_data" \
  --cloth_reference_shape rest \
  --garment_y_translation 3.0 \
  --experiment_protocol drop \
  --output_dir     "$BATCH/eval_outputs_dropping"
```

`--experiment_protocol drop` only changes warning semantics; it is safe
to omit but you'll get spike-based outputs that the eval otherwise
flags as not meaningful for free-fall runs. See
[`drop_experiment.md`](drop_experiment.md) and
[`eval_export.md`](eval_export.md) for the full flag reference.

## Why subprocess-per-sample

Taichi fields are module-global and allocated on solver construction.
Building a new `XPBDCloth` in the same process on top of a previous
sample's fields can leak memory and, with different mesh sizes, bloat
the Taichi kernel cache. A fresh process per sample is the simplest
guarantee of isolation and makes `--timeout` a no-brainer: kill one bad
sample without risking the rest.

## Troubleshooting

- **A sample failed, the rest succeeded.** Open
  `{batch}/logs/{sample}.log`. `summary.txt` also records the return
  code. Fix the underlying issue and rerun just that sample with
  `--samples {id}` into the same `--batch_name` to slot its outputs
  alongside the rest.
- **The whole batch is too slow.** Drop `--arch cpu` for `--arch gpu`.
  If your GPU refuses one sample, use `--timeout 900` so it doesn't
  stall the queue and inspect the log.
- **Eval can't find `{sample}.npz`.** Either the batch ran with
  `--no_save_sample_npz`, or you pointed `--cloth3d_npz_path` at the
  wrong folder. The batch prints both paths at the end.

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
```

The batch prints the exact `batch_*` folder it created and the
`results_root` / `cloth3d_npz_path` paths you feed to the eval.

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

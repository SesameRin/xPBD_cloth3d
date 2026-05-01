# Eval-compatible NPZ export

xPBD can write its simulation trajectories in the NPZ schema that the
teammate's
[`cloth3d-ipc-xpbd/cloth3d_benchmark`](../../cloth3d-ipc-xpbd/cloth3d_benchmark)
eval module reads. The goal: run our solver, drop the resulting files
into the benchmark's results directory, and get full run-level metrics
(stability, stretch, shear, cloth-body contact, time-per-frame) without
any bridging code.

## What gets written

Two kinds of NPZ files, both produced by `xpbd.export`:

1. **Per-garment result file** — one per garment simulated, written by
   `write_result_npz(...)`:
   ```
   {npz_out}/{sample}_{garment}/{sample}_{garment}_sim.npz
   ```
   Payload matches `result_format_version = 2`, `solver_family = "xpbd"`
   and the full field set consumed by
   `cloth3d_eval.io_result.load_result_run`: `sim_V_seq`, `gt_V_seq`,
   `faces`, `human_V_seq`, `human_faces`, `frame_wall_time_ms_seq`,
   `frame_dt`, `start_frame`, `simulated_frames`, `effective_gravity`,
   `scene_mode`, `cloth_reference_shape`, and all bookkeeping fields.

2. **Per-sample CLOTH3D source file** — one per sample, written by
   `write_sample_npz(...)`:
   ```
   {sample_npz_dir}/{sample}.npz
   ```
   This is the exact CLOTH3D extraction the eval's source loader
   reopens via `cloth3d_sim.data.convert_from_cloth3d`. We produce it
   by calling `cloth3d/Demo/extract_sample_data.extract_sample_all_frames`
   — the same extractor the teammate copied into `code_to_copy/`, so
   the files are byte-for-byte compatible.

Only (1) is strictly required. (2) is a convenience so the benchmark
repo doesn't need its own extraction pass.

## Coordinate frame

CLOTH3D is z-up, libuipc / the eval pipeline is y-up. `write_result_npz`
rotates every written vertex sequence — `sim_V_seq`, `gt_V_seq`,
`human_V_seq` — from z-up to y-up using the same mapping the eval uses:
`(x, y, z) → (x, z, -y)`. `write_sample_npz` keeps raw CLOTH3D z-up
because the eval's source loader does the rotation itself when it
reopens the sample file.

## CLI flags

Added in `xpbd/cli.py`:

| flag | default | role |
|---|---|---|
| `--save_npz`         | off       | run in export mode; writes per-garment `_sim.npz` |
| `--npz_out DIR`      | `xpbd_out/results_xpbd` | where the per-garment files go |
| `--save_sample_npz`  | off       | also extract the per-sample CLOTH3D NPZ |
| `--sample_npz_dir DIR` | `xpbd_out/cloth3d_data` | where `{sample}.npz` goes |
| `--garment_y_translation T` | `0.0` | lift cloth (`V0` and per-frame GT) by `T` m along z; recorded in the result NPZ so the eval picks the same offset up via `run.extra`. Use `3.0` for partner's drop run. |

`--save_npz` overrides the viewer choice: it drives a headless sim via
`viewers.run_export` that steps once per frame, snapshots `cloth.x`,
measures wall time per frame, and calls `write_result_npz` once per
garment. When `--save_npz` is set, `data.load_sample` is called with
`need_gt_trajectory=True`, which loads per-garment CLOTH3D ground-truth
positions for the same frame range (so `gt_V_seq` is populated).

**Run order inside `run_export`:**
1. If `--save_sample_npz` is set, extract the per-sample CLOTH3D NPZ
   **first** (fails fast if the extractor is broken).
2. Step the solver and snapshot `cloth.x` per frame.
3. Slice per garment and write `{sample}_{garment}_sim.npz`.

**Auto-sized runs.** `--body_frames` and `--steps` both default to the
full sample length (`get_num_frames(sample)` — 300 for `00016`). You can
omit them entirely and xPBD will simulate the whole sequence; pass `-1`
explicitly to do the same thing.

## Batch mode

For running every sample at once, see [`batch.md`](batch.md). The batch
driver wraps this export path in a subprocess-per-sample loop and drops
everything into a single timestamped folder that the eval then consumes
as one `--results_root`.

## End-to-end example

```bash
# 1. Simulate + export (body_frames and steps auto-default to the
#    sample's full frame count — 300 for 00016).
python3 xpbd_cloth.py \
  --sample 00016 --garments all \
  --arch gpu --save_npz --save_sample_npz

# Produces:
#   xpbd_out/results_xpbd/00016_Tshirt/00016_Tshirt_sim.npz
#   xpbd_out/results_xpbd/00016_Trousers/00016_Trousers_sim.npz
#   xpbd_out/cloth3d_data/00016.npz

# 2. Point the teammate's eval at those directories
  # Single run:
  python3 eval_metrics.py \
    --result_npz /home/ula/CMU/pba-proj/xPBD/xpbd_out/results_xpbd/00016_Tshirt/00016_Tshirt_sim.npz \
    --cloth3d_npz_path /home/ula/CMU/pba-proj/xPBD/xpbd_out/cloth3d_data \
    --output_dir ./eval_outputs

  # Or batch (all garments under the results root):
  python3 eval_metrics.py \
    --results_root /home/ula/CMU/pba-proj/xPBD/xpbd_out/results_xpbd \
    --cloth3d_npz_path /home/ula/CMU/pba-proj/xPBD/xpbd_out/cloth3d_data \
    --output_dir ./eval_outputs


cd ../cloth3d-ipc-xpbd/cloth3d_benchmark
python3 eval_metrics.py \
  --results_root /path/to/xPBD/xpbd_out/results_xpbd \
  --cloth3d_npz_path /path/to/xPBD/xpbd_out/cloth3d_data \
  --output_dir ./eval_outputs
```

Batch eval discovers `*_sim.npz` recursively, matches each one to
`{cloth3d_npz_path}/{sample}.npz`, and writes per-run JSON + a summary
CSV. Single-run mode via `--result_npz path/to/foo_sim.npz` also works.

## Drop runs: extra eval flags

If you exported with `--garment_y_translation 3.0`, the eval needs two
extra flags to interpret the result correctly. They mirror partner's
drop command exactly:

| eval flag | value | role |
|---|---|---|
| `--garment_y_translation` | match the export (e.g. `3.0`) | tells the eval's source loader to lift the source-side GT by the same amount the exporter lifted `sim_V_seq` / `gt_V_seq`. The eval also reads this from `run.extra` in the result NPZ as a fallback, so you can omit it if the file was exported by current xPBD; passing it explicitly is the safe match for partner's command. |
| `--experiment_protocol` | `drop` | switches warning/info semantics: suppresses the "Numerical temporal instability" failure-taxonomy label and tags velocity / acceleration spike outputs as not meaningful for a free fall. Does *not* change metric formulas. |
| `--cloth_reference_shape` | `rest` | use the un-draped CLOTH3D rest mesh as the stretch/shear reference. Already the default in our exporter. |

End-to-end, drop variant:

```bash
# 1. Simulate + export under drop protocol (one sample)
python3 xpbd_cloth.py \
  --sample 00007 --garments Tshirt \
  --arch gpu --garment_y_translation 3.0 \
  --save_npz --save_sample_npz

# 2. Eval the drop run
cd ../cloth3d-ipc-xpbd/cloth3d_benchmark

# Single run:
python3 eval_metrics.py \
  --result_npz /home/ula/CMU/pba-proj/xPBD/xpbd_out/results_xpbd/00007_Tshirt/00007_Tshirt_sim.npz \
  --cloth3d_npz_path /home/ula/CMU/pba-proj/xPBD/xpbd_out/cloth3d_data \
  --cloth_reference_shape rest \
  --garment_y_translation 3.0 \
  --experiment_protocol drop \
  --output_dir ./eval_outputs_xpbd_dropping

# Or batch (all drop runs at once):
python3 eval_metrics.py \
  --results_root /home/ula/CMU/pba-proj/xPBD/xpbd_out/results_xpbd \
  --cloth3d_npz_path /home/ula/CMU/pba-proj/xPBD/xpbd_out/cloth3d_data \
  --cloth_reference_shape rest \
  --garment_y_translation 3.0 \
  --experiment_protocol drop \
  --output_dir ./eval_outputs_xpbd_dropping
```

See [`drop_experiment.md`](drop_experiment.md) for the full drop recipe
(simulator side + how it lines up with partner's IPC drop), and
`compare_eval_outputs.py` for cross-solver plots once both
`eval_outputs_ipc_dropping` and `eval_outputs_xpbd_dropping` exist.

## Schema crib sheet

The fields `write_result_npz` emits, and where they come from in the
simulator:

| key | source |
|---|---|
| `sim_V_seq`      | `cloth.x` snapshotted after each `step()`, sliced per garment, rotated z-up → y-up |
| `gt_V_seq`       | `reader.read_garment_vertices(sample, garment, frame)` for each frame, rotated z-up → y-up |
| `faces`          | re-indexed per-garment face list (0..N_g-1), from `slice_garment` |
| `human_V_seq`    | `data.body_V_seq[:n_frames]`, rotated z-up → y-up |
| `human_faces`    | `data.body_F` (SMPL faces) |
| `frame_dt`       | `args.dt` — one sim step == one frame |
| `frame_wall_time_ms_seq` | `time.perf_counter()` around `cloth.step()` plus `ti.sync()` |
| `effective_gravity` | `(0, -9.81, 0)` — the y-up equivalent of the z-up gravity the solver uses internally |
| `solver_family`, `solver_name` | `"xpbd"`, `"xPBD-Taichi-CMU"` |

Anything the eval reads but we don't produce (wind, scene_mode variants,
contact_tolerance) gets a benign default.

## Why this lives in `xpbd.export`

The writer is a pure NumPy module with no Taichi dependency — so unit
tests and other tools can build an NPZ from any `(V_seq, F, ...)`
without spinning up the solver. `viewers.run_export` is the thin glue
that takes the live `cloth` object, does the per-frame bookkeeping, and
calls the writer.

# Drop experiment (matching partner's IPC drop run)

A control experiment in which the cloth starts a few metres above the
body in T-pose and falls under gravity. There is no initial penetration,
so the contact kernel should not explode — this is the "good"
counterfactual for the worn-cloth contact-blow-ups documented in
[`contact_explosion_cloth3d.md`](../contact_explosion_cloth3d.md) (if
that file is present in your tree).

The experiment was added on the partner's side in commit `9ff9fa3`
("dropping experiment") of `cloth3d-ipc-xpbd`. xPBD now mirrors it via a
single CLI flag, `--garment_y_translation`.

## What the flag does

- Lifts every cloth vertex (`V0` and the per-frame GT used by export)
  by `t` metres along **CLOTH3D's z-axis**.
- The body (`body_V_seq`), the un-draped rest mesh
  (`garment_V_rest`), and edge rest-lengths are **not** lifted —
  translation preserves distances, so XPBD's distance/bending
  constraints are unchanged.
- The same offset is recorded in the per-garment `_sim.npz` as
  `garment_y_translation`. Eval reads this back via
  `cloth3d_eval/io_source.py:46`
  (`run.extra.get("garment_y_translation", …)`) so the source-side GT
  it reopens from `{sample}.npz` is lifted to the same frame as our
  already-lifted `sim_V_seq` / `gt_V_seq`.
- Whenever `--garment_y_translation` is non-zero the CLI also enables
  `--freeze_body`, which holds the SMPL body collider at frame 0 for
  the entire run. This mirrors partner's `--freeze_human_mesh on`: the
  body is a static T-pose that the cloth falls onto, not an animated
  body. Pass `--no_freeze_body` to override.

## Why z-up `+t` matches partner's y-up `+t`

The eval works in libuipc's y-up frame; xPBD runs in CLOTH3D's z-up
frame. The shared rotation is `(x, y, z) → (x, z, -y)`, so `y'` in the
eval frame is `z` in the CLOTH3D frame. Adding `+t` to `y'` is the same
as adding `+t` to the original `z`. That is why we apply the lift to
the `z` component on the CLOTH3D side and the recorded
`garment_y_translation` value travels through unchanged.

The shift is applied:
- to `V0` and per-frame GT in [`xpbd/data.py:load_sample`](../xpbd/data.py).
- to the saved `garment_y_translation` field in
  [`xpbd/export.py:write_result_npz`](../xpbd/export.py).

## Run an xPBD drop

Single sample (sample `00007`, just the Tshirt — partner's example):

```bash
python3 xpbd_cloth.py \
    --sample 00007 --garments Tshirt \
    --arch gpu \
    --garment_y_translation 3.0 \
    --save_npz --save_sample_npz \
    --npz_out xpbd_out/results_xpbd_drop \
    --sample_npz_dir xpbd_out/cloth3d_data
```

Whole batch (every sample under `cloth3d/Samples/`):

```bash
python3 -m xpbd.batch \
    --arch gpu \
    --garment_y_translation 3.0
```

`xpbd.batch` propagates `--garment_y_translation` into every per-sample
subprocess and writes one timestamped folder under `xpbd_out/`.

## Eval: comparing the result against partner's drop run

The eval lives in `cloth3d-ipc-xpbd/cloth3d_benchmark` and imports
`uipc` (libuipc) at module load. Run it from the partner's `libuipc`
Python env, **not** from this repo's env. xPBD's only job is producing
the `_sim.npz` and `{sample}.npz` files; the comparison runs over there.

Once the run finishes, point the partner's eval at the xPBD output and
their IPC output. Use `--experiment_protocol drop` and the matching
offset:

```bash
cd ../cloth3d-ipc-xpbd/cloth3d_benchmark

# Single result file
python3 eval_metrics.py \
    --result_npz /path/to/xPBD/xpbd_out/results_xpbd_drop/00007_Tshirt/00007_Tshirt_sim.npz \
    --cloth3d_npz_path /path/to/xPBD/xpbd_out/cloth3d_data \
    --cloth_reference_shape rest \
    --garment_y_translation 3.0 \
    --experiment_protocol drop \
    --output_dir ./eval_outputs_xpbd_dropping

# All xPBD drop runs at once
python3 eval_metrics.py \
    --results_root /path/to/xPBD/xpbd_out/results_xpbd_drop \
    --cloth3d_npz_path /path/to/xPBD/xpbd_out/cloth3d_data \
    --cloth_reference_shape rest \
    --garment_y_translation 3.0 \
    --experiment_protocol drop \
    --output_dir ./eval_outputs_xpbd_dropping
```

`--experiment_protocol drop` only changes warning semantics (it does
not change metric formulas). It suppresses the failure-taxonomy
"Numerical temporal instability" label and tags Velocity / acceleration
spike outputs as not meaningful, because their reference trajectory is
the worn-cloth GT — undefined for a free fall. Contact-quality metrics
remain valid. See `cloth3d_eval/pipeline/eval_run.py` (functions
`protocol_warnings` / `protocol_infos`) for the exact text.

After both `eval_outputs_ipc_dropping` and `eval_outputs_xpbd_dropping`
exist, the partner's `compare_eval_outputs.py` produces the side-by-side:

```bash
python3 compare_eval_outputs.py \
    --ipc_eval_path ./eval_outputs_ipc_dropping \
    --xpbd_eval_path ./eval_outputs_xpbd_dropping \
    --max_plot_samples 10
```

## Matching protocol checklist

To get a like-for-like comparison against the partner's drop:

1. Same sample id (`00007` in partner's example command).
2. Same garment(s) — partner's command uses `--garment_name Tshirt`.
3. Same offset on both sides: `--garment_y_translation 3.0`.
4. `--cloth_reference_shape rest` on both sides — the rest mesh is
   un-draped and is **not** lifted, so it stays the same reference.
5. Eval invoked with `--experiment_protocol drop`.

Initial cloth state matches: partner's IPC drop initialises with
`cloth_gt_window[0]`, which is `_V_seq_y[0]` (worn frame-0) **lifted by
+t** in y-up. Our `V0` (worn frame-0) lifted by `+t` in z-up is the same
mesh in the same world location.

## Sanity check the export

After running, you can spot-check the recorded offset and that the
cloth really sits above the body:

```python
import numpy as np
d = np.load("xpbd_out/results_xpbd_drop/00007_Tshirt/00007_Tshirt_sim.npz")
print(d["garment_y_translation"])         # 3.0
print(d["sim_V_seq"][0].mean(0)[1] -      # cloth y-mean
      d["human_V_seq"][0].mean(0)[1])     # body  y-mean  → ~+3
```

"""Batch driver: simulate every sample in ``cloth3d/Samples/`` and export
eval-compatible NPZs into a fresh per-batch output directory.

Each sample is run in its own subprocess so Taichi field allocations and
sample-specific state cannot leak between runs. Everything from one
invocation lands under::

    {out}/batch_{YYYYMMDD_HHMMSS}/
      results_xpbd/    <- per-garment {sample}_{garment}_sim.npz
      cloth3d_data/    <- per-sample {sample}.npz (optional)
      logs/            <- stdout/stderr for each sample
      summary.txt      <- run status per sample (ok / fail / rc)

This is designed for the teammate's ``cloth3d_benchmark/eval_metrics.py``
workflow: point its ``--results_root`` at ``results_xpbd`` and its
``--cloth3d_npz_path`` at ``cloth3d_data`` and you'll evaluate the whole
batch in one shot.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import time


_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir))
_SAMPLES_DIR = os.path.join(_REPO, "cloth3d", "Samples")


def discover_samples(samples_dir: str) -> list[str]:
    """Return sorted sample ids (subdirectory names) in `samples_dir`."""
    if not os.path.isdir(samples_dir):
        raise SystemExit(f"[batch] samples dir not found: {samples_dir}")
    entries = sorted(
        e for e in os.listdir(samples_dir)
        if os.path.isdir(os.path.join(samples_dir, e))
    )
    return entries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run xPBD on every CLOTH3D sample and export NPZs."
    )
    p.add_argument(
        "--samples_dir", default=_SAMPLES_DIR,
        help="directory containing sample subfolders (default: cloth3d/Samples)",
    )
    p.add_argument(
        "--samples", default=None,
        help="comma-separated subset of sample ids to run; default = all",
    )
    p.add_argument(
        "--out", default=os.path.join(_REPO, "xpbd_out"),
        help="parent dir for the new batch_* folder",
    )
    p.add_argument(
        "--batch_name", default=None,
        help="override the default timestamped batch folder name",
    )
    p.add_argument(
        "--garments", default="all",
        help="passed to each sample run (default: all)",
    )
    p.add_argument("--arch", default="cpu", choices=["cpu", "gpu", "vulkan"])
    p.add_argument("--body_frames", type=int, default=None,
                   help="omit or -1 for full sample length (per-sample auto)")
    p.add_argument("--steps", type=int, default=None,
                   help="omit or -1 to match --body_frames")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--substeps", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--force_fabric", default=None)
    p.add_argument(
        "--garment_y_translation", type=float, default=0.0,
        help="drop-experiment lift in metres along z; passed through to "
             "each sample run. Use 3.0 to mirror partner's drop batch.",
    )
    # Default on: the downstream eval needs `{sample}.npz`, and extracting
    # it here is cheap relative to the sim itself. Pass `--no_save_sample_npz`
    # to skip if you only want simulation results.
    p.add_argument(
        "--save_sample_npz", action=argparse.BooleanOptionalAction, default=True,
        help="also extract the per-sample CLOTH3D NPZ each sample needs "
             "for the eval's source loader (default: on).",
    )
    p.add_argument(
        "--timeout", type=float, default=None,
        help="per-sample subprocess timeout in seconds (default: no limit)",
    )
    p.add_argument(
        "--python", default=sys.executable,
        help="python executable used to spawn each sample run",
    )
    return p


def _build_child_command(args, sample: str, npz_out: str, sample_npz_dir: str) -> list[str]:
    """Build the `python -m xpbd` invocation for one sample."""
    cmd: list[str] = [
        args.python, "-m", "xpbd",
        "--sample", sample,
        "--garments", args.garments,
        "--arch", args.arch,
        "--save_npz",
        "--npz_out", npz_out,
    ]
    if args.save_sample_npz:
        cmd += ["--save_sample_npz", "--sample_npz_dir", sample_npz_dir]
    if args.body_frames is not None:
        cmd += ["--body_frames", str(args.body_frames)]
    if args.steps is not None:
        cmd += ["--steps", str(args.steps)]
    if args.dt is not None:
        cmd += ["--dt", str(args.dt)]
    if args.substeps is not None:
        cmd += ["--substeps", str(args.substeps)]
    if args.iters is not None:
        cmd += ["--iters", str(args.iters)]
    if args.force_fabric:
        cmd += ["--force_fabric", args.force_fabric]
    if args.garment_y_translation:
        cmd += ["--garment_y_translation", str(args.garment_y_translation)]
    return cmd


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    samples = discover_samples(args.samples_dir)
    if args.samples:
        wanted = [s.strip() for s in args.samples.split(",") if s.strip()]
        missing = [s for s in wanted if s not in samples]
        if missing:
            raise SystemExit(f"[batch] unknown samples: {missing}")
        samples = wanted

    if not samples:
        raise SystemExit("[batch] no samples to run")

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = args.batch_name or f"batch_{stamp}"
    batch_dir = os.path.join(args.out, batch_name)
    results_dir = os.path.join(batch_dir, "results_xpbd")
    cloth3d_dir = os.path.join(batch_dir, "cloth3d_data")
    logs_dir = os.path.join(batch_dir, "logs")
    for d in (batch_dir, results_dir, cloth3d_dir, logs_dir):
        os.makedirs(d, exist_ok=True)
    print(f"[batch] writing to {batch_dir}")
    print(f"[batch] {len(samples)} sample(s): {samples}")

    rows: list[tuple[str, str, float, int]] = []  # (sample, status, secs, rc)
    t_start = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        log_path = os.path.join(logs_dir, f"{sample}.log")
        cmd = _build_child_command(args, sample, results_dir, cloth3d_dir)
        print(f"\n[batch] ({i}/{len(samples)}) sample={sample}")
        print(f"        cmd: {' '.join(cmd)}")
        print(f"        log: {log_path}")

        t0 = time.perf_counter()
        try:
            with open(log_path, "w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    cmd,
                    stdout=log_f, stderr=subprocess.STDOUT,
                    cwd=_REPO, timeout=args.timeout,
                )
            rc = proc.returncode
            status = "ok" if rc == 0 else f"fail_rc{rc}"
        except subprocess.TimeoutExpired:
            rc = -1
            status = f"timeout_{args.timeout}s"
        secs = time.perf_counter() - t0
        rows.append((sample, status, secs, rc))
        print(f"        status={status} elapsed={secs:.1f}s")

    # Summary
    total_secs = time.perf_counter() - t_start
    n_ok = sum(1 for _, s, _, _ in rows if s == "ok")
    summary_path = os.path.join(batch_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"batch={batch_name} total_samples={len(rows)} ok={n_ok} "
                f"fail={len(rows) - n_ok} total_elapsed_s={total_secs:.1f}\n")
        f.write("sample,status,elapsed_s,return_code\n")
        for sample, status, secs, rc in rows:
            f.write(f"{sample},{status},{secs:.2f},{rc}\n")

    print(
        f"\n[batch] done. ok={n_ok}/{len(rows)} "
        f"total {total_secs:.1f}s. summary={summary_path}"
    )
    print(f"[batch] results_root: {results_dir}")
    print(f"[batch] cloth3d_npz_path: {cloth3d_dir}")

    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())

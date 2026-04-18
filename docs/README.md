# Documentation

These are the explanation docs for the xPBD-on-CLOTH3D project. The
top-level `README.md` is the cheat-sheet; this folder is the deep dive.

| File | Read this when you want to know |
|---|---|
| [`architecture.md`](architecture.md) | The module map: what lives in each file and how they snap together. |
| [`pipeline.md`](pipeline.md) | The end-to-end flow from a CLOTH3D sample on disk to a rendered mp4. |
| [`xpbd_method.md`](xpbd_method.md) | The XPBD math: predict → constraint solve → finalize, derivations, and how the kernels implement it. |
| [`data_pipeline.md`](data_pipeline.md) | How CLOTH3D samples are loaded and how multi-garment outfits get merged into one cloth system. |
| [`fabric_presets.md`](fabric_presets.md) | What each fabric's compliance / damping / density values mean and how to pick / tune them. |
| [`viewers.md`](viewers.md) | The three viewer back-ends (GGUI, matplotlib, headless) and when to use each. |
| [`cipc_comparison.md`](cipc_comparison.md) | The recipe for a like-for-like comparison run against the C-IPC cotton baseline. |

Suggested reading order if this is your first time:

1. `architecture.md` (5 min): the map.
2. `pipeline.md` (5 min): the flow.
3. `xpbd_method.md` (15 min): the math and where to find it in code.
4. `data_pipeline.md` and `fabric_presets.md` (10 min): the inputs that
   parameterise the simulation.
5. `viewers.md` and `cipc_comparison.md` (5 min): the outputs.

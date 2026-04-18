# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**xPBD** is a Python toolkit for the [CLOTH3D dataset](https://chalearnlap.cvc.uab.es/dataset/38/description/) — a large-scale 3D cloth simulation dataset. It provides tools to read/write animated garment meshes, extract video frames, visualize in Blender, and compress data for submission.

## Environment Setup

```bash
pip install -r cloth3d/requirements.txt
# pymesh2 (for evaluation metrics) is best installed via Conda
```

**Docker:**
```bash
docker build -t cloth3d:cpu cloth3d/
docker build -t cloth3d:gpu -f cloth3d/Dockerfile.gpu cloth3d/
```

## Common Commands

**Extract video frames from a sample:**
```bash
python cloth3d/DataReader/extract_frames.py <sample_name>
```

**Run full data extraction demo:**
```bash
python cloth3d/Demo/extract_sample_data.py --sample 00016 --frame -1 --output_path ./output/
# --frame -1 extracts all frames; otherwise specify a 0-based frame index
```

**Visualize a sample in Blender:**
```bash
blender --python cloth3d/DataReader/view.py -- <sample_name> <frame_number>
```

**Interactive demo:**
```bash
jupyter notebook cloth3d/Demo/demo.ipynb
```

## Architecture

The main package is `cloth3d/DataReader/`. All sample data lives under `cloth3d/Samples/<sample_id>/`.

### Key Modules

| Module | Role |
|--------|------|
| `DataReader/read.py` | **Primary API.** `DataReader` class — call `read_info()`, `read_human()`, `read_garment_vertices()`, `read_garment_topology()`, `read_garment_vertex_colors()`, `read_camera()` |
| `DataReader/IO.py` | File format I/O: `readOBJ`/`writeOBJ`, `readPC2`/`writePC2` (32-bit), PC16 (16-bit float), `readFaceBIN`/`writeFaceBIN` |
| `DataReader/compress.py` | Submission format: `compress(fname, V, F)` → `.pc16` + `.bin`; `decompress(fname)` → `V, F` |
| `DataReader/smpl/smpl_np.py` | NumPy SMPL body model. Forward kinematics over pose/shape/trans → 6890 vertices + 13776 faces |
| `DataReader/util.py` | Z-rotation matrices, camera projection, UV-to-pixel mapping (2048×2048 texture space) |
| `DataReader/util_view.py` | Blender scene utilities (object creation, mesh caching) |
| `DataReader/view.py` | Blender rendering script for a full sample |
| `DataReader/extract_frames.py` | ffmpeg wrapper to dump `.mkv` video sequences to PNG frames |
| `Demo/extract_sample_data.py` | **Comprehensive usage example** — merges human + garments, applies textures, exports NPZ |

### Data Flow

```
Samples/<id>/
  info.mat          → read_info()   → poses (72×T), shape (10,), trans (3×T), outfit, camera
  <garment>.obj     → readOBJ()     → V, F, Vt (UV coords), Ft (UV faces)
  <garment>.pc16    → readPC2Frame()→ vertex positions for one frame (N×3, float16)
  <garment>.png     → PIL           → per-vertex colors via UV mapping
  *.mkv             → extract_frames.py → PNG frames
```

`DataReader.read_human(sample, frame)` runs the SMPL forward pass internally using `smpl/model_f.pkl` or `model_m.pkl` (loaded at `DataReader.__init__`).

### Data Formats

- **PC2 / PC16**: Point-cache animation. Header encodes frame count + vertex count. PC16 uses float16 (precision degrades for `|x| > 2`).
- **BIN**: Binary face topology — uint16 indices, max 65536 vertices per mesh.
- **info.mat**: MATLAB struct loaded with `scipy.io.loadmat`; nested dicts flattened by `IO.load_info()`.

### Coordinate System

- Z-axis up; SMPL root joint at origin for relative coordinates.
- `zrot` in `info.mat` gives the global Z-axis rotation applied to the whole scene.
- Pass `absolute=True` to `read_garment_vertices` / `read_human` to include `trans` offset.

### Garment & Fabric Types

Garments: `Tshirt`, `Top`, `Trousers`, `Skirt`, `Jumpsuit`, `Dress`  
Fabrics: `Cotton`, `Silk`, `Denim`, `Leather`

### Evaluation

Surface-to-surface distance (vertex-to-face) is the primary metric for cloth reconstruction. Requires `pymesh2`.

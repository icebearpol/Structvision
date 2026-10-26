# StructVision

AI-assisted structural health audit framework for UAV-based concrete inspection: YOLO segmentation of cracks, photogrammetric 3D reconstruction, ground-sampling-distance (GSD) measurement, and non-linear finite element analysis with OpenSeesPy.

> **Status:** early-stage build. The computer-vision, GSD, FEA, and reporting core works end-to-end; the REST triage/audit endpoints and some demo clients are still stubs (see [Known limitations](#known-limitations)).

## Overview

StructVision turns drone imagery into an engineering audit trail:

1. **Capture** — images from a UAV (or synthetic samples in `input_images/`).
2. **Detect** — a YOLO11-seg model segments cracks and damage.
3. **Measure** — `GSDEngine` converts mask pixels into centimetres/millimetres using EXIF focal length, sensor width, and flight altitude.
4. **Reconstruct** — `photogrammetry/` dispatches image batches to a NodeODM server for an orthomosaic / 3D mesh.
5. **Map** — `MeshRaycaster` projects 2D masks onto the 3D mesh with PyVista ray tracing, producing a `crack_damage` scalar field.
6. **Analyse** — `OpenSeesAuditor` runs an incremental static pushover and a dynamic time-history seismic analysis, then maps stress and displacement back onto the mesh.
7. **Report** — the pipeline exports an enriched VTK mesh, a full nodal CSV, field plots, and an executive PDF; the Streamlit dashboard visualises them.

### Audit levels

| Level | Scope | Implementation |
| --- | --- | --- |
| 1 | Real-time crack detection HUD | `realtime/rt_stream_server.py` (YOLO11-seg + GSD, WebSocket) |
| 2 | Fast 2D triage | `POST /api/v1/triage` in `main.py` (stub response) |
| 3 | 3D mesh mapping + FEA audit | `cv_pipeline/raycaster.py`, `cv_pipeline/fea_engine.py`, `cv_pipeline/pipeline.py` |

## Repository layout

```
StructVision/
├── app.py                      # Streamlit audit dashboard (3D mesh, stress explorer, benchmark, downloads)
├── main.py                     # FastAPI orchestrator; mounts the real-time router at /rt
├── requirements.txt
├── test_ws_client.py           # Minimal WebSocket HUD client
├── cv_pipeline/
│   ├── pipeline.py             # End-to-end VTK -> FEA -> CSV/PDF/PNG runner
│   ├── fea_engine.py           # OpenSeesPy pushover + seismic time-history analysis
│   ├── gsd_engine.py           # EXIF parsing, GSD, pixel -> physical crack metrics
│   ├── raycaster.py            # 2D mask -> 3D mesh projection
│   ├── pdf_reporter.py         # Executive PDF generation
│   ├── batch_processor.py      # Batch images -> damage index D -> degraded modulus
│   ├── damaged_wall.vtk        # Sample damage mesh input
│   └── mock_wall.obj           # Sample mesh for ray casting
├── realtime/
│   ├── rt_stream_server.py     # YOLO inference loop + /ws/hud WebSocket broadcast
│   └── test_client.html        # Browser HUD demo page
├── photogrammetry/
│   └── odm_client.py           # NodeODM (PyODM) task manager
├── input_images/               # Sample PGM crack images (minor/moderate/severe)
└── output/                     # Generated artefacts (gitignored)
```

## Requirements

- Python 3.10+
- A working OpenSeesPy installation (Windows DLL paths are handled in `fea_engine.py`)
- Optional: Docker + NodeODM for photogrammetry
- Optional: a trained crack-segmentation checkpoint

`requirements.txt` covers the core dependencies. The dashboard and photogrammetry client need a few extra packages that are not listed there yet:

```bash
pip install streamlit pandas plotly pyodm
```

YOLO weights are intentionally **not** committed (`*.pt` is gitignored). `realtime/rt_stream_server.py` loads:

- `cv_pipeline/models/structvision_crack_seg/weights/best.pt` if present, otherwise
- the stock `yolo11n-seg.pt`, which Ultralytics downloads automatically on first use.

## Installation

```bash
git clone https://github.com/icebearpol/Structvision.git
cd Structvision

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
pip install streamlit pandas plotly pyodm
```

Optional photogrammetry backend:

```bash
docker run -p 3000:3000 opendronemap/nodeodm
```

## Usage

### 1. Run the FEA audit pipeline

Generates every artefact in `output/` from `cv_pipeline/damaged_wall.vtk`:

```bash
python -m cv_pipeline.pipeline
```

The pipeline runs a 100 kN pushover and a 10 s seismic time history, then writes the nodal CSV, enriched VTK, plots, and the executive PDF.

### 2. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard reads `output/full_nodal_stress_report.csv`; run the pipeline first (or use the sidebar's **Run Full Audit Pipeline** button). It offers a 3D stress mesh, a critical-node explorer, the analytical-vs-FEA benchmark view, and PDF/CSV downloads.

> On Windows, the sidebar button calls `python3`, which is usually unavailable. Use `python` in the terminal or adjust `app.py:59`.

### 3. Start the API + live HUD

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

| Route | Description |
| --- | --- |
| `GET /` | Health check |
| `POST /api/v1/triage` | Level 2 2D triage (currently a static stub) |
| `POST /api/v1/audit` | Level 3 audit queue (currently a static stub) |
| `WS /rt/ws/hud` | Live detection/telemetry stream |

The real-time worker connects to a camera at `http://192.168.1.6:8080/video`; edit `main.py:48` for your feed. A minimal consumer is provided:

```bash
python test_ws_client.py
```

Note: `test_ws_client.py` and `realtime/test_client.html` still point at older paths (`/ws/hud`, `/ws/live-flight`). The mounted endpoint is `/rt/ws/hud`.

### 4. Batch-process crack images

```bash
python -m cv_pipeline.batch_processor
```

Processes everything in `input_images/` (generating synthetic PGM samples if empty), computes a damage index `D` and degraded modulus `E_damaged = (1 - D) * E0`, and writes `output/batch_processing_summary.csv`.

### 5. Photogrammetry (optional)

`photogrammetry/odm_client.py` wraps PyODM. Point `ODMTaskManager` at your NodeODM host/port, then submit a batch of images; the client polls task progress and can download the orthophoto.

## Generated outputs

All artefacts are written to `output/` and are excluded from version control:

| File | Contents |
| --- | --- |
| `Executive_Structural_Audit_Report.pdf` | Executive summary, stress contours, top critical nodes |
| `full_nodal_stress_report.csv` | Stress, strain, displacement, crack flag per node |
| `fea_audit_result.vtk` | Enriched mesh with `fea_stress_mpa` / `fea_displacement_mm` |
| `structural_report.png` | Damage map and stress field comparison |
| `pushover_capacity_curve.png` | Base shear vs. top displacement |
| `seismic_response_history.png` | Time-history response |
| `seismic_record.acc` | Synthetic ground-motion record |
| `batch_processing_summary.csv` | Per-image damage metrics |

## Benchmarking

The dashboard's **Thesis Benchmark Validation** tab compares the numerical model against closed-form Euler-Bernoulli cantilever formulae (`σ = My/I`, `δ = PL³/3EI`) for a 100 kN lateral load. The reported variance is attributed to 2D continuum effects (Poisson restraint, fixed base) and local stiffness degradation at detected cracks.

## Known limitations

- `POST /api/v1/triage` and `POST /api/v1/audit` return hard-coded responses; they are not wired to the CV or FEA engines yet.
- The RTSP/HTTP camera URL in `main.py` is hard-coded.
- Demo WebSocket clients reference stale endpoints (`/ws/hud`, `/ws/live-flight`); the actual mounted route is `/rt/ws/hud`.
- The dashboard's **Run Full Audit Pipeline** button assumes `python3` (fails on stock Windows).
- `requirements.txt` omits `streamlit`, `pandas`, `plotly`, and `pyodm`.
- Stress enrichment in `cv_pipeline/pipeline.py` uses a simplified spatial mapping rather than a full field solve for every node.

## License

No license has been specified yet. Contact the author before reuse.

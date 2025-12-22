# TBEP Invasives (Quarterly Pipeline + Dashboard)

This repository contains:
1) A quarterly pipeline that downloads and derives invasive species datasets and supporting spatial layers.
2) A Dash dashboard that visualizes the **latest** derived outputs.

## Key concept: `derived_data` vs `outputs`

- **derived_data/** = the **latest snapshot** (stable paths).  
  The dashboard always reads from here.
- **outputs/<run_id>/** = archived artifacts per run (traceability / rollback).

A typical quarterly run:
- writes to `outputs/<run_id>/...` (archive)
- then updates `derived_data/...` (latest snapshot)

---

## Repository layout (high level)

- `config/settings.yaml` — all paths + runtime parameters
- `input_data/` — static inputs (AOI, boundaries, base hexbins, FLAM raster, etc.)
- `derived_data/` — latest snapshot created by the pipeline (app reads here)
- `outputs/` — archived quarterly run outputs
- `src/tbep_invasives/`
  - `steps/` — individual pipeline steps (each has a `run(cfg)` entrypoint)
  - `pipeline/` — orchestrators (quarterly runner + preflight checks)
  - `app/` — dashboard app (Dash factory + runner + WSGI entrypoint)

---

## Execution flows (what calls what)

### Quarterly pipeline flow

`src/tbep_invasives/pipeline/quarterly.py`

- `main()`:
  - loads config via `tbep_invasives.paths.load_config()`
  - calls `run(cfg)`

- `run(cfg)`:
  - (optional) calls `pipeline.preflight.preflight(cfg, mode="quarterly")`
  - calls each step’s `run(cfg)` in sequence:
    1. `steps.download_invasives.run(cfg)`
    2. `steps.flam_overlay.run(cfg)`
    3. `steps.hex_enrichment.run(cfg)`
    4. `steps.report_cards.run(cfg)`

Each step reads from `input_data/` and/or previously generated `derived_data/` files,
writes archived outputs into `outputs/<run_id>/...`, then updates `derived_data/...`.

### Dashboard app flow (development)

`src/tbep_invasives/app/run_app.py`

- `main()`:
  - loads config via `load_config()`
  - calls `dashboard.create_app(cfg)`
  - starts a dev server (Dash) using `app.run(...)`

### Dashboard app flow (production / Docker)

`src/tbep_invasives/app/wsgi.py`

- module-level:
  - loads config via `load_config()`
  - calls `dashboard.create_app(cfg)`
  - exposes `server = dash_app.server` for Gunicorn

Docker runs Gunicorn, which serves the Dash/Flask server via WSGI.

---

## What each step does (and who calls it)

All steps follow the same interface:
- `run(cfg) -> dict` does the work and returns output paths.
- `main()` loads config and calls `run(cfg)` so the file can be run directly.

### 1) `steps/download_invasives.py`
**Purpose:** Download/filter invasive observations, standardize fields, assign `hexbinID` by spatial join, and export a dataset the app can consume.

**Called by:** `pipeline/quarterly.py` (Step 1)

**Primary inputs (config):**
- `paths.hexbins_shp` (base hexbins with `id`)
- `paths.invasives_concern_csv` (top concern species list)
- other parameters under `parameters:` (HUCs, min_year, etc.)

**Outputs:**
- Latest snapshot:
  - `derived.invasives_csv`
  - `derived.invasives_geojson`
- Archive:
  - `outputs/<run_id>/invasives/...`

### 2) `steps/flam_overlay.py`
**Purpose:** Clip FLAM raster to AOI and create a transparent PNG overlay + bounds metadata for map display.

**Called by:** `pipeline/quarterly.py` (Step 2)

**Inputs:**
- `paths.flam_raster`
- `paths.aoi_shp`
- `flam_overlay.colormap`, `flam_overlay.alpha`

**Outputs:**
- Latest snapshot:
  - `derived.flam_overlay_png`
  - `derived.flam_meta_json`
- Archive:
  - `outputs/<run_id>/rasters/...`

> Note: This step can be rerun at any time; it doesn’t depend on invasives downloads.

### 3) `steps/hex_enrichment.py`
**Purpose:** Enrich hexbins with spatial attributes and FLAM stats (dominant muni/segment, zonal mean, FLAM rank). This is used by the dashboard to compute priority hexbins.

**Called by:** `pipeline/quarterly.py` (Step 3)

**Inputs:**
- `paths.hexbins_shp`
- `paths.municipalities_shp`
- `paths.bay_segments_shp`
- `paths.flam_raster`

**Outputs:**
- Latest snapshot:
  - `derived.hexbins_v2_shp` (shapefile set)
- Archive:
  - `outputs/<run_id>/shp/...`

### 4) `steps/report_cards.py`
**Purpose:** Generate summary “report card” PNGs (abundance/richness) from the latest invasives dataset.

**Called by:** `pipeline/quarterly.py` (Step 4)

**Inputs:**
- `derived.invasives_csv`
- `paths.aoi_shp`
- `paths.bay_segments_shp`
- `report_cards.*` parameters

**Outputs:**
- Archive:
  - `outputs/<run_id>/plots/report_cards/...png`
- Optional latest snapshot:
  - `derived.report_cards_dir/...png` (if enabled in config)

---

## Dashboard modules

### `app/dashboard.py`
**Purpose:** The Dash “app factory.”

- `create_app(cfg)` loads:
  - derived invasives datasets (`derived.*`)
  - derived FLAM overlay (`derived.*`)
  - derived enriched hexbins (`derived.hexbins_v2_shp`)
  - boundary layers (`paths.*`)
- and returns a Dash `app` object.

### `app/run_app.py`
**Purpose:** Development runner.
- Calls `create_app(cfg)` and runs a dev server.

### `app/wsgi.py`
**Purpose:** Production runner entrypoint (Gunicorn).
- Exposes `server = dash_app.server`.

---

## Configuration: `config/settings.yaml`

### Project metadata
- `project.name` — project identifier (informational)

### Requests / networking
- `requests.verify_ssl` — set `false` only if necessary behind corporate SSL interception
- `requests.ca_bundle` — optional path to a corporate CA bundle PEM

### Paths
**Input directories**
- `paths.input_data_dir`
- `paths.shp_dir`, `paths.rasters_dir`, `paths.tables_dir`

**Input files**
- `paths.aoi_shp`
- `paths.municipalities_shp`
- `paths.bay_segments_shp`
- `paths.hexbins_shp`
- `paths.flam_raster`
- `paths.invasives_concern_csv`

**Outputs**
- `paths.derived_data_dir`
- `paths.outputs_dir`

### Derived outputs (latest snapshot)
- `derived.invasives_csv`
- `derived.invasives_geojson`
- `derived.flam_overlay_png`
- `derived.flam_meta_json`
- `derived.hexbins_v2_shp`
- `derived.report_cards_dir` (optional convenience)

### Run controls
- `run.run_id` — set e.g. `"2025Q1"`; if null a date-based id is used
- `run.archive_outputs` — write to `outputs/<run_id>/...`
- `run.update_derived` — update `derived_data/...` latest snapshot

### Step parameters
- `parameters.*` — download filters (min_year, HUCs, etc.)
- `flam_overlay.*` — colormap, alpha
- `hex_enrichment.*` — logging frequency, etc.
- `report_cards.*` — plotting options
- `app.*` — dev server host/port/debug (used by `run_app.py`)

---

## How to run (PowerShell)

### Install
```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Verification (GitHub Actions)

This repo includes manual GitHub Actions workflows used to validate the system on a clean Linux runner.

### Quarterly Pipeline E2E
Runs the full quarterly pipeline end-to-end and uploads artifacts.
- Produces: `derived_data/` (latest snapshot) and `outputs/` (archived run outputs)
- Artifacts are downloadable from the workflow run summary.

How to run:
1) GitHub → Actions → **Quarterly Pipeline E2E** → Run workflow
2) Optionally provide `run_id` (e.g., `2025Q1`)

### Docker App E2E (Pipeline + Gunicorn)
Validates Docker deployment by:
1) Building the Docker image
2) Running the quarterly pipeline inside the container
3) Starting Gunicorn and verifying Dash endpoints respond

The workflow checks:
- `/`
- `/_dash-layout`
- `/_dash-dependencies`

A successful run indicates the containerized app is deployable and healthy.

Artifacts:
- `derived_data` (latest snapshot)
- `outputs` (including CI logs when enabled)


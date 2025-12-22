"""
Build a transparent PNG overlay and metadata JSON from a FLAM raster clipped to the AOI.

Outputs (latest snapshot for the app):
- derived_data/rasters/FLAM_overlay.png
- derived_data/rasters/flam_meta.json

Optional archive outputs:
- outputs/<run_id>/rasters/FLAM_overlay.png
- outputs/<run_id>/rasters/flam_meta.json

The Dash app expects flam_meta.json to contain:
  bounds_lonlat: [min_lon, min_lat, max_lon, max_lat]
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.mask import mask
from rasterio.transform import array_bounds
from matplotlib import colormaps

logger = logging.getLogger(__name__)
CRS_WGS84 = "EPSG:4326"


# --- import config helpers with a fallback for running this file directly ---
try:
    from tbep_invasives.paths import load_config, resolve_path
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))  # adds ./src
    from tbep_invasives.paths import load_config, resolve_path


def get_run_id(cfg: Dict[str, Any]) -> str:
    run_id = (cfg.get("run", {}) or {}).get("run_id")
    if run_id:
        return str(run_id)
    return datetime.now().strftime("%Y%m%d")


def _normalize(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Normalize finite values to 0-1 and return (norm, vmin, vmax)."""
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros_like(data, dtype=float), float("nan"), float("nan")

    vmin = float(np.nanmin(data[finite]))
    vmax = float(np.nanmax(data[finite]))
    if np.isclose(vmin, vmax):
        norm = np.zeros_like(data, dtype=float)
    else:
        norm = (data - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)

    # Replace NaNs with 0 for colormap lookup; alpha will handle transparency
    norm[~finite] = 0.0
    return norm, vmin, vmax


def build_overlay_rgba(
    data: np.ndarray,
    *,
    colormap_name: str = "viridis",
    alpha: int = 160,
) -> Tuple[np.ndarray, float, float]:
    """
    Convert raster values to RGBA uint8 with transparent NoData.

    alpha: 0-255 constant alpha for valid pixels.
    """
    norm, vmin, vmax = _normalize(data)

    cmap = colormaps.get(colormap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)  # (H, W, 4)

    valid = np.isfinite(data)
    rgba[..., 3] = 0
    rgba[valid, 3] = np.uint8(alpha)

    return rgba, vmin, vmax


def compute_bounds_lonlat(src_crs: Any, transform: Any, height: int, width: int) -> Tuple[float, float, float, float]:
    """Return bounds as (min_lon, min_lat, max_lon, max_lat) for the clipped raster."""
    left, bottom, right, top = array_bounds(height, width, transform)
    to_ll = Transformer.from_crs(src_crs, CRS_WGS84, always_xy=True)
    min_lon, min_lat = to_ll.transform(left, bottom)
    max_lon, max_lat = to_ll.transform(right, top)
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def write_png_and_meta(png_path: Path, meta_path: Path, rgba: np.ndarray, meta: Dict[str, Any]) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Write PNG without pyplot state
    import matplotlib.image as mpimg
    mpimg.imsave(png_path, rgba)

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def copy_to_latest(src_png: Path, src_meta: Path, dst_png: Path, dst_meta: Path) -> None:
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    dst_meta.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_png, dst_png)
    shutil.copy2(src_meta, dst_meta)


def run(cfg: Dict[str, Any]) -> Dict[str, Path]:
    flam_tif = resolve_path(cfg, "paths.flam_raster")
    aoi_shp = resolve_path(cfg, "paths.aoi_shp")

    outputs_dir = resolve_path(cfg, "paths.outputs_dir")
    run_id = get_run_id(cfg)

    archive_enabled = bool((cfg.get("run", {}) or {}).get("archive_outputs", True))
    update_derived = bool((cfg.get("run", {}) or {}).get("update_derived", True))

    # Latest snapshot paths (what the app will read)
    latest_png = resolve_path(cfg, "derived.flam_overlay_png")
    latest_meta = resolve_path(cfg, "derived.flam_meta_json")

    # Rendering options
    render_cfg = (cfg.get("flam_overlay", {}) or {})
    colormap_name = str(render_cfg.get("colormap", "viridis"))
    alpha = int(render_cfg.get("alpha", 160))

    if not flam_tif.exists():
        raise FileNotFoundError(f"FLAM raster not found: {flam_tif}")
    if not aoi_shp.exists():
        raise FileNotFoundError(f"AOI shapefile not found: {aoi_shp}")

    logger.info("Reading FLAM raster: %s", flam_tif)
    logger.info("Clipping to AOI: %s", aoi_shp)

    with rasterio.open(flam_tif) as src:
        aoi = gpd.read_file(aoi_shp).to_crs(src.crs)

        # filled=False returns a masked array so nodata stays masked
        out, out_transform = mask(src, aoi.geometry, crop=True, filled=False)

        # Use first band
        band = out[0]
        data = band.astype("float32").filled(np.nan)

        rgba, vmin, vmax = build_overlay_rgba(data, colormap_name=colormap_name, alpha=alpha)
        bounds_lonlat = compute_bounds_lonlat(src.crs, out_transform, rgba.shape[0], rgba.shape[1])

    meta = {
        "bounds_lonlat": list(bounds_lonlat),  # [min_lon, min_lat, max_lon, max_lat]
        "vmin": vmin,
        "vmax": vmax,
        "colormap": colormap_name,
        "alpha": alpha,
    }

    if archive_enabled:
        archive_png = outputs_dir / run_id / "rasters" / latest_png.name
        archive_meta = outputs_dir / run_id / "rasters" / latest_meta.name

        write_png_and_meta(archive_png, archive_meta, rgba, meta)
        logger.info("Archived overlay: %s", archive_png)
        logger.info("Archived meta:    %s", archive_meta)

        if update_derived:
            copy_to_latest(archive_png, archive_meta, latest_png, latest_meta)
            logger.info("Updated latest snapshot in derived_data.")
    else:
        write_png_and_meta(latest_png, latest_meta, rgba, meta)
        logger.info("Wrote latest snapshot only (no archive).")

    return {"png": latest_png, "meta": latest_meta}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()

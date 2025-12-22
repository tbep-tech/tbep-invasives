"""
Enrich hexbins with:
- dominant municipality by overlap area
- dominant bay segment by overlap area
- mean FLAM raster value per hex (zonal mean)
- FLAM rank (1 = highest mean; dense ranking; NaN -> 0)

Outputs:
- Archive: outputs/<run_id>/shp/Hexbins_2mi_4326_v2.*
- Latest snapshot for app: derived_data/shp/Hexbins_2mi_4326_v2.*
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

logger = logging.getLogger(__name__)

OUTPUT_CRS = "EPSG:4326"   # final output
AREA_CRS = "EPSG:2237"     # accurate area overlap (StatePlane FL West ftUS)


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


def dominant_category_by_area(
    hex_gdf: gpd.GeoDataFrame,
    src_gdf: gpd.GeoDataFrame,
    src_field: str,
    area_crs: str,
) -> pd.Series:
    """Return Series indexed by hex 'id' with dominant category by area overlap."""
    H = hex_gdf[["id", "geometry"]].to_crs(area_crs)
    S = src_gdf[[src_field, "geometry"]].to_crs(area_crs)

    inter = gpd.overlay(H, S, how="intersection", keep_geom_type=False)
    if inter.empty:
        return pd.Series(index=hex_gdf["id"], dtype="object")

    inter["a"] = inter.geometry.area
    g = inter.groupby(["id", src_field])["a"].sum().reset_index()
    idx = g.groupby("id")["a"].idxmax()
    dom = g.loc[idx, ["id", src_field]].set_index("id")[src_field]
    return dom.reindex(hex_gdf["id"])


def zonal_mean_raster(
    hex_gdf: gpd.GeoDataFrame,
    raster_path: Path,
    *,
    progress_every: int = 200,
) -> pd.Series:
    """Mean raster value per hex; ignores NODATA. Returns Series indexed by 'id'."""
    with rasterio.open(raster_path) as src:
        hb = hex_gdf[["id", "geometry"]].to_crs(src.crs)
        nodata = src.nodata

        vals: list[float] = []
        n = len(hb)

        for i, geom in enumerate(hb.geometry, start=1):
            if progress_every and (i == 1 or i % progress_every == 0 or i == n):
                logger.info("Zonal mean progress: %s / %s", i, n)

            try:
                out, _ = mask(src, [mapping(geom)], crop=True)
                arr = out[0].astype("float32")

                if nodata is not None:
                    arr = np.where(arr == nodata, np.nan, arr)

                vals.append(float(np.nanmean(arr)))
            except Exception:
                vals.append(np.nan)

        return pd.Series(vals, index=hex_gdf["id"], name="flam_mean")


def _remove_shapefile_set(shp_path: Path) -> None:
    """
    Remove existing shapefile component files with the same stem:
    .shp, .shx, .dbf, .prj, .cpg, etc.
    """
    stem = shp_path.stem
    for p in shp_path.parent.glob(stem + ".*"):
        try:
            p.unlink()
        except Exception:
            # If Windows has the file locked, you'll see it here.
            logger.warning("Could not delete existing file: %s", p)


def write_shapefile(gdf: gpd.GeoDataFrame, out_shp: Path) -> None:
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    _remove_shapefile_set(out_shp)
    gdf.to_file(out_shp)
    logger.info("Wrote shapefile: %s", out_shp)


def copy_shapefile_set(src_shp: Path, dst_shp: Path) -> None:
    """
    Copy all shapefile sidecar files from src stem to dst stem.
    Example: copy Hexbins_...v2.shp, .dbf, .shx, .prj, .cpg, etc.
    """
    dst_shp.parent.mkdir(parents=True, exist_ok=True)

    src_stem = src_shp.stem
    dst_stem = dst_shp.stem

    # Clear existing destination set first (avoid stale sidecars)
    _remove_shapefile_set(dst_shp)

    for p in src_shp.parent.glob(src_stem + ".*"):
        dst = dst_shp.parent / (dst_stem + p.suffix)
        shutil.copy2(p, dst)

    logger.info("Updated latest snapshot shapefile set in: %s", dst_shp.parent)


def run(cfg: Dict[str, Any]) -> Dict[str, Path]:
    hexbins_shp = resolve_path(cfg, "paths.hexbins_shp")
    muni_shp = resolve_path(cfg, "paths.municipalities_shp")
    bseg_shp = resolve_path(cfg, "paths.bay_segments_shp")
    flam_tif = resolve_path(cfg, "paths.flam_raster")

    outputs_dir = resolve_path(cfg, "paths.outputs_dir")
    run_id = get_run_id(cfg)

    archive_enabled = bool((cfg.get("run", {}) or {}).get("archive_outputs", True))
    update_derived = bool((cfg.get("run", {}) or {}).get("update_derived", True))

    latest_out_shp = resolve_path(cfg, "derived.hexbins_v2_shp")

    progress_every = int((cfg.get("hex_enrichment", {}) or {}).get("progress_every", 200))

    if not hexbins_shp.exists():
        raise FileNotFoundError(f"Hexbins shapefile not found: {hexbins_shp}")
    if not muni_shp.exists():
        raise FileNotFoundError(f"Municipalities shapefile not found: {muni_shp}")
    if not bseg_shp.exists():
        raise FileNotFoundError(f"Bay segments shapefile not found: {bseg_shp}")
    if not flam_tif.exists():
        raise FileNotFoundError(f"FLAM raster not found: {flam_tif}")

    logger.info("Reading hexbins: %s", hexbins_shp)
    hexb = gpd.read_file(hexbins_shp).copy()

    # Ensure 'id' exists and is integer
    if "id" not in hexb.columns:
        hexb["id"] = hexb.index.astype(int)
    else:
        hexb["id"] = pd.to_numeric(hexb["id"], errors="coerce").fillna(-1).astype(int)

    muni = gpd.read_file(muni_shp)
    bseg = gpd.read_file(bseg_shp)

    if "Municipali" not in muni.columns:
        raise ValueError("Expected column 'Municipali' in municipalities layer.")
    if "BAY_SEG_GP" not in bseg.columns:
        raise ValueError("Expected column 'BAY_SEG_GP' in bay segments layer.")

    logger.info("Computing dominant municipality by overlap area...")
    dom_muni = dominant_category_by_area(hexb, muni, "Municipali", AREA_CRS).rename("dom_muni")

    logger.info("Computing dominant bay segment by overlap area...")
    dom_bseg = dominant_category_by_area(hexb, bseg, "BAY_SEG_GP", AREA_CRS).rename("dom_bayseg")

    logger.info("Computing zonal mean FLAM per hex...")
    flam_mean = zonal_mean_raster(hexb, flam_tif, progress_every=progress_every)

    # Rank: 1 = highest avg FLAM (dense ranks). NaN -> 0 for shapefile int field.
    flam_rank = flam_mean.rank(ascending=False, method="dense")
    flam_rank = flam_rank.round(0).fillna(0).astype(int).rename("flam_rank")

    out = (
        hexb.merge(dom_muni, left_on="id", right_index=True, how="left")
            .merge(dom_bseg, left_on="id", right_index=True, how="left")
            .merge(flam_mean, left_on="id", right_index=True, how="left")
            .merge(flam_rank, left_on="id", right_index=True, how="left")
    ).to_crs(OUTPUT_CRS)

    out["id"] = out["id"].astype(int)

    # Ensure shapefile field-name limit (<=10 chars) remains satisfied
    # dom_muni, dom_bayseg, flam_mean, flam_rank are all <= 10 chars

    if archive_enabled:
        archive_shp = outputs_dir / run_id / "shp" / latest_out_shp.name
        write_shapefile(out, archive_shp)

        if update_derived:
            copy_shapefile_set(archive_shp, latest_out_shp)
    else:
        write_shapefile(out, latest_out_shp)

    return {"hexbins_v2": latest_out_shp}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()

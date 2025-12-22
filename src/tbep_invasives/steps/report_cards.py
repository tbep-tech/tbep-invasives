"""
Generate "report card" tables for invasive species metrics.

Inputs (from config):
- derived.invasives_csv
- paths.aoi_shp
- paths.bay_segments_shp

Outputs:
- Archive: outputs/<run_id>/plots/report_cards/*.png
- Latest snapshot (optional): derived_data/report_cards/*.png
"""

from __future__ import annotations

import os
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from matplotlib import font_manager as fm
from textwrap import fill
from scipy.stats import norm

logger = logging.getLogger(__name__)

# --- import config helpers with a fallback for running this file directly ---
try:
    from tbep_invasives.paths import load_config, resolve_path
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))  # adds ./src
    from tbep_invasives.paths import load_config, resolve_path


# =============================================================================
# Fonts & Styles (same intent as V4_3)
# =============================================================================
HEADER_FONT = ["Rubik", "DejaVu Sans", "Arial"]
DATA_FONT   = ["Roboto", "DejaVu Sans", "Arial"]
PRIMARY_BLUE = "#005293"

# Optional: if you have local font files, point here and we'll register them at runtime
LOCAL_FONTS: Dict[str, str] = {
    # "Rubik-Regular": r"fonts/Rubik-Regular.ttf",
    # "Rubik-Bold":    r"fonts/Rubik-Bold.ttf",
    # "Roboto-Regular":r"fonts/Roboto-Regular.ttf",
    # "Roboto-Bold":   r"fonts/Roboto-Bold.ttf",
}

def register_local_fonts(font_map):
    for _, path in (font_map or {}).items():
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass

def set_matplotlib_style():
    rcParams["font.family"] = DATA_FONT
    rcParams["axes.titlesize"] = 16
    rcParams["axes.titleweight"] = "bold"
    rcParams["axes.labelsize"] = 12
    rcParams["text.color"] = "#333333"
    rcParams["axes.titlecolor"] = PRIMARY_BLUE
    rcParams["figure.dpi"] = 220

def cmap_white_teal_blue():
    return LinearSegmentedColormap.from_list(
        "white_teal_blue",
        ["#ffffff", "#84cabd", "#4a92b6"],
        N=256,
    )

def cmap_hulk():
    return LinearSegmentedColormap.from_list("hulk", ["#128a31", "#ffffff", "#6d3ca0"], N=256)

def get_cmap(choice: str):
    return cmap_hulk() if str(choice).lower() == "hulk" else cmap_white_teal_blue()

def wrap_labels(labels, width=16):
    return [fill(str(lbl), width=width) if isinstance(lbl, str) else lbl for lbl in labels]


# =============================================================================
# Core data calculations (same as V4_3)
# =============================================================================
def fix_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    return df.dropna(subset=["year"])

def acres_to_units(acres: float, units: str) -> float:
    if units == "sqft":
        return acres * 43560.0
    if units == "sqmi":
        return acres / 640.0
    return acres  # acres

def calc_aoi_area(aoi_path: Path, units="acre") -> float:
    aoi_gdf = gpd.read_file(aoi_path)
    if aoi_gdf.crs is None or aoi_gdf.crs.to_epsg() != 4326:
        aoi_gdf = aoi_gdf.set_crs(epsg=4326)
    aoi_eq = aoi_gdf.to_crs(epsg=3086)
    total_area_m2 = aoi_eq.geometry.area.sum()
    acres = total_area_m2 / 4046.8564224
    return acres_to_units(acres, units)

def calc_watershed_area(watershed_path: Path, segment_field: str, units="acre") -> Dict[str, float]:
    wb = gpd.read_file(watershed_path)
    if wb.crs is None or wb.crs.to_epsg() != 4326:
        wb = wb.set_crs(epsg=4326)
    wb_eq = wb.to_crs(epsg=3086)
    wb_eq["area_m2"] = wb_eq.geometry.area
    area_m2 = wb_eq.groupby(segment_field)["area_m2"].sum()
    acres = (area_m2 / 4046.8564224)
    in_units = acres.apply(lambda x: acres_to_units(float(x), units))
    return in_units.to_dict()

def per_watershed_abundance(df: pd.DataFrame, area_dict: Dict[str, float]) -> pd.DataFrame:
    yearly_counts = df.groupby(["year", "baySegment"]).size().unstack(fill_value=0)
    return yearly_counts.div(pd.Series(area_dict), axis=1)

def per_watershed_richness(df: pd.DataFrame, area_dict: Dict[str, float]) -> pd.DataFrame:
    genus_counts = df.groupby(["year", "baySegment"])["scientificName"].nunique().unstack(fill_value=0)
    return genus_counts.div(pd.Series(area_dict), axis=1)

def total_aoi_abundance(df: pd.DataFrame, total_area_units: float) -> pd.Series:
    counts = df.groupby("year").size()
    return counts / float(total_area_units)

def total_aoi_richness(df: pd.DataFrame, total_area_units: float) -> pd.Series:
    uniq = df.groupby("year")["scientificName"].nunique()
    return uniq / float(total_area_units)

def as_percentile_by_mean_std(values_df: pd.DataFrame) -> pd.DataFrame:
    arr = values_df.values.astype(float)
    mu = np.nanmean(arr)
    sigma = np.nanstd(arr)
    if sigma <= 0 or np.isnan(sigma):
        z = np.zeros_like(arr)
    else:
        z = (arr - mu) / sigma
    p = norm.cdf(z)
    out = pd.DataFrame(p, index=values_df.index, columns=values_df.columns)
    return out * 100.0


# =============================================================================
# Plotting
# =============================================================================
def plot_report_card(
    values_df: pd.DataFrame,
    plot_type: str,
    out_path: Path,
    *,
    color_ramp: str,
    mode: str = "raw",
    units_label: str = "per sq mi per year",
    min_display_threshold: float = 1e-4,
    show_plots: bool = False,
):
    register_local_fonts(LOCAL_FONTS)
    set_matplotlib_style()
    cmap = get_cmap(color_ramp)

    plot_values = values_df.copy()

    if mode == "percentile":
        plot_values = as_percentile_by_mean_std(values_df)
        vmin, vmax = 0, 100
        cbar_label = "Percentile (normalized by μ and σ)"
        label_fmt = lambda x: "" if pd.isna(x) else f"{x:0.0f}"
    else:
        vmin = float(np.nanmin(plot_values.values))
        vmax = float(np.nanmax(plot_values.values))
        cbar_label = f"Value ({units_label})"

        def _fmt(x):
            if pd.isna(x):
                return ""
            if abs(x) < min_display_threshold:
                return f"<{min_display_threshold:.4f}"
            if abs(x) < 1e-5: return f"{x:.7f}"
            if abs(x) < 1e-4: return f"{x:.6f}"
            if abs(x) < 1e-3: return f"{x:.5f}"
            if abs(x) < 1e-2: return f"{x:.4f}"
            return f"{x:.3f}"

        label_fmt = _fmt

    norm_ = plt.Normalize(vmin=vmin, vmax=vmax)
    cell_colours = plot_values.applymap(lambda x: cmap(norm_(x)))
    cell_text = plot_values.applymap(label_fmt).values

    fig, ax = plt.subplots(figsize=(8.5, 9.6))
    ax.axis("off")

    wrapped_cols = wrap_labels(plot_values.columns, width=16)
    table = ax.table(
        cellText=cell_text,
        cellColours=cell_colours.values,
        rowLabels=plot_values.index,
        colLabels=wrapped_cols,
        loc="center",
        cellLoc="center",
        rowLoc="center",
        bbox=[0.075, 0.075, 0.875, 0.925],
    )

    ncols = len(plot_values.columns)
    for (r, c), cell in table.get_celld().items():
        if c < ncols and r >= 0:
            cell.set_width(0.90 / ncols)
        if r == 0 or c == -1:
            cell.set_text_props(fontfamily=HEADER_FONT, color="#005293", fontsize=10, fontweight="bold")
        else:
            cell.set_text_props(fontfamily=DATA_FONT, color="#000000", fontsize=9.5, fontweight="bold")

    title = f"Waterbody Invasive Species {plot_type} Report Card"
    fig.suptitle(title, fontsize=18, fontfamily=HEADER_FONT, color="#005293", y=0.925)
    subtitle = {
        "Abundance": f"Based on invasive species observations {units_label}",
        "Richness": f"Based on species diversity {units_label}",
    }.get(plot_type, f"Based on metric {units_label}")
    ax.set_title(subtitle, fontsize=11, fontfamily=DATA_FONT, color="#4a4a4a", pad=15)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width="60%", height="2.4%", loc="lower center", borderpad=1.4)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_, cmap=cmap), cax=cax, orientation="horizontal")
    cb.ax.set_xlabel(cbar_label, fontsize=9, fontfamily=DATA_FONT, color="#444444")

    fig.subplots_adjust(top=0.85, bottom=0.09)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.08)

    if show_plots:
        plt.show()

    plt.close(fig)


# =============================================================================
# Output helpers (archive + latest snapshot)
# =============================================================================
def get_run_id(cfg: Dict[str, Any]) -> str:
    run_id = (cfg.get("run", {}) or {}).get("run_id")
    if run_id:
        return str(run_id)
    return datetime.now().strftime("%Y%m%d")

def copy_to_latest(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# =============================================================================
# Main runner
# =============================================================================
def run(cfg: Dict[str, Any]) -> Dict[str, Path]:
    invasives_csv = resolve_path(cfg, "derived.invasives_csv")
    aoi_path = resolve_path(cfg, "paths.aoi_shp")
    bay_segments_path = resolve_path(cfg, "paths.bay_segments_shp")

    outputs_dir = resolve_path(cfg, "paths.outputs_dir")
    run_id = get_run_id(cfg)

    archive_enabled = bool((cfg.get("run", {}) or {}).get("archive_outputs", True))
    update_derived = bool((cfg.get("run", {}) or {}).get("update_derived", False))

    # Optional “latest snapshot” dir for report cards (not used by app by default)
    latest_dir = None
    if update_derived and "derived" in cfg and (cfg["derived"] or {}).get("report_cards_dir"):
        latest_dir = resolve_path(cfg, "derived.report_cards_dir")

    rc_cfg = (cfg.get("report_cards", {}) or {})
    value_mode = str(rc_cfg.get("value_mode", "percentile"))  # "raw" or "percentile"
    color_ramp = str(rc_cfg.get("color_ramp", "white_teal_blue"))  # "white_teal_blue" or "hulk"
    exclude_year = rc_cfg.get("exclude_year", None)  # e.g. 2025 or null
    include_total = bool(rc_cfg.get("include_total", True))
    area_units = str(rc_cfg.get("area_units", "sqmi"))  # "acre", "sqft", or "sqmi"
    min_display_threshold = float(rc_cfg.get("min_display_threshold", 1e-4))
    show_plots = bool(rc_cfg.get("show_plots", False))
    filename_suffix = str(rc_cfg.get("filename_suffix", ""))  # e.g. "_V4_3" or ""

    if not invasives_csv.exists():
        raise FileNotFoundError(f"Invasives CSV not found: {invasives_csv}")

    df = pd.read_csv(invasives_csv)
    df = fix_dates(df)

    if exclude_year is not None:
        df = df[df["year"] != int(exclude_year)]

    units_label = {"acre": "per acre per year", "sqft": "per sq ft per year", "sqmi": "per sq mi per year"}.get(
        area_units, "per sq mi per year"
    )

    total_area_units = calc_aoi_area(aoi_path, units=area_units)
    watershed_area_units = calc_watershed_area(bay_segments_path, "BAY_SEG_GP", units=area_units)

    ws_ab = per_watershed_abundance(df, watershed_area_units).sort_index()
    ws_rich = per_watershed_richness(df, watershed_area_units).sort_index()

    if include_total:
        ws_ab.insert(0, "Total AOI", total_aoi_abundance(df, total_area_units))
        ws_rich.insert(0, "Total AOI", total_aoi_richness(df, total_area_units))

    mode = "percentile" if value_mode.lower().startswith("percent") else "raw"

    # Archive output location
    if archive_enabled:
        out_dir = outputs_dir / run_id / "plots" / "report_cards"
    else:
        out_dir = Path("plots")  # fallback if you disable archiving

    out_ab = out_dir / f"abundance_report_card{filename_suffix}.png"
    out_rich = out_dir / f"richness_report_card{filename_suffix}.png"

    logger.info("Writing report cards to: %s", out_dir)

    plot_report_card(
        ws_ab, "Abundance", out_ab,
        color_ramp=color_ramp, mode=mode, units_label=units_label,
        min_display_threshold=min_display_threshold, show_plots=show_plots
    )
    plot_report_card(
        ws_rich, "Richness", out_rich,
        color_ramp=color_ramp, mode=mode, units_label=units_label,
        min_display_threshold=min_display_threshold, show_plots=show_plots
    )

    # Optional latest snapshot copies
    if latest_dir is not None:
        copy_to_latest(out_ab, latest_dir / out_ab.name)
        copy_to_latest(out_rich, latest_dir / out_rich.name)
        logger.info("Updated latest snapshot report cards: %s", latest_dir)

    return {"abundance_png": out_ab, "richness_png": out_rich}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()

"""
TBEP Invasives Dash app (config-driven).

Reads "latest snapshot" inputs from config/settings.yaml:
- derived.invasives_csv
- derived.invasives_geojson
- derived.flam_overlay_png
- derived.flam_meta_json
- derived.hexbins_v2_shp

And boundary layers from:
- paths.aoi_shp
- paths.municipalities_shp
- paths.bay_segments_shp

Assets are served from: src/tbep_invasives/app/assets
"""

from __future__ import annotations

import os
import json
import base64
import time
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc


# --- config helpers (fallback allows running this file directly) ---
try:
    from tbep_invasives.paths import load_config, resolve_path
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))  # adds ./src
    from tbep_invasives.paths import load_config, resolve_path


# ------------------ Helpers ------------------
def estimate_zoom_level(extent_width: float) -> int:
    if extent_width is None or extent_width <= 0:
        return 9
    if extent_width < 0.01: return 14
    if extent_width < 0.05: return 13
    if extent_width < 0.1:  return 12
    if extent_width < 0.2:  return 11
    if extent_width < 0.5:  return 10
    if extent_width < 1:    return 9
    if extent_width < 2:    return 8
    if extent_width < 5:    return 7
    return 6


def aoi_center(aoi_gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    """
    Compute a robust map center.
    Project first to avoid centroid-in-geographic-CRS issues, then transform back.
    """
    g = aoi_gdf.to_crs(epsg=3857)  # Web Mercator for centroid math
    c = g.geometry.unary_union.centroid
    c_ll = gpd.GeoSeries([c], crs=g.crs).to_crs("EPSG:4326").iloc[0]
    return float(c_ll.y), float(c_ll.x)



def b64_image(path: Path) -> str:
    with path.open("rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def safe_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def featurecollection_to_gdf(fc_dict: Dict[str, Any]) -> gpd.GeoDataFrame:
    if not isinstance(fc_dict, dict) or "features" not in fc_dict:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(fc_dict["features"], crs="EPSG:4326")


def _safe_date_str(s: Any) -> str:
    try:
        ts = pd.to_datetime(s, errors="coerce")
        return "" if pd.isna(ts) else ts.strftime("%Y-%m-%d")
    except Exception:
        return ""


# ------------------ UI (tabs) ------------------
def render_overview_tab() -> dbc.Container:
    return dbc.Container([
        html.Div([
            html.P("WELCOME TO THE TAMPA BAY NON-NATIVE SPECIES DASHBOARD",
                   style={"fontWeight": "bold", "fontSize": "1rem", "marginBottom": "0.5rem"}),

            html.Img(src="/assets/cubantreefrog.png", style={
                "width": "100%", "height": "auto", "marginBottom": "1rem", "borderRadius": "5px"
            }),

            html.P(
                "The Tampa Bay Non-Native Species Dashboard offers a comprehensive view of non-native species distributions "
                "alongside ecological indicators throughout the Tampa Bay region. By integrating spatial and temporal data "
                "with ecological metrics such as the Florida Landscape Assessment Model (FLAM), this dashboard supports "
                "informed regional management and conservation efforts."
            ),

            html.P([
                html.A("MAP:", href="#map-tab", id="map-link",
                       style={"fontWeight": "bold", "fontSize": "1rem", "cursor": "pointer",
                              "textDecoration": "underline", "color": "#0d6efd"}),
                " The Map tab enables users to visualize non-native species data overlaid with FLAM ecological rankings. "
                "Interactive filters allow exploration by species, group, year, municipality, waterbody segments, and location accuracy, "
                "providing a dynamic tool to analyze spatial patterns and identify hotspots across the bay."
            ]),

            html.P(
                "This dashboard includes non-native species records classified as both “accurate” and “approximate” locations. "
                "Hexagonal binning aggregates spatial data, balancing broad data inclusion with spatial precision to "
                "facilitate effective visualization and hotspot detection."
            ),

            html.P(
                "The Florida Landscape Assessment Model (FLAM) combines 15 weighted ecological variables sourced from key "
                "organizations (FWC, FNAI, and LCC) to rank landscapes based on ecological value. FLAM data helps highlight "
                "critical habitats and prioritize areas for conservation or restoration within Tampa Bay."
            ),

            html.P("Website and Data Information", style={"fontWeight": "bold", "fontSize": "1rem", "marginTop": "2rem"}),

            html.P([
                "Questions and comments about the dashboard can be sent to ",
                html.A("Marcus Beck", href="mailto:mbeck@tbep.org",
                       style={"textDecoration": "underline", "color": "#0d6efd"}),
                ". The page source content can be viewed on ",
                html.A("Github", href="https://github.com/tbep-tech/tbep-invasives", target="_blank",
                       style={"textDecoration": "underline", "color": "#0d6efd"}),
                ". Non-native species data were obtained from the ",
                html.A("USGS NAS Dataset", href="https://nas.er.usgs.gov", target="_blank",
                       style={"textDecoration": "underline", "color": "#0d6efd"}),
                " and the FWC FIM Database, curated to extract non-native species by the Tampa Bay Estuary Program on ",
                html.A("GitHub", href="https://github.com/kflahertywalia/tb_fim_data/tree/main/Output", target="_blank",
                       style={"textDecoration": "underline", "color": "#0d6efd"}),
                ". FLAM  data were obtained from the ",
                html.A("FWC FLAM Dataset", href="https://myfwc.com/research/gis/", target="_blank",
                       style={"textDecoration": "underline", "color": "#0d6efd"}),
                ". Invasive species of Top Concern were developed from the Suncoast CISMA ",
                html.A("plant species of concern", href="https://www.floridainvasives.org/suncoast/edrr/", target="_blank",
                       style={"textDecoration": "underline", "color": "#0d6efd"}),
                " and the FWC ",
                html.A("wildlife invasive species", href="https://myfwc.com/wildlifehabitats/profiles/#!categoryid=&subcategoryid=&status=Invasive",
                       target="_blank",
                       style={"textDecoration": "underline", "color": "#0d6efd"}),
                ". Additional information about the data products and methods used to create this dashboard can be found in TBEP technical publication ",
                html.A("#05-26", href="https://drive.google.com/file/d/1Des3GsvmFvOpd8yEXd2SPlSN-mKVxoAG/view", 
                        target="_blank",
                       style={"textDecoration": "underline", "color": "#0d6efd"}), 
                "."
            ]),
        ], style={"maxWidth": "700px", "margin": "0 auto", "paddingTop": "15px"})
    ])


def render_map_tab(group_vals, bay_vals, muni_vals, name_options, min_year, max_year) -> html.Div:
    return html.Div([
        dcc.Store(id="left-panel-style", data={
            "position": "absolute", "top": "80px", "left": "0", "width": "340px",
            "maxHeight": "calc(100vh - 80px)", "overflowY": "auto",
            "backgroundColor": "rgba(248, 249, 250, 0.9)", "zIndex": 1000,
            "padding": "20px", "borderRight": "1px solid #ccc"
        }),
        html.Button("❮", id="toggle-left", className="btn btn-light", style={
            "position": "absolute", "top": "80px", "left": "0", "zIndex": 1100,
            "borderRadius": "0 4px 4 0", "boxShadow": "2px 2px 6px rgba(0,0,0,0.2)"
        }),
        html.Div([
            dbc.Collapse(id="left-collapse", is_open=True, children=[
                html.Div([
                    html.P("Filters", className="h6"),
                    html.Label("Bay Segment"),
                    dcc.Dropdown(id="segment-dropdown",
                                 options=[{"label": s, "value": s} for s in bay_vals],
                                 multi=True, placeholder="Select segment(s)", className="mb-2"),
                    html.Label("Municipality"),
                    dcc.Dropdown(id="municipality-dropdown",
                                 options=[{"label": m, "value": m} for m in muni_vals],
                                 multi=True, placeholder="Select municipality(ies)", className="mb-2"),
                    html.Label("Group"),
                    dcc.Dropdown(id="group-dropdown",
                                 options=[{"label": g, "value": g} for g in group_vals],
                                 multi=True, placeholder="Select group(s)", className="mb-2"),
                    html.Label("Scientific or Common Name"),
                    dcc.Dropdown(id="name-dropdown",
                                 options=name_options, multi=True, searchable=True,
                                 placeholder="Type scientific/common name...", className="mb-2"),
                    html.Label("Year Range"),
                    dcc.RangeSlider(
                        id="year-slider",
                        min=min_year, max=max_year, step=1,
                        value=[min_year, max_year],
                        marks={min_year: str(min_year), max_year: str(max_year)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="mb-1"
                    ),
                    html.Div(id="year-range-display", className="mb-2"),
                    dcc.Checklist(id="topconcern-toggle",
                                  options=[{"label": " Only Top Concern species", "value": "only"}],
                                  value=[], inline=True, className="mb-2"),
                    dcc.Checklist(id="firstoccur-toggle",
                                  options=[{"label": " Only First Occurrence", "value": "only"}],
                                  value=[], inline=True, className="mb-2"),
                    dcc.Checklist(id="accuracy-toggle",
                                  options=[{"label": " Only Accurate observations", "value": "accurate"}],
                                  value=[], inline=True, className="mb-3"),

                    html.Hr(),
                    html.Label("Priority Hexbins"),
                    dcc.Input(id="priority-hex-count", type="number", value=50, min=1, step=1, style={"width": "100%"}),
                    dcc.Checklist(id="priority-hex-toggle",
                                  options=[{"label": " Show prioritized hexbins", "value": "show"}],
                                  value=["show"], inline=True, className="mb-3"),

                    html.Hr(),
                    html.Label("Map Overlays"),
                    dcc.Checklist(
                        id="overlay-layers",
                        options=[
                            {"label": " FLAM overlay (purple-green)", "value": "flam"},
                            {"label": " Heatmap (white-red)", "value": "heat"},
                            {"label": " AOI (red outline)", "value": "aoi"},
                            {"label": " Municipalities (orange outline)", "value": "muni"},
                            {"label": " Bay Segments (yellow outline)", "value": "seg"},
                        ],
                        value=["flam"],
                        inline=False,
                        className="mb-2"
                    ),
                ])
            ])
        ], id="left-panel", style={"display": "none"}),

        html.Div(id="map-no-data-banner", style={
            "position": "absolute", "top": "90px", "right": "10px", "zIndex": 1200, "maxWidth": "40%"
        }),

        dcc.Loading(
            id="loading-map",
            type="default",
            color="#00bc8c",
            style={'position': 'absolute', 'top': '300px'},
            children=[
                dcc.Graph(
                    id="map",
                    style={'position': 'fixed', 'top': "80px", 'left': 0, 'right': 0, 'bottom': 0, 'zIndex': 0},
                    config={'scrollZoom': True}
                )
            ]
        )
    ], style={"position": "fixed", "top": 0, "left": 0, "right": 0, "bottom": 0, "zIndex": 0})


def render_summary_tab() -> dbc.Container:
    return dbc.Container([
        html.Div(id="summary-no-data-banner", className="mb-2"),

        html.H4("Summary of Filtered Non-Native Species"),
        html.P("Based on map filtering, your study area contains:", className="text-muted",
               style={"marginBottom": "0.5rem"}),
        html.Div(id="active-filters", className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Total Observations"), html.H2(id="sum-total-obs")])]), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Unique Species (Scientific Names)"), html.H2(id="sum-unique-species")])]), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("Species of Concern (Unique)"), html.H2(id="sum-concern-count")])]), md=3),
            dbc.Col(dbc.Card([dbc.CardBody([html.H6("First Occurrence (Count)"), html.H2(id="sum-firstocc-count")])]), md=3),
        ], className="g-3 mt-1"),

        html.Div(className="d-flex align-items-center gap-2 mt-3", children=[
            html.Button("Download filtered data (.csv)", id="btn-download-summary", className="btn btn-secondary"),
            dcc.Download(id="download-summary-csv")
        ]),

        html.Hr(),

        html.H5("Top 10 Most Common Species"),
        html.P("Ranked by observation count within the current filter. Reported as Scientific – Common.", className="text-muted"),
        html.Ul(id="sum-top10-list", style={"columns": 2, "WebkitColumns": 2, "MozColumns": 2}),

        html.Hr(),

        html.H5("First Occurrence Species"),
        html.P("Species where at least one observation is marked as first occurrence in the current filtered dataset.",
               className="text-muted"),
        html.Ul(id="sum-firstocc-list"),

        html.Hr(),

        html.H5("Species of Concern"),
        html.P("Unique species flagged as Top Concern in the current filtered dataset.", className="text-muted"),
        html.Ul(id="sum-concern-list")
    ], fluid=True, style={"paddingTop": "24px"})


# ------------------ App factory ------------------
def create_app(cfg: Dict[str, Any]) -> Dash:
    # Resolve paths from config (latest snapshot locations)
    flam_png_path = resolve_path(cfg, "derived.flam_overlay_png")
    flam_meta_path = resolve_path(cfg, "derived.flam_meta_json")

    inv_csv_path = resolve_path(cfg, "derived.invasives_csv")
    inv_geojson_path = resolve_path(cfg, "derived.invasives_geojson")

    hex_path = resolve_path(cfg, "derived.hexbins_v2_shp")

    muni_path = resolve_path(cfg, "paths.municipalities_shp")
    seg_path = resolve_path(cfg, "paths.bay_segments_shp")
    aoi_path = resolve_path(cfg, "paths.aoi_shp")

    # Validate required files early with clear errors
    required = [
        ("FLAM overlay PNG", flam_png_path),
        ("FLAM meta JSON", flam_meta_path),
        ("Invasives CSV", inv_csv_path),
        ("Invasives GeoJSON", inv_geojson_path),
        ("Hexbins v2 shapefile", hex_path),
        ("Municipalities shapefile", muni_path),
        ("Bay Segments shapefile", seg_path),
        ("AOI shapefile", aoi_path),
    ]
    missing = [(label, p) for label, p in required if not p.exists()]
    if missing:
        msg = "Missing required app inputs:\n" + "\n".join([f"- {lbl}: {pth}" for lbl, pth in missing])
        raise FileNotFoundError(msg)

    # Load boundaries
    aoi = gpd.read_file(aoi_path).to_crs("EPSG:4326")
    aoi_geojson = json.loads(aoi.to_json())
    AOI_CENT_LAT, AOI_CENT_LON = aoi_center(aoi)

    muni = gpd.read_file(muni_path).to_crs("EPSG:4326")
    muni_geojson = json.loads(muni.to_json())

    seg = gpd.read_file(seg_path).to_crs("EPSG:4326")
    seg_geojson = json.loads(seg.to_json())

    # Load FLAM overlay + bounds
    flam_png_b64 = b64_image(flam_png_path)
    with flam_meta_path.open("r", encoding="utf-8") as f:
        flam_meta = json.load(f)
    raster_bounds_lonlat = flam_meta["bounds_lonlat"]  # [min_lon, min_lat, max_lon, max_lat]

    # Load hexbins
    hexbins = gpd.read_file(hex_path)
    if hexbins.crs != "EPSG:4326":
        hexbins = hexbins.to_crs("EPSG:4326")
    for col in ["id", "area_ac", "flam_rank"]:
        if col not in hexbins.columns:
            raise ValueError(f"Missing '{col}' in {hex_path}")

    # Load invasives geometry + attributes
    inv_gdf = gpd.read_file(inv_geojson_path).to_crs("EPSG:4326")
    inv_csv = pd.read_csv(inv_csv_path)

    # Normalize expected columns in CSV
    for need in ["group", "spatialAccuracy", "year", "baySegment", "municipality", "hexbinID",
                 "firstOccur", "topConcern", "scientificName", "commonName", "date"]:
        if need not in inv_csv.columns:
            inv_csv[need] = pd.NA

    # Bring revised CSV attributes into GeoJSON rows if needed
    gdf = inv_gdf.copy()
    needed_cols = {"group", "spatialAccuracy", "year", "baySegment", "municipality", "hexbinID",
                   "firstOccur", "topConcern", "scientificName", "commonName", "date"}

    can_merge = (
        all(k in inv_csv.columns for k in ["scientificName", "commonName", "date"]) and
        all(k in gdf.columns for k in ["scientificName", "commonName", "date"])
    )

    if not needed_cols.issubset(gdf.columns):
        if can_merge:
            gdf = gdf.merge(inv_csv, on=["scientificName", "commonName", "date"], how="left", suffixes=("", "_csv"))
        else:
            if len(inv_csv) == len(gdf):
                for c in inv_csv.columns:
                    if c not in gdf.columns:
                        gdf[c] = inv_csv[c].values

    # Final hygiene
    gdf["Year"] = pd.to_datetime(gdf.get("date"), errors="coerce").dt.year.where(gdf.get("year").isna(), gdf.get("year"))
    gdf["Year"] = gdf["Year"].fillna(gdf.get("year")).astype("Int64")

    for col in ["group", "spatialAccuracy", "baySegment", "municipality", "scientificName", "commonName"]:
        gdf[col] = safe_str_series(gdf.get(col, pd.Series(index=gdf.index)))

    gdf["hexbinID"] = pd.to_numeric(gdf.get("hexbinID", pd.Series(index=gdf.index)), errors="coerce").astype("Int64")
    gdf["firstOccur"] = pd.to_numeric(gdf.get("firstOccur", pd.Series(index=gdf.index)), errors="coerce").fillna(0).astype(int)
    gdf["topConcern"] = pd.to_numeric(gdf.get("topConcern", pd.Series(index=gdf.index)), errors="coerce").fillna(0).astype(int)

    # UI lists
    group_vals = sorted([v for v in gdf["group"].dropna().unique() if v])
    bay_vals = sorted([v for v in gdf["baySegment"].dropna().unique() if v])
    muni_vals = sorted([v for v in gdf["municipality"].dropna().unique() if v])

    name_options: List[Dict[str, str]] = []
    _seen = set()
    for name in list(gdf["scientificName"].dropna().unique()) + list(gdf["commonName"].dropna().unique()):
        if name and name not in _seen:
            name_options.append({"label": name, "value": name})
            _seen.add(name)

    valid_years = gdf["Year"].dropna().astype(int)
    min_year = int(valid_years.min()) if not valid_years.empty else 1990
    max_year = 2025 # int(valid_years.max()) if not valid_years.empty else 2025

    # group colors
    COLORS = ["#C44536", "#F4A261", "#3D5A80", "#6A4C93", "#E9C46A", "#2A9D8F", "#E07A5F"]
    group_color_map = {grp: COLORS[i % len(COLORS)] for i, grp in enumerate(group_vals)}

    # Dash assets folder (stable even if CWD changes)
    assets_dir = Path(__file__).resolve().parent / "assets"

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        assets_folder=str(assets_dir),
    )
    app.title = "TAMPA BAY NON-NATIVE SPECIES DASHBOARD"

    # Layout
    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="filtered-data-store"),
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("TAMPA BAY NON-NATIVE SPECIES DASHBOARD"),
                dbc.Tabs([
                    dbc.Tab(label="Overview", tab_id="overview-tab"),
                    dbc.Tab(label="Map", tab_id="map-tab"),
                    dbc.Tab(label="Summary", tab_id="summary-tab"),
                ], id="tabs", active_tab="overview-tab")
            ]),
            color="success", dark=True,
            style={"position": "fixed", "top": 0, "left": 0, "right": 0, "zIndex": 1030}
        ),
        html.Div([
            html.Div(id="overview-tab-content", children=render_overview_tab(), style={"display": "none"}),
            html.Div(
                id="map-tab-content",
                children=render_map_tab(group_vals, bay_vals, muni_vals, name_options, min_year, max_year),
                style={"display": "none"}
            ),
            html.Div(id="summary-tab-content", children=render_summary_tab(), style={"display": "none"})
        ], style={"marginTop": "70px"})
    ])

    # ------------------ Tab switching ------------------
    @app.callback(
        Output("overview-tab-content", "style"),
        Output("map-tab-content", "style"),
        Output("summary-tab-content", "style"),
        Input("tabs", "active_tab")
    )
    def switch_tabs(tab):
        styles = {"display": "none"}
        if tab == "overview-tab":
            return {"display": "block"}, styles, styles
        if tab == "summary-tab":
            return styles, styles, {"display": "block"}
        return styles, {"display": "block"}, styles

    @app.callback(Output("left-panel", "style"), Input("left-panel-style", "data"))
    def update_left_panel_style(style_data):  # noqa: ANN001
        return style_data

    @app.callback(
        Output("left-collapse", "is_open"),
        Output("left-panel-style", "data"),
        Input("toggle-left", "n_clicks"),
        State("left-collapse", "is_open"),
        prevent_initial_call=True
    )
    def toggle_left_panel(n, is_open):  # noqa: ANN001
        new_state = not is_open
        style = {
            "position": "absolute", "top": "80px", "left": "0", "width": "340px",
            "maxHeight": "calc(100vh - 80px)", "overflowY": "auto",
            "backgroundColor": "rgba(248, 249, 250, 0.9)", "zIndex": 1000,
            "padding": "20px", "borderRight": "1px solid #ccc"
        } if new_state else {"display": "none"}
        return new_state, style

    @app.callback(Output("year-range-display", "children"), Input("year-slider", "value"))
    def show_years(val):  # noqa: ANN001
        if not val:
            return ""
        return f"{int(val[0])} – {int(val[1])}"

    # ------------------ Filtered store (Map & Summary) ------------------
    @app.callback(
        Output("filtered-data-store", "data"),
        Input("segment-dropdown", "value"),
        Input("municipality-dropdown", "value"),
        Input("group-dropdown", "value"),
        Input("name-dropdown", "value"),
        Input("year-slider", "value"),
        Input("topconcern-toggle", "value"),
        Input("firstoccur-toggle", "value"),
        Input("accuracy-toggle", "value"),
    )
    def filter_points(seg_vals, muni_vals, grp_vals, name_vals, year_range, topconcern, firstoccur, acc):  # noqa: ANN001
        df = gdf.copy()

        # Year
        if year_range:
            y0, y1 = map(int, year_range)
            df = df[(df["Year"] >= y0) & (df["Year"] <= y1)]

        # Bay segment
        if seg_vals:
            df = df[df["baySegment"].isin(seg_vals)]

        # Municipality
        if muni_vals:
            df = df[df["municipality"].isin(muni_vals)]

        # Group
        if grp_vals:
            df = df[df["group"].isin(grp_vals)]

        # Name search (scientific or common, substring match)
        if name_vals:
            mask = pd.Series(False, index=df.index)
            sci = df["scientificName"].str.lower()
            com = df["commonName"].str.lower()
            for term in name_vals:
                t = str(term).lower()
                mask = mask | sci.str.contains(t, na=False) | com.str.contains(t, na=False)
            df = df[mask]

        # Top Concern only
        if "only" in (topconcern or []):
            df = df[df["topConcern"] == 1]

        # First Occurrence only
        if "only" in (firstoccur or []):
            df = df[df["firstOccur"] == 1]

        # Accuracy only
        if "accurate" in (acc or []):
            df = df[df["spatialAccuracy"].str.lower() == "accurate"]

        keep_cols = [
            "scientificName", "commonName", "date", "Year", "group", "spatialAccuracy",
            "baySegment", "municipality", "hexbinID", "firstOccur", "topConcern"
        ]
        out = df[keep_cols + ["geometry"]].copy()

        # JSON-safe date
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # Always return a valid FeatureCollection JSON string (even if empty)
        return out.to_json()

    # ------------------ Map figure ------------------
    @app.callback(
        Output("map", "figure"),
        Output("map-no-data-banner", "children"),
        Input("filtered-data-store", "data"),
        Input("priority-hex-count", "value"),
        Input("priority-hex-toggle", "value"),
        Input("overlay-layers", "value")
    )
    def update_map(store_json, hex_count, hex_toggle, overlays):  # noqa: ANN001
        no_data_alert = None

        # Parse safely
        if not store_json:
            df = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        else:
            try:
                df = featurecollection_to_gdf(json.loads(store_json))
            except Exception:
                df = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        fig = go.Figure()

        # Decide center/zoom
        if df.empty:
            center_lat, center_lon = AOI_CENT_LAT, AOI_CENT_LON
            zoom = 9

            fig.add_annotation(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text="No data matches current filters",
                showarrow=False, font=dict(size=14, color="red"),
                bgcolor="rgba(255,255,255,0.7)"
            )
            no_data_alert = dbc.Alert(
                "No data matches current filters. Adjust filters to see results.",
                color="warning", dismissable=True, fade=True
            )
        else:
            bounds = df.total_bounds
            center_lon, center_lat = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            extent_width = bounds[2] - bounds[0]
            zoom = estimate_zoom_level(extent_width)

            # --- Priority Hexbins ---
            if "show" in (hex_toggle or []):
                if "hexbinID" in df.columns and df["hexbinID"].notna().any():
                    df_hex = df[pd.notna(df["hexbinID"])].copy()
                    counts = df_hex["hexbinID"].value_counts()
                    hb = hexbins.copy()
                    hb = hb[hb["id"].isin(counts.index)].copy()
                    hb["obs"] = hb["id"].map(counts).fillna(0)
                else:
                    pts = df.set_geometry("geometry")
                    hb = hexbins.copy()
                    joined = gpd.sjoin(pts, hb[["id", "geometry"]], how="inner", predicate="within")
                    counts = joined["id"].value_counts()
                    hb = hb[hb["id"].isin(counts.index)].copy()
                    hb["obs"] = hb["id"].map(counts).fillna(0)

                hb["area_ac"] = pd.to_numeric(hb["area_ac"], errors="coerce")
                hb["area_sqmi"] = hb["area_ac"] / 640.0
                hb["density_sqmi"] = hb["obs"] / hb["area_sqmi"].replace({0: np.nan})

                hb["density_rank"] = hb["density_sqmi"].rank(ascending=False, method="min")
                hb["flam_rank"] = pd.to_numeric(hb["flam_rank"], errors="coerce")
                hb["priority_score"] = (hb["density_rank"] + hb["flam_rank"]) / 2.0

                top_n = int(hex_count or 50)
                top_hex = hb.nsmallest(top_n, "priority_score").copy()
                top_hex["priority_rank"] = range(1, len(top_hex) + 1)

                from matplotlib.colors import LinearSegmentedColormap, to_hex
                import matplotlib.colors as mcolors
                hex_colors = ["#01579b", "#0288d1", "#4fc3f7", "#b3e5fc", "#e6f7ff"]
                cmap = LinearSegmentedColormap.from_list("priority_hex", hex_colors)
                vmax = max(1, int(top_hex["priority_rank"].max()))
                norm = mcolors.Normalize(vmin=1, vmax=vmax)

                for _, row in top_hex.iterrows():
                    geom = row.geometry
                    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
                    rgba = cmap(norm(row["priority_rank"]))
                    fill_rgba = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.75)"
                    line_color = to_hex(rgba)
                    for poly in polys:
                        xs, ys = poly.exterior.xy
                        fig.add_trace(go.Scattermapbox(
                            lon=list(xs), lat=list(ys), mode="lines",
                            fill="toself", fillcolor=fill_rgba,
                            line=dict(color=line_color, width=1),
                            name="Priority Hex", hoverinfo="text",
                            hovertext=(f"Priority Rank: {int(row['priority_rank'])}"
                                       f"<br>Density (obs/sqmi): {row['density_sqmi']:.3f}"
                                       f"<br>FLAM Rank: {int(row['flam_rank'])}"),
                            showlegend=False
                        ))

            # --- Heatmap ---
            if "heat" in (overlays or []):
                lat = df.geometry.y.values
                lon = df.geometry.x.values
                heat_colorscale = [[0.0, "white"], [1.0, "red"]]
                fig.add_trace(go.Densitymapbox(
                    lat=lat, lon=lon,
                    z=None,
                    radius=30,
                    colorscale=heat_colorscale,
                    opacity=0.5,
                    hoverinfo="skip",
                    showscale=False
                ))

            # --- Points by group ---
            POINT_SIZE = 8
            for grp, sub in df.groupby("group"):
                if sub.empty:
                    continue
                fig.add_trace(go.Scattermapbox(
                    lon=sub.geometry.x, lat=sub.geometry.y, mode="markers",
                    marker=dict(size=POINT_SIZE, color=group_color_map.get(grp, "#444"), opacity=0.85),
                    name=str(grp), hoverinfo="text",
                    hovertext=sub.apply(lambda r: f"{r.get('scientificName','')}<br>{r.get('commonName','')}<br>{str(r.get('date',''))}", axis=1),
                    showlegend=True
                ))

        # Map layers (order matters)
        layers = []
        if "flam" in (overlays or []):
            layers.append({
                "sourcetype": "image",
                "source": flam_png_b64,
                "coordinates": [
                    [raster_bounds_lonlat[0], raster_bounds_lonlat[3]],
                    [raster_bounds_lonlat[2], raster_bounds_lonlat[3]],
                    [raster_bounds_lonlat[2], raster_bounds_lonlat[1]],
                    [raster_bounds_lonlat[0], raster_bounds_lonlat[1]],
                ],
                "opacity": 0.5, "below": "traces"
            })
        if "aoi" in (overlays or []):
            layers.append({"source": aoi_geojson, "type": "line", "color": "red", "line": {"width": 1}, "below": "traces"})
        if "muni" in (overlays or []):
            layers.append({"source": muni_geojson, "type": "line", "color": "orange", "line": {"width": 1}, "below": "traces"})
        if "seg" in (overlays or []):
            layers.append({"source": seg_geojson, "type": "line", "color": "yellow", "line": {"width": 1}, "below": "traces"})

        fig.update_layout(
            mapbox={"style": "carto-positron", "zoom": zoom, "center": {"lat": center_lat, "lon": center_lon}, "layers": layers},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(title="", bgcolor="rgba(255,255,255,0.6)", bordercolor="gray", borderwidth=1, font=dict(size=10),
                        x=0.99, y=0.99, xanchor="right", yanchor="top"),
            dragmode="zoom"
        )
        return fig, no_data_alert

    # ------------------ Summary builders ------------------
    @app.callback(
        Output("sum-total-obs", "children"),
        Output("sum-unique-species", "children"),
        Output("sum-concern-count", "children"),
        Output("sum-firstocc-count", "children"),
        Output("sum-top10-list", "children"),
        Output("sum-firstocc-list", "children"),
        Output("sum-concern-list", "children"),
        Input("filtered-data-store", "data")
    )
    def build_summary(store_json):  # noqa: ANN001
        if not store_json:
            return "0", "0", "0", "0", [], [], []

        try:
            df = featurecollection_to_gdf(json.loads(store_json))
        except Exception:
            df = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if df.empty:
            return "0", "0", "0", "0", [], [], []

        for col in ["scientificName", "commonName", "firstOccur", "topConcern", "date"]:
            if col not in df.columns:
                df[col] = pd.NA

        total_obs = len(df)
        unique_species = df["scientificName"].nunique(dropna=True)

        concern = df[df["topConcern"] == 1]
        concern_unique = concern.dropna(subset=["scientificName"]).drop_duplicates(subset=["scientificName"])
        concern_count = len(concern_unique)

        firstocc = df[df["firstOccur"] == 1].copy()
        if not firstocc.empty:
            firstocc["date_parsed"] = pd.to_datetime(firstocc["date"], errors="coerce")
            firstocc = (firstocc
                        .sort_values(["scientificName", "date_parsed"], na_position="last")
                        .drop_duplicates(subset=["scientificName"], keep="first"))
        firstocc_count = len(firstocc) if not firstocc.empty else 0

        top10_items = []
        if "scientificName" in df.columns:
            counts = df["scientificName"].value_counts().head(10)
            sci_to_common = (df.dropna(subset=["scientificName"])
                             .drop_duplicates(subset=["scientificName"])
                             .set_index("scientificName")["commonName"]
                             .to_dict())
            for sci, cnt in counts.items():
                com = sci_to_common.get(sci, "")
                label = f"{sci} – {com}" if com else f"{sci}"
                top10_items.append(html.Li(f"{label}: {cnt}"))

        firstocc_items = []
        if not firstocc.empty:
            for _, r in firstocc.iterrows():
                sci = r.get("scientificName", "")
                com = r.get("commonName", "")
                dte = _safe_date_str(r.get("date", ""))
                label = f"{sci} – {com}" if com else f"{sci}"
                firstocc_items.append(html.Li(f"{label}: {dte}"))

        concern_items = []
        if not concern_unique.empty:
            for _, r in concern_unique.iterrows():
                sci = r.get("scientificName", "")
                com = r.get("commonName", "")
                label = f"{sci} – {com}" if com else f"{sci}"
                concern_items.append(html.Li(label))

        return (f"{total_obs}",
                f"{unique_species}",
                f"{concern_count}",
                f"{firstocc_count}",
                top10_items,
                firstocc_items,
                concern_items)

    # ------------------ Applied filters + empty banner ------------------
    @app.callback(
        Output("active-filters", "children"),
        Output("summary-no-data-banner", "children"),
        Input("segment-dropdown", "value"),
        Input("municipality-dropdown", "value"),
        Input("group-dropdown", "value"),
        Input("name-dropdown", "value"),
        Input("year-slider", "value"),
        Input("topconcern-toggle", "value"),
        Input("firstoccur-toggle", "value"),
        Input("accuracy-toggle", "value"),
        Input("filtered-data-store", "data")
    )
    def show_active_filters(seg_vals, muni_vals, grp_vals, name_vals, year_range, tc, fo, acc, store_json):  # noqa: ANN001
        items = []

        if year_range:
            y0, y1 = map(int, year_range)
            if not (y0 == min_year and y1 == max_year):
                items.append(html.Li(f"Year Range: {y0}–{y1}"))
        if seg_vals:
            items.append(html.Li("Bay Segment: " + ", ".join(seg_vals)))
        if muni_vals:
            items.append(html.Li("Municipality: " + ", ".join(muni_vals)))
        if grp_vals:
            items.append(html.Li("Group: " + ", ".join(grp_vals)))
        if name_vals:
            items.append(html.Li("Name filter: " + ", ".join(name_vals)))
        if "only" in (tc or []):
            items.append(html.Li("Only Top Concern species"))
        if "only" in (fo or []):
            items.append(html.Li("Only First Occurrence"))
        if "accurate" in (acc or []):
            items.append(html.Li("Only Accurate observations"))

        applied = "" if not items else html.Div([
            html.Small("Applied Filters:", className="text-muted"),
            html.Ul(items, style={"marginTop": "0.25rem", "marginBottom": "0"})
        ])

        is_empty = False
        try:
            df = featurecollection_to_gdf(json.loads(store_json)) if store_json else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            is_empty = df.empty
        except Exception:
            is_empty = True

        summary_alert = dbc.Alert(
            "No data matches current filters. Adjust filters to see results.",
            color="warning", dismissable=True, fade=True
        ) if is_empty else None

        return applied, summary_alert

    # ------------------ CSV Download callback ------------------
    @app.callback(
        Output("download-summary-csv", "data"),
        Input("btn-download-summary", "n_clicks"),
        State("filtered-data-store", "data"),
        prevent_initial_call=True
    )
    def download_filtered_csv(n_clicks, store_json):  # noqa: ANN001
        if not store_json:
            return dash.no_update
        try:
            df = featurecollection_to_gdf(json.loads(store_json))
        except Exception:
            return dash.no_update

        if df.empty:
            return dash.no_update

        cols = ["lon", "lat", "scientificName", "commonName", "date", "Year", "group", "spatialAccuracy",
                "baySegment", "municipality", "hexbinID", "firstOccur", "topConcern"]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA

        out = df[cols].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        filename = f"nonnatives_filtered_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        return dcc.send_data_frame(out.to_csv, filename, index=False)

    # URL hash -> tab
    @app.callback(Output("tabs", "active_tab"), Input("url", "hash"))
    def hash_to_tab(url_hash):  # noqa: ANN001
        if url_hash == "#map-tab":
            return "map-tab"
        if url_hash == "#overview-tab":
            return "overview-tab"
        if url_hash == "#summary-tab":
            return "summary-tab"
        return dash.no_update

    return app


def main() -> None:
    cfg = load_config()
    app = create_app(cfg)
    port = int(cfg.get("app", {}).get("port", 8430))
    host = str(cfg.get("app", {}).get("host", "127.0.0.1"))
    debug = bool(cfg.get("app", {}).get("debug", True))
    # Dash v3+: app.run(); older Dash: app.run_server()
    if callable(getattr(app, "run", None)):
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    else:
        app.run_server(host=host, port=port, debug=debug, use_reloader=False)



if __name__ == "__main__":
    main()

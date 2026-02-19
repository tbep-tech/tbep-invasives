"""
Download and preprocess invasive species occurrence data from USGS NAS.

Outputs (configured in config/settings.yaml):
- derived_data/invasives/invasive_species.csv
- derived_data/invasives/invasive_species.geojson

Key steps:
1) Query NAS API by HUC8 list
2) Build points GeoDataFrame in EPSG:4326
3) Clip to AOI
4) Filter by minimum year
5) Spatial joins: bay segments, municipalities, hexbins
6) Flag first occurrence per species and "top concern" species
7) Expand references to columns (first ref only)
8) Write GeoJSON + CSV
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from datetime import datetime
import shutil
import pandas as pd
import geopandas as gpd
import requests
import rdata as rd


logger = logging.getLogger(__name__)

CRS_EPSG = "EPSG:4326"
NAS_OCCURRENCE_URL = "https://nas.er.usgs.gov/api/v2/occurrence/search"
FIM_URL = "https://github.com/kflahertywalia/tb_fim_data/raw/refs/heads/main/Output/tb_fim_inv.RData"
FIM_INVASIVES_KEY = "https://github.com/kflahertywalia/tb_fim_data/raw/refs/heads/main/Output/fim_nonnative_list.csv"

# --- import config helpers with a fallback for running this file directly ---
try:
    from tbep_invasives.paths import load_config, resolve_path
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))  # adds ./src
    from tbep_invasives.paths import load_config, resolve_path

def get_run_id(cfg: dict) -> str:
    run_id = (cfg.get("run", {}) or {}).get("run_id")
    if run_id:
        return str(run_id)
    # default: YYYYMMDD (simple + deterministic enough if you run once per quarter)
    return datetime.now().strftime("%Y%m%d")


def write_outputs(gdf: gpd.GeoDataFrame, out_csv: Path, out_geojson: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.parent.mkdir(parents=True, exist_ok=True)

    gdf_out = gdf.to_crs("EPSG:4326")
    out_geojson.write_text(gdf_out.to_json(), encoding="utf-8")
    gdf_out.drop(columns=["geometry"], errors="ignore").to_csv(out_csv, index=False)


def copy_to_latest(src_csv: Path, src_geojson: Path, dst_csv: Path, dst_geojson: Path) -> None:
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    dst_geojson.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_csv, dst_csv)
    shutil.copy2(src_geojson, dst_geojson)

def _unique_preserve_order(items: Sequence[str]) -> List[str]:
    """Deduplicate a list while preserving order."""
    return list(dict.fromkeys(items))


def fetch_invasive_species_data(
    huc8_list: Sequence[str],
    *,
    verify_ssl: Union[bool, str] = True,
    timeout_s: int = 60,
) -> List[Dict[str, Any]]:
    """
    Fetch invasive species occurrences from USGS NAS for each HUC8.

    verify_ssl can be:
      - True (default): verify TLS certs
      - False: disable verification (not recommended; useful behind some proxies)
      - path to a CA bundle file
    """
    huc8_list = _unique_preserve_order([h.strip() for h in huc8_list if str(h).strip()])

    if verify_ssl is False:
        # Silence only the warning (we still want other warnings/errors)
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        logger.warning("TLS verification disabled for NAS API requests (verify_ssl=False).")

    all_rows: List[Dict[str, Any]] = []
    session = requests.Session()

    for huc8 in huc8_list:
        url = f"{NAS_OCCURRENCE_URL}?huc8={huc8}"
        try:
            resp = session.get(url, verify=verify_ssl, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()

            # NAS sometimes returns {"results":[...]} or a raw list
            rows = data.get("results", data if isinstance(data, list) else [])
            if not isinstance(rows, list):
                rows = []
            all_rows.extend(rows)

            logger.info("Fetched %s rows for HUC8=%s", len(rows), huc8)
        except Exception as exc:
            logger.exception("Fetch failed for HUC8=%s (%s)", huc8, exc)

    logger.info("Total rows fetched: %s", len(all_rows))
    return all_rows

def parse_fim_reference_date(reference: str) -> Dict[str, Any]:
    """
    Parse FIM Reference field to extract date components.
    Format: TBD2005013902 or TBM2000051203
    Where: TBD/TBM + YYYY + MM + DD + sequence
    """
    try:
        if not isinstance(reference, str) or len(reference) < 13:
            return {'year': None, 'month': None, 'day': None}
        
        # Extract date portion (skip first 3 chars like TBD/TBM)
        date_part = reference[3:11]  # YYYYMMDD
        year = int(date_part[0:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        
        # Validate and handle invalid dates
        if month < 1 or month > 12:
            month = None
        if day < 1 or day > 31:
            day = None
            
        return {'year': year, 'month': month, 'day': day}
    except (ValueError, IndexError):
        return {'year': None, 'month': None, 'day': None}


def fetch_fim_invasives_data(fim_url: str) -> List[Dict[str, Any]]:
    """Fetch invasive species list from FIM RData - returns list of dicts like NAS function."""
    try:
            
        # Fetch and parse the RData file
        rdata_content = requests.get(fim_url).content
        parsed = rd.parser.parse_data(rdata_content)
        rdata_converted = rd.conversion.convert(parsed)

        # Extract the DataFrame from the 'inv' key
        df = rdata_converted['inv']
        
        # Parse date from Reference field
        date_info = df['Reference'].apply(parse_fim_reference_date)
        df['year'] = [d['year'] for d in date_info]
        df['month'] = [d['month'] for d in date_info]
        df['day'] = [d['day'] for d in date_info]
        
        # Rename columns to match NAS format
        df = df.rename(columns={
            'Scientificname': 'scientificName',
            'Commonname': 'commonName',
            'Latitude': 'decimalLatitude',
            'Longitude': 'decimalLongitude',
            'Taxa_Type': 'group',
            'Reference': 'key',  # Use Reference as unique identifier
        })
        
        # Add placeholder columns that exist in NAS but not in FIM
        df['speciesID'] = None
        df['family'] = df.get('family_nm', None)  # Use family_nm if available
        df['genus'] = None
        df['species'] = None
        df['state'] = 'Florida'
        df['county'] = None
        df['locality'] = 'Tampa Bay'
        df['latLongSource'] = 'FIM Survey'
        df['latLongAccuracy'] = 'GPS'
        df['Centroid Type'] = ''
        df['huc8Name'] = ''
        df['huc8'] = ''
        df['huc10Name'] = ''
        df['huc10'] = ''
        df['huc12Name'] = ''
        df['huc12'] = ''
        df['status'] = 'collected'
        df['comments'] = ''
        df['recordType'] = 'Specimen'
        df['disposal'] = ''
        df['museumCatNumber'] = ''
        df['freshMarineIntro'] = 'Marine'
        df['UUID'] = ''
        df['references'] = None  # FIM doesn't have references like NAS
        
        # Build date string (format: 'YYYY-M' or 'YYYY-M-D')
        def build_date_string(row):
            if pd.isna(row['year']):
                return None
            date_str = str(int(row['year']))
            if pd.notna(row['month']):
                date_str += f"-{int(row['month'])}"
                if pd.notna(row['day']):
                    date_str += f"-{int(row['day'])}"
            return date_str
        
        df['date'] = df.apply(build_date_string, axis=1)
        
        # Normalize group names to match NAS (if needed)
        group_mapping = {
            'Fish': 'Fishes',
            'Turtle': 'Reptiles',
            # Add more mappings as needed
        }
        if 'group' in df.columns:
            df['group'] = df['group'].replace(group_mapping)

        # Remove any rows where group is not Fishes or Reptiles
        df = df[df['group'].isin(['Fishes', 'Reptiles'])]

        # Select only the columns that match NAS format (in similar order)
        nas_columns = [
            'key', 'speciesID', 'group', 'family', 'genus', 'species',
            'scientificName', 'commonName', 'state', 'county', 'locality',
            'decimalLatitude', 'decimalLongitude', 'latLongSource', 'latLongAccuracy',
            'Centroid Type', 'huc8Name', 'huc8', 'huc10Name', 'huc10',
            'huc12Name', 'huc12', 'date', 'year', 'month', 'day',
            'status', 'comments', 'recordType', 'disposal', 'museumCatNumber',
            'freshMarineIntro', 'UUID', 'references'
        ]
        
        # Keep only columns that exist and are in our target list
        available_columns = [col for col in nas_columns if col in df.columns]
        df = df[available_columns]
        
        # Convert to list of dicts (same format as NAS function)
        records = df.to_dict('records')
        
        logger.info("Fetched FIM invasives data with %s rows.", len(records))
        return records

    except Exception as exc:
        logger.exception("Failed to fetch/process FIM invasives data: %s", exc)
        return []

def clean_and_format_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a 'date' column from year/month/day.
    - Drop rows missing year
    - Fill missing month with 6 and day with 15
    """
    if "year" not in df.columns:
        return df

    df = df[df["year"].notna()].copy()

    if "month" not in df.columns:
        df["month"] = pd.NA
    if "day" not in df.columns:
        df["day"] = pd.NA

    df["month"] = df["month"].fillna(6).astype(int)
    df["day"] = df["day"].fillna(15).astype(int)

    df["date"] = df.apply(
        lambda r: f"{int(r['year'])}-{int(r['month']):02d}-{int(r['day']):02d}", axis=1
    )
    return df


def clip_to_aoi(gdf: gpd.GeoDataFrame, aoi_path: Path) -> gpd.GeoDataFrame:
    """Clip point observations to AOI polygon (if AOI exists)."""
    if not aoi_path.exists():
        logger.warning("AOI shapefile not found; skipping clip: %s", aoi_path)
        return gdf

    aoi = gpd.read_file(aoi_path)
    if aoi.crs != gdf.crs:
        aoi = aoi.to_crs(gdf.crs)
    return gpd.clip(gdf, aoi)

def filter_by_year(
    gdf: gpd.GeoDataFrame,
    *,
    min_year: int,
    date_col: str = "date",
) -> gpd.GeoDataFrame:
    """Filter to observations occurring on/after Jan 1 of min_year."""
    if date_col not in gdf.columns:
        logger.warning("No '%s' column found; skipping year filter.", date_col)
        return gdf

    dt = pd.to_datetime(gdf[date_col], errors="coerce")
    keep = dt >= pd.Timestamp(year=min_year, month=1, day=1)
    out = gdf.loc[keep].copy()
    logger.info("Year filter (>= %s): kept %s of %s rows", min_year, len(out), len(gdf))
    return out

def spatial_intersect_add_fields(
    gdf: gpd.GeoDataFrame,
    shp_path: Path,
    fields: Sequence[str],
    *,
    rename: Optional[Dict[str, str]] = None,
    predicate: str = "within",
    keep: str = "first",
) -> gpd.GeoDataFrame:
    """
    Spatial join fields from shp_path into gdf.

    - fields: list of fields to pull from the right dataset
    - rename: optional mapping for incoming columns
    """
    if not shp_path.exists():
        logger.warning("Spatial join source not found; skipping: %s", shp_path)
        return gdf

    right = gpd.read_file(shp_path)
    if right.crs != gdf.crs:
        right = right.to_crs(gdf.crs)

    # Avoid conflicts with sjoin bookkeeping columns
    for col in ("index_right", "index_left"):
        if col in gdf.columns:
            gdf = gdf.drop(columns=col)
        if col in right.columns:
            right = right.drop(columns=col)

    fields_present = [c for c in fields if c in right.columns]
    missing = sorted(set(fields) - set(fields_present))
    if missing:
        logger.warning("Missing fields in %s: %s", shp_path.name, missing)

    sel = right[fields_present + ["geometry"]].copy()
    joined = gpd.sjoin(gdf, sel, how="left", predicate=predicate)

    if joined.index.has_duplicates:
        joined = joined[~joined.index.duplicated(keep=keep)]

    if "index_right" in joined.columns:
        joined = joined.drop(columns="index_right")

    if rename:
        joined = joined.rename(columns={k: v for k, v in rename.items() if k in joined.columns})

    return joined


def extract_reference_columns(df: pd.DataFrame, ref_col: str = "references") -> pd.DataFrame:
    """
    Parse the `references` field (often a stringified list of dicts) and expand the FIRST entry.
    New columns are prefixed with 'references_'.
    """
    def _first_ref(rec: Any) -> Dict[str, Any]:
        if isinstance(rec, list):
            return rec[0] if rec else {}
        if pd.isna(rec) or not isinstance(rec, str) or not rec.strip():
            return {}
        try:
            parsed = ast.literal_eval(rec)
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                return first if isinstance(first, dict) else {}
        except (ValueError, SyntaxError):
            return {}
        return {}

    first_refs = df[ref_col].apply(_first_ref)
    refs_df = pd.json_normalize(first_refs).add_prefix("references_")
    return pd.concat([df.reset_index(drop=True), refs_df], axis=1)


def add_first_occurrence_flag(
    gdf: gpd.GeoDataFrame,
    *,
    species_col: str = "scientificName",
    date_col: str = "date",
    out_col: str = "firstOccur",
) -> gpd.GeoDataFrame:
    """Mark earliest observation per species as 1; others 0."""
    if species_col not in gdf.columns or date_col not in gdf.columns:
        logger.warning("Missing %s or %s; skipping first occurrence flag.", species_col, date_col)
        gdf[out_col] = 0
        return gdf

    dates = pd.to_datetime(gdf[date_col], errors="coerce")
    min_dates = dates.groupby(gdf[species_col]).transform("min")
    gdf[out_col] = (dates == min_dates).astype(int)
    return gdf


def add_top_concern(
    gdf: gpd.GeoDataFrame,
    concern_csv: Path,
    *,
    species_col: str = "scientificName",
    common_col: str = "commonName",
    out_col: str = "topConcern",
    do_fuzzy: bool = True,
    fuzzy_threshold: int = 90,
) -> gpd.GeoDataFrame:
    """Flag rows matching 'top concern' species/common names from a CSV."""
    if not concern_csv.exists():
        logger.warning("Concern CSV not found; setting %s=0: %s", out_col, concern_csv)
        gdf[out_col] = 0
        return gdf

    concerns = pd.read_csv(concern_csv)

    cols_lower = {c.lower(): c for c in concerns.columns}
    sci_cols = [cols_lower[c] for c in cols_lower if "scient" in c]
    com_cols = [cols_lower[c] for c in cols_lower if "common" in c]

    def _norm(x: Any) -> str:
        return str(x).strip().lower() if pd.notna(x) else ""

    sci_set = set(_norm(x) for c in sci_cols for x in concerns[c].dropna().unique()) if sci_cols else set()
    com_set = set(_norm(x) for c in com_cols for x in concerns[c].dropna().unique()) if com_cols else set()

    exact_hit = gdf[species_col].map(lambda x: _norm(x) in sci_set) | gdf[common_col].map(lambda x: _norm(x) in com_set)

    fuzzy_hit = pd.Series(False, index=gdf.index)

    if do_fuzzy:
        try:
            from rapidfuzz import process, fuzz  # type: ignore

            concern_names = list(sci_set | com_set)

            # scientific fuzzy
            candidates = gdf.loc[~exact_hit, species_col].fillna("").astype(str)
            for idx, name in candidates.items():
                if not name:
                    continue
                match = process.extractOne(name, concern_names, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= fuzzy_threshold:
                    fuzzy_hit.at[idx] = True

            # common fuzzy
            candidates = gdf.loc[~exact_hit & ~fuzzy_hit, common_col].fillna("").astype(str)
            for idx, name in candidates.items():
                if not name:
                    continue
                match = process.extractOne(name, concern_names, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= fuzzy_threshold:
                    fuzzy_hit.at[idx] = True

        except ImportError:
            logger.warning("rapidfuzz not installed; skipping fuzzy matching.")

    gdf[out_col] = (exact_hit | fuzzy_hit).astype(int)
    logger.info("TopConcern flagged: %s rows", int(gdf[out_col].sum()))
    return gdf


def run(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Run the full NAS + FIM download + spatial enrichment step."""
    # --- config ---
    huc8_list = cfg.get("parameters", {}).get(
        "huc8_list",
        [
            "03100101",
            "03100201",
            "03100202",
            "03100203",
            "03100204",
            "03100205",
            "03100206",
            "03100207",
            "03100208",
        ],
    )
    min_year = int(cfg.get("parameters", {}).get("min_year", 2000))

    verify_ssl = cfg.get("requests", {}).get("verify_ssl", True)
    ca_bundle = cfg.get("requests", {}).get("ca_bundle", None)
    if ca_bundle:
        verify_ssl = str(ca_bundle)

    aoi_path = resolve_path(cfg, "paths.aoi_shp")
    bay_segments_path = resolve_path(cfg, "paths.bay_segments_shp")
    municipalities_path = resolve_path(cfg, "paths.municipalities_shp")
    hexbins_path = resolve_path(cfg, "paths.hexbins_shp")
    concern_csv = resolve_path(cfg, "paths.invasives_concern_csv")

    out_geojson = resolve_path(cfg, "derived.invasives_geojson")
    out_csv = resolve_path(cfg, "derived.invasives_csv")
    out_geojson.parent.mkdir(parents=True, exist_ok=True)

    # --- fetch NAS data ---
    logger.info("Fetching NAS invasive species data...")
    nas_data = fetch_invasive_species_data(huc8_list, verify_ssl=verify_ssl)
    logger.info("NAS records fetched: %s", len(nas_data))
    
    # --- fetch FIM data ---
    logger.info("Fetching FIM invasive species data...")
    fim_data = fetch_fim_invasives_data(FIM_URL)
    logger.info("FIM records fetched: %s", len(fim_data))
    
    # --- combine datasets ---
    combined_data = nas_data + fim_data
    logger.info("Total combined records: %s", len(combined_data))
    
    df = pd.DataFrame(combined_data)

    if df.empty:
        logger.warning("No records returned. Writing empty outputs.")
        gdf_empty = gpd.GeoDataFrame(df, geometry=[], crs=CRS_EPSG)
        out_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        df.to_csv(out_csv, index=False)
        return {"csv": out_csv, "geojson": out_geojson}

    # Rename lat/lon (already standardized in both datasets)
    df = df.rename(columns={"decimalLatitude": "lat", "decimalLongitude": "lon"})

    # Build date (this will handle rows that already have date vs those that need it)
    df = clean_and_format_dates(df)

    # Normalize group names (optional)
    group_name_changes = {
        "Amphibians-Frogs": "Amphibians",
        "Crustaceans-Crabs": "Crustaceans",
        "Marine Fishes": "Fishes",
        "Mollusks-Bivalves": "Mollusks",
        "Mollusks-Gastropods": "Mollusks",
        "Reptiles-Lizards": "Reptiles",
        "Reptiles-Snakes": "Reptiles",
        "Reptiles-Turtles": "Reptiles",
    }
    if "group" in df.columns:
        df["group"] = df["group"].replace(group_name_changes)

    # Remove any rows where group is not those from above
    df = df[df['group'].isin(["Amphibians", "Crustaceans", "Fishes", "Mammals", "Mollusks", "Plants", "Reptiles"])]

    # Geometry
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"], crs=CRS_EPSG),
    )

    # Clip + filter
    gdf = clip_to_aoi(gdf, aoi_path)
    gdf = filter_by_year(gdf, min_year=min_year, date_col="date")

    # Spatial joins
    gdf = spatial_intersect_add_fields(
        gdf, bay_segments_path, fields=["BAY_SEG_GP"], predicate="within", keep="first"
    )
    gdf = spatial_intersect_add_fields(
        gdf, municipalities_path, fields=["Municipali"], predicate="within", keep="first"
    )
    gdf = spatial_intersect_add_fields(
        gdf, hexbins_path, fields=["id"], rename={"id": "hexbinID"}, predicate="within", keep="first"
    )

    # Flags + concern list
    gdf = add_first_occurrence_flag(gdf, species_col="scientificName", date_col="date", out_col="firstOccur")
    gdf = add_top_concern(
        gdf,
        concern_csv,
        species_col="scientificName",
        common_col="commonName",
        out_col="topConcern",
        do_fuzzy=True,
        fuzzy_threshold=90,
    )

    # Expand references (only for NAS data that has references)
    if "references" in gdf.columns:
        gdf = extract_reference_columns(gdf, ref_col="references")

    # Common name formatting
    if "commonName" in gdf.columns:
        gdf["commonName"] = gdf["commonName"].astype("string").str.strip().str.title()

    # Drop non-needed columns safely
    gdf = gdf.drop(
        [
            "family",
            "genus",
            "species",
            "state",
            "county",
            "locality",
            "latLongSource",
            "Centroid Type",
            "huc8Name",
            "huc8",
            "huc10Name",
            "huc10",
            "huc12Name",
            "huc12",
            "month",
            "day",
            "status",
            "comments",
            "disposal",
            "museumCatNumber",
            "freshMarineIntro",
            "UUID",
            "references",
            "references_key",
            "references_year",
            "references_publisher",
            "references_publisherLocation",
        ],
        axis=1,
        errors="ignore",
    )

    # Rename columns
    gdf = gdf.rename(
        columns={
            "latLongAccuracy": "spatialAccuracy",
            "BAY_SEG_GP": "baySegment",
            "Municipali": "municipality",
            "references_refType": "refType",
            "references_author": "refAuthor",
            "references_title": "refTitle",
        }
    )

    # Write GeoJSON (robust, avoids driver issues)
    outputs_dir = resolve_path(cfg, "paths.outputs_dir")
    run_id = get_run_id(cfg)
    
    archive_enabled = bool((cfg.get("run", {}) or {}).get("archive_outputs", True))
    update_derived = bool((cfg.get("run", {}) or {}).get("update_derived", True))
    
    # canonical "latest" paths for the app
    latest_csv = resolve_path(cfg, "derived.invasives_csv")
    latest_geojson = resolve_path(cfg, "derived.invasives_geojson")
    
    if archive_enabled:
        archive_csv = outputs_dir / run_id / "invasives" / latest_csv.name
        archive_geojson = outputs_dir / run_id / "invasives" / latest_geojson.name
    
        write_outputs(gdf, archive_csv, archive_geojson)
    
        if update_derived:
            copy_to_latest(archive_csv, archive_geojson, latest_csv, latest_geojson)
    else:
        # if you ever disable archiving, write directly to derived_data
        write_outputs(gdf, latest_csv, latest_geojson)

    logger.info("Final output: %s total records (%s NAS, %s FIM)", 
                len(gdf), len(nas_data), len(fim_data))

    return {"csv": out_csv, "geojson": out_geojson}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = load_config()  # defaults to config/settings.yaml
    run(cfg)


if __name__ == "__main__":
    main()

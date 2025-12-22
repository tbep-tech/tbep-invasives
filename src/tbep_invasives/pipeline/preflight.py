from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tbep_invasives.paths import resolve_path
except ImportError:  # pragma: no cover
    # Allows running this file directly during development
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tbep_invasives.paths import resolve_path


@dataclass(frozen=True)
class CheckItem:
    label: str
    path: Path
    kind: str  # "file" | "dir" | "shapefile"


def _check_shapefile_set(shp: Path) -> Tuple[bool, List[str]]:
    """
    Shapefiles are a file *set*. In practice, missing .dbf or .shx causes confusing failures.
    We require: .shp, .dbf, .shx. (.prj/.cpg are recommended but not required.)
    """
    missing: List[str] = []
    if shp.suffix.lower() != ".shp":
        return False, [f"Not a .shp file: {shp.name}"]

    base = shp.with_suffix("")
    required = [base.with_suffix(".shp"), base.with_suffix(".dbf"), base.with_suffix(".shx")]
    for p in required:
        if not p.exists():
            missing.append(p.name)

    return (len(missing) == 0), missing


def _can_create_dir(p: Path) -> Optional[str]:
    try:
        p.mkdir(parents=True, exist_ok=True)
        test_file = p / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)  # type: ignore[arg-type]
        return None
    except Exception as exc:
        return str(exc)


def preflight(cfg: Dict[str, Any], *, mode: str = "quarterly") -> None:
    """
    mode:
      - "quarterly": validates required inputs for pipeline steps and output dirs
      - "app": validates required derived outputs + boundary layers for dashboard
    """

    run_cfg = cfg.get("run", {}) or {}
    strict = bool(run_cfg.get("preflight_strict", True))

    checks: List[CheckItem] = []

    # Shared boundary layers (used by multiple steps and app)
    checks += [
        CheckItem("AOI shapefile", resolve_path(cfg, "paths.aoi_shp"), "shapefile"),
        CheckItem("Municipalities shapefile", resolve_path(cfg, "paths.municipalities_shp"), "shapefile"),
        CheckItem("Bay Segments shapefile", resolve_path(cfg, "paths.bay_segments_shp"), "shapefile"),
    ]

    if mode == "quarterly":
        # Pipeline inputs
        checks += [
            CheckItem("Hexbins shapefile (base)", resolve_path(cfg, "paths.hexbins_shp"), "shapefile"),
            CheckItem("FLAM raster", resolve_path(cfg, "paths.flam_raster"), "file"),
            CheckItem("Top concern species CSV", resolve_path(cfg, "paths.invasives_concern_csv"), "file"),
        ]

        # Output root dirs (must be creatable)
        outputs_dir = resolve_path(cfg, "paths.outputs_dir")
        derived_root = resolve_path(cfg, "paths.derived_root") if "paths" in cfg and (cfg["paths"] or {}).get("derived_root") else None

        # Always ensure outputs dir is writable if archiving is enabled
        if bool(run_cfg.get("archive_outputs", True)):
            err = _can_create_dir(outputs_dir)
            if err and strict:
                raise RuntimeError(f"Cannot create/write outputs_dir: {outputs_dir}\nReason: {err}")

        # Ensure derived dirs are writable if updating derived snapshot
        if bool(run_cfg.get("update_derived", True)) and derived_root is not None:
            err = _can_create_dir(derived_root)
            if err and strict:
                raise RuntimeError(f"Cannot create/write derived_root: {derived_root}\nReason: {err}")

    if mode == "app":
        # App requires latest snapshot artifacts
        checks += [
            CheckItem("Derived invasives CSV", resolve_path(cfg, "derived.invasives_csv"), "file"),
            CheckItem("Derived invasives GeoJSON", resolve_path(cfg, "derived.invasives_geojson"), "file"),
            CheckItem("Derived FLAM overlay PNG", resolve_path(cfg, "derived.flam_overlay_png"), "file"),
            CheckItem("Derived FLAM meta JSON", resolve_path(cfg, "derived.flam_meta_json"), "file"),
            CheckItem("Derived hexbins v2 shapefile", resolve_path(cfg, "derived.hexbins_v2_shp"), "shapefile"),
        ]

    missing_msgs: List[str] = []
    for item in checks:
        if item.kind == "dir":
            if not item.path.exists():
                missing_msgs.append(f"- {item.label}: missing directory {item.path}")
        elif item.kind == "shapefile":
            ok, missing_parts = _check_shapefile_set(item.path)
            if not ok:
                if item.path.exists():
                    missing_msgs.append(
                        f"- {item.label}: shapefile set incomplete at {item.path} (missing: {', '.join(missing_parts)})"
                    )
                else:
                    missing_msgs.append(f"- {item.label}: missing {item.path}")
        else:
            if not item.path.exists():
                missing_msgs.append(f"- {item.label}: missing {item.path}")

    if missing_msgs:
        message = (
            "Preflight failed. Fix the following before running:\n"
            + "\n".join(missing_msgs)
            + "\n\nTip: verify config/settings.yaml path keys match your folder structure."
        )
        if strict:
            raise FileNotFoundError(message)
        else:
            # Non-strict mode: print warnings but continue (useful for development)
            print(message)

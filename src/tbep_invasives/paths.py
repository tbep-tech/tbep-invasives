from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Walk upward from `start` (or this file) to find the repository root.
    Repo root is identified by presence of config/settings.yaml OR pyproject.toml.
    """
    if start is None:
        start = Path(__file__).resolve()

    for p in [start, *start.parents]:
        if (p / "config" / "settings.yaml").exists() or (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError(
        "Could not find repo root. Expected config/settings.yaml or pyproject.toml in a parent directory."
    )


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    repo_root = find_repo_root()
    cfg_path = Path(config_path) if config_path else (repo_root / "config" / "settings.yaml")
    cfg_path = cfg_path if cfg_path.is_absolute() else (repo_root / cfg_path)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format in {cfg_path}; expected a YAML mapping/object.")
    cfg["_config_path"] = str(cfg_path)
    cfg["_repo_root"] = str(repo_root)
    return cfg


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """
    Resolve a path from config using dotted keys, e.g.:
      resolve_path(cfg, "paths.aoi_shp")
      resolve_path(cfg, "derived.invasives_csv")
    """
    repo_root = Path(cfg["_repo_root"])
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing config key: {key}")
        cur = cur[part]

    if not isinstance(cur, str):
        raise TypeError(f"Config key {key} must be a string path.")

    p = Path(cur)
    return p if p.is_absolute() else (repo_root / p)


@dataclass(frozen=True)
class RepoPaths:
    repo_root: Path
    config_path: Path

    input_data: Path
    derived_data: Path
    outputs: Path


def get_repo_paths(cfg: Dict[str, Any]) -> RepoPaths:
    repo_root = Path(cfg["_repo_root"])
    config_path = Path(cfg["_config_path"])

    input_data = resolve_path(cfg, "paths.input_data_dir")
    derived_data = resolve_path(cfg, "paths.derived_data_dir")
    outputs = resolve_path(cfg, "paths.outputs_dir")

    return RepoPaths(
        repo_root=repo_root,
        config_path=config_path,
        input_data=input_data,
        derived_data=derived_data,
        outputs=outputs,
    )

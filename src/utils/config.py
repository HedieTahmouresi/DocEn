"""Config loader and layering utility.

Implements base -> env -> exp YAML configuration merging and environment path resolution.
No absolute paths in committed code.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load YAML file safely into a dictionary."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_paths(config: Dict[str, Any], root_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Resolve DATA_ROOT and RUNS_ROOT from environment variables or gitignored paths.yaml.
    Guarantees no hardcoded absolute paths in committed files.
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent.parent

    paths_file = root_dir / "paths.yaml"
    local_paths = load_yaml(paths_file)

    data_root = os.environ.get("DATA_ROOT") or local_paths.get("DATA_ROOT") or str(root_dir / "data")
    runs_root = os.environ.get("RUNS_ROOT") or local_paths.get("RUNS_ROOT") or str(root_dir / "runs")

    config["data_root"] = str(Path(data_root).resolve())
    config["runs_root"] = str(Path(runs_root).resolve())
    return config


def load_config(
    env: str = "local_cpu",
    exp_file: Optional[str] = None,
    base_file: Optional[str] = None,
    root_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load and layer configuration: base.yaml -> env/{env}.yaml -> exp/{exp_file}.yaml
    """
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent.parent

    if base_file is None:
        base_path = root_dir / "configs" / "base.yaml"
    else:
        base_path = Path(base_file)

    config = load_yaml(base_path)

    env_path = root_dir / "configs" / "env" / f"{env}.yaml"
    if env_path.exists():
        env_config = load_yaml(env_path)
        config = deep_merge(config, env_config)

    if exp_file:
        exp_path = Path(exp_file)
        if not exp_path.exists():
            exp_path = root_dir / "configs" / "exp" / exp_file
        if not exp_path.exists() and not exp_file.endswith(".yaml"):
            exp_path = root_dir / "configs" / "exp" / f"{exp_file}.yaml"

        if exp_path.exists():
            exp_config = load_yaml(exp_path)
            config = deep_merge(config, exp_config)

    config = resolve_paths(config, root_dir=root_dir)
    return config


def save_resolved_config(config: Dict[str, Any], save_dir: Path) -> None:
    """Save the fully resolved configuration to a run directory."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / "config.yaml"
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

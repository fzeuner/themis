#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fr Jul 4 11:52:55 2025

@author: franziskaz

Configuration file

Files supported: scan, dark, flat
Reduction levels supported: raw, l0

"""
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

# Python 3.11+ has tomllib in the stdlib. For 3.12 (used here) this is available.
try:
    import tomllib  # type: ignore[attr-defined]
except Exception as e:  # pragma: no cover
    tomllib = None

# Core dependencies used to assemble a configuration
from themis.core import themis_data_reduction as tdr
from themis.core import cam_config as cc

class DataType:
    def __init__(self, name, file_ext):
        self.name = name
        self.file_ext = file_ext

    def __repr__(self):
        return f"<DataType(name={self.name}, file_extension={self.file_ext})>"


class DataTypeRegistry:
    def __init__(self):
        self._types = {}

    def add(self, datatype):
        self._types[datatype.name] = datatype

    def __getitem__(self, name):
        return self._types[name]

    def __getattr__(self, name):
        if name in self._types:
            return self._types[name]
        raise AttributeError(f"No data type named '{name}'")

    def list_types(self):
        return list(self._types.keys())

"""
Configuration loading

This module supports both hardcoded defaults and external configuration files
in TOML format. Use get_config(config_path=...) to populate dataset variables
from a TOML file. If no config is provided, the previous hardcoded defaults
are used to preserve backward compatibility.

Example TOML structure:

    [dataset]
    line = "sr"
    date = "2025-07-07"
    sequence = 26
    dark_sequence = 1
    flat_sequence = 25
    states = ["pQ", "mQ", "pU", "mU", "pV", "mV"]

    [paths]
    base = "/home/USER/data/themis"          # optional; defaults to prior value
    figures = "/home/USER/figures/themis/"   # optional override
    inversion = "/home/USER/data/themis/inversion"  # optional override

    [params]
    slit_width = 0.33

Notes:
- file_types are fixed as ["scan", "dark", "flat"] to match the existing code.
"""

# defaults (preserve behavior if no config file is supplied)
DEFAULTS = {
    "dataset": {
        "line": "sr",
        "date": "2025-07-07",
        "sequence": 26,
        "dark_sequence": 1,
        "flat_sequence": 25,
        "states": ["pQ", "mQ", "pU", "mU", "pV", "mV"],
    },
    "paths": {
        "base": "/home/franziskaz/data/themis",
        # figures and inversion derived below if not provided
    },
    "params": {
        "slit_width": 0.33,
    },
}

file_types = ['scan', 'dark', 'flat']

# ++++++++++++++++++++++++++++++++++++++++++

# Create data types
dark = DataType(name=file_types[1], file_ext="_x3")
flat = DataType(name=file_types[2], file_ext="_y3")
scan = DataType(name=file_types[0], file_ext="_b3")

class DirectoryPaths:
    def __init__(self, date: str, base: str, figures: Optional[str] = None,
                 inversion: Optional[str] = None, auto_create: bool = False):
        self.base = Path(base)
        self.date = date

        # Define paths
        self.raw = self.base / "rdata" / date
        self.reduced = self.base / "pdata" / date
        self.figures = Path(figures) if figures else Path("/home/franziskaz/figures/themis/")
        self.inversion = Path(inversion) if inversion else (self.base / "inversion")

        # Automatically create directories if they don't exist
        if auto_create:
            self._ensure_paths_exist()

    def _ensure_paths_exist(self):
        for path in [self.raw, self.reduced, self.figures, self.inversion]:
            path.mkdir(parents=True, exist_ok=True)
            
    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return "\n".join(f"{key}: {getattr(self, key)}" for key in ['raw', 'reduced', 'figures', 'inversion'])

def _deep_get(dct: dict, path: str, default=None):
    cur = dct
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_config_from_toml(config_path: Path) -> dict:
    if tomllib is None:
        raise RuntimeError("tomllib is not available. Please use Python >= 3.11 or install tomli for older versions.")
    with config_path.open('rb') as f:
        data = tomllib.load(f)
    return data


def _merge_defaults(user_cfg: Optional[dict]) -> dict:
    # merge DEFAULTS with user_cfg (shallow nested merge for our keys)
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if not user_cfg:
        return cfg
    for top_key in ("dataset", "paths", "params"):
        if top_key in user_cfg and isinstance(user_cfg[top_key], dict):
            cfg[top_key].update(user_cfg[top_key])
    return cfg


# --- useful functions


def _project_root() -> Path:
    # this file: .../src/themis/datasets/themis_datasets_2025.py
    # project root is three levels above this file: datasets -> themis -> src -> [project root]
    return Path(__file__).resolve().parents[3]


def _resolve_config_path(config_path_str: str) -> Optional[Path]:
    """Resolve a config path with project-root semantics.

    Policy:
    - If an absolute path is provided, return it if it exists; otherwise None.
    - If a relative path is provided, ALWAYS resolve it relative to the project root.
      If that doesn't exist and the path is a bare filename (no directory parts),
      try `project_root/configs/<filename>`.
    """
    p = Path(config_path_str)
    if p.is_absolute():
        return p if p.exists() else None

    root = _project_root()

    # Resolve relative to project root
    cand = (root / p)
    if cand.exists():
        return cand

    # If given a bare filename, try project_root/configs/<filename>
    if p.parent == Path('.'):
        cand2 = root / 'configs' / p.name
        if cand2.exists():
            return cand2

    return None


def _make_dataset_dict(line: str, sequence: int, flat_sequence: int, dark_sequence: int) -> dict:
    return {
        'line': line,
        file_types[0]: {
            'data_type': file_types[0],
            'sequence': sequence,
            'files': ''
        },
        file_types[2]: {
            'data_type': file_types[2],
            'sequence': flat_sequence,
            'files': ''
        },
        file_types[1]: {
            'data_type': file_types[1],
            'sequence': dark_sequence,
            'files': ''
        }
    }


# Register them
data_types = DataTypeRegistry()
data_types.add(dark)
data_types.add(flat)
data_types.add(scan)


# Lightweight FileSet implementation (kept local to datasets)
class FileSet:
    def __init__(self):
        self._files: Dict[str, Path] = {}

    def add(self, level_name: str, file_path: Path):
        self._files[level_name] = file_path

    def get(self, level_name: str, default=None):
        return self._files.get(level_name, default)

    def items(self):
        return self._files.items()

    def __repr__(self) -> str:
        lines = [f"<FileSet with {len(self._files)} entries>"]
        for level, path in self._files.items():
            exists = "\u2713" if path.exists() else "\u2717"
            lines.append(f"  {level}: {path.name} {exists}")
        return "\n".join(lines)


@dataclass
class Config:
    directories: DirectoryPaths
    dataset: dict
    cam: object
    data_types: DataTypeRegistry
    reduction_levels: tdr.ReductionRegistry
    slit_width: float
    polarization_states: list


def _build_file_set(directories: DirectoryPaths, dataset_entry: dict, data_types: DataTypeRegistry,
                    reduction_levels: tdr.ReductionRegistry, cam) -> FileSet:
    file_set = FileSet()
    data_t = dataset_entry['data_type']
    seq = dataset_entry['sequence']
    cam_str = cam.file_ext
    data_str = data_types[data_t].file_ext
    seq_str = f"t{seq:03d}"

    # Find the best matching file for each reduction level
    known_suffixes = [lvl.file_ext for lvl in reduction_levels.values() if lvl.file_ext]

    for level_name, level_obj in reduction_levels.items():
        suffix = level_obj.file_ext
        directory = getattr(directories, level_name, directories.raw)
        files = list(Path(directory).glob("*"))
        matches = []

        for f in files:
            name = f.name
            if cam_str in name and data_str in name and seq_str in name:
                if suffix == '':
                    if not any(suf in name for suf in known_suffixes):
                        matches.append(f)
                else:
                    if suffix in name:
                        matches.append(f)

        # Prefer files marked with _fx if available
        matches.sort(key=lambda x: (0 if '_fx' in x.name else 1, x.name))
        if matches:
            file_set.add(level_name, matches[0])

    return file_set


def get_config(auto_discover_files: bool = True,
               config_path: Optional[str] = None,
               auto_create_dirs: bool = False) -> Config:
    """
    Build and return a configuration object for the current dataset selection.

    Returns a lightweight dataclass holding directory paths, dataset metadata,
    camera configuration, reduction registry, and pre-discovered file paths per level.
    """
    # Load user configuration if provided, else use defaults
    user_cfg = None
    if config_path:
        resolved = _resolve_config_path(config_path)
        if not resolved:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        user_cfg = _load_config_from_toml(resolved)

    merged = _merge_defaults(user_cfg)

    # Unpack merged configuration
    line = merged['dataset']['line']
    date = merged['dataset']['date']
    sequence = int(merged['dataset']['sequence'])
    dark_sequence = int(merged['dataset']['dark_sequence'])
    flat_sequence = int(merged['dataset']['flat_sequence'])
    states = list(merged['dataset']['states'])

    slit_width = float(merged['params']['slit_width'])

    base = merged['paths'].get('base', DEFAULTS['paths']['base'])
    figures = merged['paths'].get('figures', None)
    inversion = merged['paths'].get('inversion', None)

    directories = DirectoryPaths(date=date, base=base, figures=figures,
                                 inversion=inversion, auto_create=auto_create_dirs)

    dataset = _make_dataset_dict(line=line, sequence=sequence,
                                 flat_sequence=flat_sequence, dark_sequence=dark_sequence)

    cfg = Config(
        directories=directories,
        dataset=dataset,
        cam=cc.cam[dataset['line']],
        data_types=data_types,
        reduction_levels=tdr.reduction_levels,
        slit_width=slit_width,
        polarization_states=states,
    )

    if auto_discover_files:
        for key in file_types:
            entry = cfg.dataset[key]
            entry['files'] = _build_file_set(cfg.directories, entry, cfg.data_types, cfg.reduction_levels, cfg.cam)

    return cfg
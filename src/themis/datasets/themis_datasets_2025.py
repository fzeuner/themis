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
from typing import Dict
from dataclasses import dataclass

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

# ++++++++++++++++++++++++++++++++++++++++++
# --- parameters for the loading
# ++++++++++++++++++++++++++++++++++++++++++

line = 'sr'
date = '2025-07-07'
sequence = 26
dark_sequence = 1
flat_sequence = 25
states = ['pQ', 'mQ', 'pU', 'mU', 'pV', 'mV' ]

# ++++++++++++++++++++++++++++++++++++++++++
# --- parameters to be changed only once  (in principle)
# ++++++++++++++++++++++++++++++++++++++++++

slit_width=0.33 #/ [arcsec] SlitWidth
file_types = ['scan', 'dark', 'flat']

# ++++++++++++++++++++++++++++++++++++++++++

# Create data types
dark = DataType(name=file_types[1], file_ext="_x3")
flat = DataType(name=file_types[2], file_ext="_y3")
scan = DataType(name=file_types[0], file_ext="_b3")

class DirectoryPaths:
    def __init__(self, date: str, base="/home/franziskaz/data/themis", auto_create=False):
        self.base = Path(base)
        self.date = date

        # Define paths
        self.raw = self.base / "rdata" / date
        self.reduced = self.base / "pdata" /date 
        self.figures = Path("/home/franziskaz/figures/themis/")
        self.inversion = self.base / "inversion"

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

directories = DirectoryPaths(date=date)


# --- useful functions


dataset = {
    'line': line,
    
    file_types[0]: {
        'data_type': file_types[0],
        'sequence': sequence,
        'files' : ''
    },
    file_types[2]: {
        'data_type': file_types[2],
        'sequence': flat_sequence,
        'files' : ''
    },
    file_types[1]: {
        'data_type': file_types[1],
        'sequence': dark_sequence,
        'files' : ''
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


def get_config(auto_discover_files: bool = True) -> Config:
    """
    Build and return a configuration object for the current dataset selection.

    Returns a lightweight dataclass holding directory paths, dataset metadata,
    camera configuration, reduction registry, and pre-discovered file paths per level.
    """
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
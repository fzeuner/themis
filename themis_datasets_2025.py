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
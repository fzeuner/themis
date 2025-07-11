#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 12:43:33 2025

@author: zeuner
"""

class ReductionLevel:
    def __init__(self, name, file_ext, func, per_type_meta=None):
        self.name = name
        self.file_ext = file_ext
        self.func = func  # Callable
        self.per_type_meta = per_type_meta or {}  # Optional: interpretation per data type

    def reduce(self, data):
        return self.func(data)

    def get_description(self, data_type=None):
        if data_type and data_type in self.per_type_meta:
            return self.per_type_meta[data_type]
        return f"{self.name} level"
    
    
class ReductionRegistry:
    def __init__(self):
        self._levels = {}

    def add(self, level):
        self._levels[level.name] = level

    def __getitem__(self, name):
        return self._levels[name]

    def __getattr__(self, name):
        if name in self._levels:
            return self._levels[name]
        raise AttributeError(f"No reduction level named '{name}'")

    def __iter__(self):
        return iter(self._levels)

    def items(self):
        return self._levels.items()

    def keys(self):
        return self._levels.keys()

    def values(self):
        return self._levels.values()

    def list_levels(self):
        return list(self._levels.keys())

    def __repr__(self):
        return f"<ReductionRegistry: {list(self._levels.keys())}>"



def reduce_raw(ff):
    print('No processing')
    return None

def reduce_raw_to_l0(ff):
    # No processing
    return None


reduction_levels = ReductionRegistry()

reduction_levels.add(ReductionLevel("raw", "fts", reduce_raw))
reduction_levels.add(ReductionLevel("l0", "_l0.fits", reduce_raw_to_l0, {
    "dark": "Averaging",
    "scan": "Apply dark, flatfield, cut upper and lower image, shift images",
    "flat": "Generate flatfield"
}))


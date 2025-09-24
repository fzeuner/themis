#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 12:43:33 2025

@author: zeuner
"""
from themis.core import themis_tools as tt
from themis.core import data_classes as dct

class ReductionLevel:
    def __init__(self, name, file_ext, func, per_type_meta=None):
        self.name = name
        self.file_ext = file_ext
        self.func = func  # Callable
        self.per_type_meta = per_type_meta or {}  # Optional: interpretation per data type

    def reduce(self, *args, **kwargs):
        """
        Forward all arguments to the underlying reduction function.
        This allows level-specific functions to define their own signatures
        (e.g., reduce_raw_to_l0(config, data_type=None, return_reduced=False)).
        """
        return self.func(*args, **kwargs)

    def get_description(self, data_type=None):
        if data_type and data_type in self.per_type_meta:
            return self.per_type_meta[data_type]
        return f"{self.name} level. For full description provide optional keyword data_type='data_type'"
    
    
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



def reduce_raw(config):
    print('No processing for reduction level raw')
    return None

def reduce_raw_to_l0(config, data_type=None, return_reduced=False):
    # No processing
    
    if data_type==None:
        print('No processing - provide a specific data type.')
        return None
    
    elif data_type == 'dark':
        data, header = tt.read_any_file(config, 'dark', verbose=False, status='raw')
        
        upper = data.stack_all('upper') # should always return one extra dimension that we can "average"
        lower = data.stack_all('lower') # should always return one extra dimension that we can "average"
        
        reduced_frames = dct.FramesSet()
        
        frame_name_str = f"{data_type}_l0_frame{0:04d}"
        single_frame = dct.Frame(frame_name_str)
        single_frame.set_half("upper", upper.mean(axis=0)) 
        single_frame.set_half("lower", lower.mean(axis=0))  
        reduced_frames.add_frame(single_frame, 0)
        
        frame_name_str = f"{data_type}_l0_frame{1:04d}"
        single_frame = dct.Frame(frame_name_str)
        single_frame.set_half("upper", tt.z3denoise(upper.mean(axis=0)) )
        single_frame.set_half("lower", tt.z3denoise(lower.mean(axis=0))   )
        reduced_frames.add_frame(single_frame, 1)
        
        if return_reduced:
            return reduced_frames
        
        else:
            print('saving not implemented yet')
            return None
    else:
        print('Unknown data_type.')
        return None


reduction_levels = ReductionRegistry()

reduction_levels.add(ReductionLevel("raw", "fts", reduce_raw))
reduction_levels.add(ReductionLevel("l0", "_l0.fits", reduce_raw_to_l0, {
    "dark": "Averaging raw, create a low-order polynomial",
    "scan": "Apply dark, cut upper and lower image, flatfield, shift images",
    "flat": "Generate flatfield"
}))


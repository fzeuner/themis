#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 15:49:17 2025

@author: zeuner

Data class handling
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FrameHalf:
    data: np.ndarray
    pol_state: str

class Frame:
    def __init__(self, name):
        self.name = name
        self.halves = {}  # 'upper' → FrameHalf, 'lower' → FrameHalf

    def set_half(self, position: str, data: np.ndarray, pol_state: str):
        assert position in ['upper', 'lower']
        self.halves[position] = FrameHalf(data=data, pol_state=pol_state)

    def get_half(self, position: str):
        return self.halves.get(position)

    def get_by_state(self, pol_state: str):
        for pos, half in self.halves.items():
            if half.pol_state == pol_state:
                return half.data
        return None  # or raise?
    

class CycleSet:
   def __init__(self):
       # frames keyed by (polarization_state, slit_position_index, map_index)
       self.frames = {} 

   def add_frame(self, frame: Frame, key_tuple: tuple):
       """
       Adds a frame with a composite key (pol_state, slit_pos_idx, map_idx).
       """
       self.frames[key_tuple] = frame

   def get_state_slit_map(self, pol_state: str, slit_pos_idx: int, map_idx: int):
       """
       Retrieves a specific frame by its polarization state, slit position index, and map index.
       """
       return self.frames.get((pol_state, slit_pos_idx, map_idx))

   def get_state(self, pol_state: str):
       """
       Returns a new CycleSet containing only frames for the specified polarization state.
       """
       new_collection = CycleSet()
       for key, frame in self.frames.items():
           if key[0] == pol_state:
               new_collection.add_frame(frame, key)
       return new_collection
   
   def get_arrays(self):
       return [half.data for f_key in sorted(self.frames.keys()) for half in self.frames[f_key].halves.values()]

   def __repr__(self):
       frame_keys_summary = sorted(list(self.frames.keys()))
       if len(frame_keys_summary) > 5:
           frame_keys_summary = frame_keys_summary[:2] + ['...'] + frame_keys_summary[-2:]
       return f"CycleSet(frames_keys={frame_keys_summary}, total_frames={len(self.frames)})"

    
    
# Create and populate frame
# frame0 = Frame("frame0")
# frame0.set_half("upper", arr1, "pQ")
# frame0.set_half("lower", arr2, "mQ")

# frame1 = Frame("frame1")
# frame1.set_half("upper", arr3, "pU")
# frame1.set_half("lower", arr4, "mU")

# # Store in collection
# data = CycleSet()
# data.add_frame(frame0)
# data.add_frame(frame1)

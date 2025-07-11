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
        self.frames = {}

    def add_frame(self, frame: Frame):
        self.frames[frame.name] = frame

    def get_state(self, pol_state: str):
        new_collection = CycleSet()
        for frame in self.frames.values():
            for pos, half in frame.halves.items():
                if half.pol_state == pol_state:
                    # Only include the half with this pol_state
                    new_frame = Frame(frame.name)
                    new_frame.set_half(pos, half.data, half.pol_state)
                    new_collection.add_frame(new_frame)
    
        return new_collection
    
    def get_arrays(self):
        return [half.data for f in self.frames.values() for half in f.halves.values()]
    
    
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

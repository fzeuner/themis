#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 15:49:17 2025

@author: zeuner

Data class handling
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FrameHalf:
    data: np.ndarray
    pol_state: Optional[str] = None

class Frame:
    def __init__(self, name):
        self.name = name
        self.halves = {}  # 'upper' → FrameHalf, 'lower' → FrameHalf

    def set_half(self, position: str, data: np.ndarray, pol_state: Optional[str] = None):
        assert position in ['upper', 'lower']
        self.halves[position] = FrameHalf(data=data, pol_state=pol_state)

    def get_half(self, position: str):
        return self.halves.get(position)

    def get_by_state(self, pol_state: str):
        for pos, half in self.halves.items():
            if half.pol_state == pol_state:
                return half.data
        return None  # not found

    # Convenience: dict-like access, e.g. frame['upper'] or frame['name']
    def __getitem__(self, key: str):
        if key == 'name':
            return self.name
        return self.halves[key]

    def keys(self):
        """Return available keys for mapping-style access (includes 'name' and halves)."""
        return ['name', *list(self.halves.keys())]

    def __repr__(self):
        """Compact summary of the frame content.

        Example:
            Frame(name='dark_l0_frame0000', upper: shape=(1024, 2048), lower: shape=(1024, 2048))
            Frame(name='scan_pQ...', upper: shape=(..), lower: shape=(..), notes=pol_states)
        """
        if not self.halves:
            return f"Frame(name='{self.name}', no halves)"

        parts = []
        for pos, half in sorted(self.halves.items()):
            shape = None if getattr(half, 'data', None) is None else tuple(half.data.shape)
            if getattr(half, 'pol_state', None):
                parts.append(f"{pos}: shape={shape}, pol_state='{half.pol_state}'")
            else:
                parts.append(f"{pos}: shape={shape}")
        return "Frame(name='" + self.name + "', " + ", ".join(parts) + ")"

class CycleSet:
    def __init__(self):
        # frames keyed by (frame_state, slit_position_index, map_index)
        self.frames = {}

    def add_frame(self, frame: Frame, key_tuple: tuple):
        """
        Adds a frame with a composite key (frame_state, slit_pos_idx, map_idx).
        """
        self.frames[key_tuple] = frame

    def get_state_slit_map(self, frame_state: str, slit_pos_idx: int, map_idx: int):
        """
        Retrieves a specific frame by its frame state, slit position index, and map index.
        """
        return self.frames.get((frame_state, slit_pos_idx, map_idx))

    def get_state(self, frame_state: str):
        """
        Returns a new CycleSet containing only frames for the specified frame state.
        """
        new_collection = CycleSet()
        for key, frame in self.frames.items():
            if key[0] == frame_state:
                new_collection.add_frame(frame, key)
        return new_collection
    
    def get_arrays(self):
        return [half.data for f_key in sorted(self.frames.keys()) for half in self.frames[f_key].halves.values()]

    def __repr__(self):
        frame_keys_summary = sorted(list(self.frames.keys()))
        if len(frame_keys_summary) > 5:
            frame_keys_summary = frame_keys_summary[:2] + ['...'] + frame_keys_summary[-2:]
        return f"CycleSet(frames_keys={frame_keys_summary}, total_frames={len(self.frames)})"

    # Convenience methods
    def get_all_halves(self, position: str = 'upper'):
        """
        Return a list of FrameHalf objects for the requested position over all frames
        in key-sorted order. Position must be 'upper' or 'lower'.
        """
        assert position in ('upper', 'lower')
        halves = []
        for _, frame in sorted(self.frames.items()):
            half = frame.get_half(position)
            if half is not None:
                halves.append(half)
        return halves

    def get_all(self, position: str = 'upper'):
        """
        Return a list of np.ndarray data for the requested half position over all frames
        in key-sorted order. Position must be 'upper' or 'lower'.
        """
        return [h.data for h in self.get_all_halves(position)]

    def stack_all(self, position: str = 'upper', axis: int = 0):
        """
        Stack all arrays of the requested half position along a new axis.
        All arrays must have matching shapes. Raises if list is empty.
        """
        arrays = self.get_all(position)
        if not arrays:
            raise ValueError("No frames available to stack for position='{}'".format(position))
        return np.stack(arrays, axis=axis)
    
    def get_data(self, frame_state: str, slit_pos_idx: int, map_idx: int, pol_state: str):
        """
        Directly retrieve data array for a specific frame half by its polarization state.
        
        Args:
            frame_state: Frame state identifying the frame
            slit_pos_idx: Slit position index
            map_idx: Map index
            pol_state: Polarization state of the half to retrieve
            
        Returns:
            np.ndarray or None if not found
        """
        frame = self.get_state_slit_map(frame_state, slit_pos_idx, map_idx)
        if frame is None:
            return None
        return frame.get_by_state(pol_state)
    
    def get_both_halves(self, frame_state: str, slit_pos_idx: int, map_idx: int):
        """
        Retrieve both upper and lower halves for a specific frame.
        
        Args:
            frame_state: Frame state identifying the frame
            slit_pos_idx: Slit position index
            map_idx: Map index
            
        Returns:
            dict with 'upper' and 'lower' keys containing FrameHalf objects, or None if frame not found
        """
        frame = self.get_state_slit_map(frame_state, slit_pos_idx, map_idx)
        if frame is None:
            return None
        return {
            'upper': frame.get_half('upper'),
            'lower': frame.get_half('lower')
        }
    
    def get_both_data(self, frame_state: str, slit_pos_idx: int, map_idx: int):
        """
        Retrieve both upper and lower data arrays for a specific frame.
        
        Args:
            frame_state: Frame state identifying the frame
            slit_pos_idx: Slit position index
            map_idx: Map index
            
        Returns:
            dict with 'upper' and 'lower' keys containing np.ndarray, or None if frame not found
        """
        halves = self.get_both_halves(frame_state, slit_pos_idx, map_idx)
        if halves is None:
            return None
        return {
            'upper': halves['upper'].data if halves['upper'] else None,
            'lower': halves['lower'].data if halves['lower'] else None
        }
    
    def items(self):
        """Dict-like iteration over (key, frame) pairs."""
        return self.frames.items()
    
    def keys(self):
        """Dict-like access to frame keys."""
        return self.frames.keys()
    
    def __getitem__(self, key):
        """Dict-like access: cycleset[(frame_state, slit_idx, map_idx)]."""
        return self.frames[key]

class FramesSet:
    """
    Simple container for a sequence of Frame instances keyed by an integer frame index.
    Use this when you just have a list/sequence of frames without the (state, slit, map) semantics.
    """
    def __init__(self):
        self.frames = {}

    def add_frame(self, frame: Frame, frame_idx: int):
        self.frames[int(frame_idx)] = frame

    def get(self, frame_idx: int):
        return self.frames.get(int(frame_idx))

    def __len__(self):
        return len(self.frames)

    # Convenience: index-like access, e.g. frames_set[0]
    def __getitem__(self, frame_idx: int) -> Frame:
        return self.frames[int(frame_idx)]

    def keys(self):
        return sorted(self.frames.keys())

    def items(self):
        for k in self.keys():
            yield k, self.frames[k]

    def get_arrays(self):
        return [half.data for f_key in self.keys() for half in self.frames[f_key].halves.values()]

    def __repr__(self):
        frame_keys_summary = self.keys()
        if len(frame_keys_summary) > 5:
            frame_keys_summary = frame_keys_summary[:2] + ['...'] + frame_keys_summary[-2:]
        return f"FramesSet(frame_indices={frame_keys_summary}, total_frames={len(self.frames)})"

    # Convenience methods
    def get_all_halves(self, position: str = 'upper'):
        """
        Return a list of FrameHalf objects for the requested position over all frames
        in index-sorted order. Position must be 'upper' or 'lower'.
        """
        assert position in ('upper', 'lower')
        halves = []
        for idx in self.keys():
            frame = self.frames[idx]
            half = frame.get_half(position)
            if half is not None:
                halves.append(half)
        return halves

    def get_all(self, position: str = 'upper'):
        """
        Return a list of np.ndarray data for the requested half position over all frames
        in index-sorted order. Position must be 'upper' or 'lower'.
        """
        return [h.data for h in self.get_all_halves(position)]

    def stack_all(self, position: str = 'upper', axis: int = 0):
        """
        Stack all arrays of the requested half position along a new axis.
        All arrays must have matching shapes. Raises if list is empty.
        """
        arrays = self.get_all(position)
        if not arrays:
            raise ValueError("No frames available to stack for position='{}'".format(position))
        return np.stack(arrays, axis=axis)

    
    
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

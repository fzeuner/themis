#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 09:03:29 2025

@author: zeuner

Camera configuration file for MTR 2 spectropolarimeter on THEMIS for one observing run/configuration
"""

class ROI:
    def __init__(self, lower, upper):
        """
        lower and upper are tuples like: ((row_start, row_end), (col_start, col_end))
        """
        self.upper = upper
        self.lower = lower

    def extract(self, array):
        """
        Extracts two ROIs from an N-dimensional array (N >= 2).
        The ROIs are applied to the last two dimensions (rows, cols).
        Returns a tuple of two arrays with the same leading dimensions.
        """
        *leading_dims, rows, cols = array.shape

        # Build slices for each region
        def make_slicer(region):
            row_slice = slice(region[0][0], region[0][1])
            col_slice = slice(region[1][0], region[1][1])
            return (..., row_slice, col_slice)

        r1 = array[make_slicer(self.lower)]
        r2 = array[make_slicer(self.upper)]
        return r1, r2

    def __repr__(self):
        return f"ROI(upper={self.lower}, lower={self.upper})"

class Camera:
    def __init__(self, name='none', target='none', pixel_scale=1, file_ext='', wavelength='0', 
                 roi=None, **kwargs):
        self.name = name
        self.target = target
        self.pixel_scale = pixel_scale
        self.file_ext = file_ext
        self.wavelength = wavelength
        self.roi = roi  # roi of lower and upper image 
        self.properties = kwargs  # Additional optional properties
        
        # Use default ROI if none provided
        if roi is None:
            self.roi = ROI(lower=((0, -1), (0, -1)), upper=((0, -1), (0, -1)))
        elif isinstance(roi, ROI):
            self.roi = roi
        else:
            # Assume it's in the list/tuple format: [[[row1],[col1]], [[row2],[col2]]]
            r1, r2 = roi
            self.roi = ROI(lower=tuple(map(tuple, r1)), upper=tuple(map(tuple, r2)))

    def __repr__(self):
        return f"<Camera(name={self.name}, target={self.target})>"

class CameraRegistry:
    def __init__(self):
        self._cameras_by_target = {}

    def add_camera(self, camera):
        self._cameras_by_target[camera.target] = camera

    def __getitem__(self, target):
        return self._cameras_by_target[target]

    def __getattr__(self, target):
        if target in self._cameras_by_target:
            return self._cameras_by_target[target]
        raise AttributeError(f"No camera found for target '{target}'")

    def list_targets(self):
        return list(self._cameras_by_target.keys())
    
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


cam1 = Camera(name="Zyla 5", target="sr", pixel_scale=0.3, file_ext='b0505', wavelength='4607', roi=[[[0,100],[0,100]],[[0,100],[0,100]]])

roi = ROI(lower=((50, 877), (250, 1750)), upper=((1122, 1987), (250, 1750)))
cam2 = Camera(name="Zyla 6", target="ti", pixel_scale=0.3, file_ext='b0606', wavelength='4536', roi=roi)
cam3 = Camera(name="iXon 2", target="fe", pixel_scale=0.3, file_ext='b0202', wavelength='6302', roi=[[[0,100],[0,100]],[[0,100],[0,100]]])

# Register them
cam = CameraRegistry()
cam.add_camera(cam1)
cam.add_camera(cam2)
cam.add_camera(cam3)

# Access by attribute
#print(cam.target1.pixel_scale)  
# Access by key
#print(cam['target2'].pixel_scale) 
# List available targets
#print(cam.list_targets())  # ['target1', 'target2', 'target3']

# Create data types
dark = DataType(name="dark", file_ext="_x3")
flat = DataType(name="flat", file_ext="_y3")
scan = DataType(name="scan", file_ext="_b3")

# Register them
data_type = DataTypeRegistry()
data_type.add(dark)
data_type.add(flat)
data_type.add(scan)
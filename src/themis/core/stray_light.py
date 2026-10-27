#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global spectrograph stray-light removal.

Follows the atlas-fit convention (atlas-fit/src/atlas_fit/models/spectrum.py,
``Spectrum.apply_offset`` and ``fitting/parameter_fit.py``,
``ParameterFit._apply_stray_light``): a stray-light level ``s`` given in
percent is removed by subtracting a constant offset proportional to the mean
intensity,

    I_corr = I - mean(I) * s / 100

atlas-fit applies this to the 1D spectrum extracted with ``input.roi`` (e.g.
"[400,100:-100]"), so the mean is taken over the ROI's spectral range only.
Here the correction is applied per frame half (upper/lower) with a single
global offset: the mean is taken over all spatial rows and the spectral
columns of the atlas-fit ROI. Restricting the spectral range matters: the edge
columns of the L4 frames can contain huge values (desmiling artifacts) that
would otherwise dominate the mean.

The stray-light levels are read from the camera's atlas-fit config:
``input.stray_light`` (upper frame) and ``input.stray_light_lower`` (lower
frame), both in percent.
"""

import copy

import numpy as np
import yaml


def _load_atlas_fit_input(config) -> dict:
    with open(config.cam.atlas_fit_config, 'r') as f:
        return yaml.safe_load(f)['input']


def _parse_slice(text: str) -> slice:
    """Parse a numpy-style slice string, e.g. '100:-100' or '400'."""
    parts = [int(p) if p.strip() else None for p in text.split(':')]
    return slice(*parts) if len(parts) > 1 else slice(parts[0], parts[0] + 1)


def get_stray_light_levels(config) -> dict:
    """
    Read the stray-light levels [%] for the upper and lower frame and the
    spectral range used for the mean from the camera's atlas-fit config.

    Returns
    -------
    dict
        {'upper': float, 'lower': float, 'spectral_roi': slice}; levels in
        percent, spectral_roi taken from the spectral part of input.roi.
    """
    inp = _load_atlas_fit_input(config)
    roi = inp.get('roi', '[:, :]').strip().strip('[]').split(',')
    return {'upper': float(inp['stray_light']),
            'lower': float(inp['stray_light_lower']),
            'spectral_roi': _parse_slice(roi[-1])}


def remove_stray_light(frame: np.ndarray, percent: float,
                       spectral_roi: slice = slice(None)) -> np.ndarray:
    """
    Remove a global stray-light offset from a 2D frame half.

    Parameters
    ----------
    frame : ndarray
        Frame half (spatial, spectral).
    percent : float
        Stray-light level in percent of the mean frame intensity.
    spectral_roi : slice
        Spectral columns used to compute the mean (all rows are used).

    Returns
    -------
    ndarray
        frame - nanmean(frame[:, spectral_roi]) * percent / 100
        (float dtype of `frame` kept).
    """
    offset = np.nanmean(frame[:, spectral_roi], dtype='float64') * percent / 100
    return (frame - offset).astype(np.result_type(frame.dtype, np.float32), copy=False)


def remove_stray_light_from_set(data_set, levels: dict, inplace: bool = False):
    """
    Apply `remove_stray_light` to every frame half of a CycleSet or FramesSet.

    Parameters
    ----------
    data_set : CycleSet or FramesSet
    levels : dict
        {'upper': percent, 'lower': percent[, 'spectral_roi': slice]}, e.g.
        from `get_stray_light_levels`.
    inplace : bool
        If False (default), a deep copy is corrected and returned.

    Returns
    -------
    CycleSet or FramesSet with stray-light corrected frame halves.
    """
    spectral_roi = levels.get('spectral_roi', slice(None))
    out = data_set if inplace else copy.deepcopy(data_set)
    for _, frame in out.items():
        for half_name, half in frame.halves.items():
            half.data = remove_stray_light(half.data, levels[half_name], spectral_roi)
    return out

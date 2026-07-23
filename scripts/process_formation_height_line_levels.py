#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral Line Fitting and Formation Height Analysis

This script implements Voigt profile fitting for Ti and Sr spectral lines to extract
formation height parameters. It follows the methodology from run_voigt_as.pro (IDL)
with Python implementations using lmfit.

Workflow Steps:
1. Data Preparation: Load L4 data, split spectrum into continuum/line/residual regions
2. Continuum Normalization: Normalize each position to its continuum level
3. Spectral Fitting: Fit each spatial pixel with piecewise (red and blue) Voigt models
   - Ti line: 4 Voigt profiles (A, B, C, D) + constant continuum
   - Sr line: 2 Voigt profiles (SR, B) + constant continuum
4. Line Levels Calculation: Extract intensity levels at equally-spaced widths
5. Results Storage: Save fitted profiles and line levels in FITS format

Special Implementation Details:

- Per-pixel Continuum: Each pixel uses its own continuum level calculated from
  the continuum region, using central 80% of continuum data (removing top/bottom 10%
  outliers) to reduce sensitivity to bad pixels or cosmic rays.

- Continuum Sanity Check: In line_levels calculation, ensures 95% continuum level
  is below the maximum of fitted data. If too high, adjusts to 99% of max.

- Parabola Refinement: Component centers are initialized using parabola fitting
  around the minimum for improved starting parameters. 
  --> Center is fitted, and the assumed to be the same for red/blue and splitting 

- Weights around the center are higher to prevent disconuities

- Shared Memory Multiprocessing: For large datasets (>100 pixels), uses shared memory
  to avoid data copying overhead. Data is cast to float64 to match worker expectations.

- Minimizer strategy slightly optimized (max number of iterations)

- Wavelength Units: All internal calculations use Angstrom. Data is converted to nm
  only when saving to FITS files for compatibility with IDL/other tools.

- Line Levels Storage: Line levels are saved as FITS binary table extension with
  metadata (continuum level, intensity at core, width at continuum) following
  run_voigt_as.pro format.

- Error Handling: Line levels calculation can fail for bad data pixels; these are
  gracefully handled by returning None rather than crashing the entire fit.

Example usage:
    spectra = SpectrumContainer.load_all()
    spectrum = spectra['ti']['disk_center']
    spectrum.fit()  # Fit all pixels
    spectrum.save()
    spectrum.plot(fit=True, show_levels=True, slit=100, scan=10)

@author: zeuner
Created on Fri Jul 4 15:24:27 2025
"""

from themis.core import themis_tools as tt
from themis.core import themis_io as tio
from themis.datasets.themis_datasets_2025 import get_config

from spectator.controllers.app_controller import display_data # from spectator

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from pathlib import Path
from datetime import datetime
from astropy.io import fits
from lmfit.models import VoigtModel, ConstantModel
from lmfit import Model
from lmfit.lineshapes import voigt as voigt_lineshape

#----------------------------------------------------------------------
"""
MAIN: 
   
"""

# Configuration file path
CONFIG_PATH = 'configs/formation_dataset_2025-07-05.toml'

# Configuration for formation height analysis: maps disk positions to sequences
# Based on comments in formation_dataset_2025-07-05.toml
# m := minus (e.g., m40 = -40)
# Both TI and SR lines share the same positions and sequences
POSITION_TO_SEQUENCE = {
    'disk_center': 27,
    'm40': 24,
    'm30': 18,
    '30': 50,
    '40': 36,
    '45': 38,
    'm45': 48,
    'm55': 92,
    '55': 86,
}

# Available spectral lines
AVAILABLE_LINES = ['ti', 'sr']

# Helper: convert position name to numeric value (for plotting labels, etc.)
POSITION_TO_NUMERIC = {
    'disk_center': 0,
    'm40': -40,
    'm30': -30,
    '30': 30,
    '40': 40,
    '45': 45,
    'm45': -45,
    'm55': -55,
    '55': 55,
}

# Predefined crop configurations for each line
CROP_CONFIGS = {
    'sr': {'scan': (0, -1), 'slit': (2, -55), 'wavelength': (276, 886)},
    'ti': {'scan': (0, -1), 'slit': (2, -55), 'wavelength': (406, 1016)},
}



# Helper function for parabola-based center initialization
def refine_center_with_parabola(wvl, profile, center_guess, search_width=0.07):
    """
    Refine a component center using parabola fitting.

    Args:
        wvl: Wavelength array
        profile: Intensity profile
        center_guess: Initial guess for the center (in Angstrom)
        search_width: Search range around the guess (in Angstrom)

    Returns:
        Refined center position, or center_guess if refinement fails
    """
    idx_left = np.argmin(abs(wvl - (center_guess - search_width)))
    idx_right = np.argmin(abs(wvl - (center_guess + search_width)))
    if idx_left < idx_right:
        idx_center_approx = np.argmin(profile[idx_left:idx_right])
        parabola_half_width = 5
        idx_parabola_left = max(idx_left, idx_left + idx_center_approx - parabola_half_width)
        idx_parabola_right = min(idx_right, idx_left + idx_center_approx + parabola_half_width)
        if idx_parabola_right > idx_parabola_left:
            wvl_parabola = wvl[idx_parabola_left:idx_parabola_right]
            profile_parabola = profile[idx_parabola_left:idx_parabola_right]
            poly_coefs = np.polyfit(wvl_parabola, profile_parabola, 2)
            a, b, c = poly_coefs
            if a > 0:
                return -b / (2 * a)
    return center_guess


# Fitting parameter initialization
def initialize_piecewise_parameter(params, continuum, wvl=None, profile=None):

    params['ti_r_amplitude'].value= -0.08
    params['ti_r_amplitude'].min= -0.1
    params['ti_r_amplitude'].max= -0.03

    params['ti_b_amplitude'].value= -0.08
    params['ti_b_amplitude'].min= -0.1
    params['ti_b_amplitude'].max= -0.03

    # Initialize Ti center with parabola fit if wvl and profile provided
    ti_center_guess = 4536.39  # in A
    if wvl is not None and profile is not None:
        ti_center_refined = refine_center_with_parabola(wvl, profile, ti_center_guess)
        params['ti_r_center'].value = ti_center_refined

    else:
        params['ti_r_center'].value = ti_center_guess
 
    params['ti_r_center'].min =  params['ti_r_center'].value - 0.005 # in A
    params['ti_r_center'].max =  params['ti_r_center'].value + 0.005 # in A
    params['ti_r_center'].vary = True
    params['ti_b_center'].expr = 'ti_r_center'

    params['ti_r_sigma'].value= 0.03
    params['ti_b_sigma'].value= 0.03
    params['ti_r_sigma'].max= 0.05
    params['ti_b_sigma'].max= 0.05
    params['ti_r_sigma'].min= 0.01
    params['ti_b_sigma'].min= 0.01

    params['ti_b_gamma'].value= 0.03
    params['ti_r_gamma'].value= 0.03
    params['ti_r_gamma'].min = 0.001
    params['ti_r_gamma'].max = 0.09
    params['ti_b_gamma'].min = 0.001
    params['ti_b_gamma'].max = 0.09
    # Initialize y to the center for each component
    params['ti_y'].expr = 'ti_r_center'


    params['B_r_amplitude'].value= -0.09
    params['B_b_amplitude'].value= -0.09
    params['B_r_amplitude'].max= -0.03
    params['B_r_amplitude'].min= -0.1
    params['B_b_amplitude'].max= -0.03
    params['B_b_amplitude'].min= -0.1

    # Initialize B center with parabola fit if wvl and profile provided
    if wvl is not None and profile is not None:
        params['B_r_center'].value = refine_center_with_parabola(wvl, profile, 4536.268)

    else:
        params['B_r_center'].value = 4536.268 # in A

    params['B_r_center'].min = params['B_r_center'].value - 0.005 # in A
    params['B_r_center'].max = params['B_r_center'].value + 0.005 # in A
    params['B_r_center'].vary = True
    params['B_b_center'].expr = 'B_r_center'

    params['B_r_sigma'].value= 0.03
    params['B_b_sigma'].value= 0.03
    params['B_r_sigma'].max= 0.05
    params['B_b_sigma'].max= 0.05
    params['B_r_sigma'].min= 0.01
    params['B_b_sigma'].min= 0.01

    params['B_r_gamma'].value= 0.03
    params['B_b_gamma'].value= 0.03
    params['B_r_gamma'].min = 0.001
    params['B_r_gamma'].max = 0.09
    params['B_b_gamma'].min = 0.001
    params['B_b_gamma'].max = 0.09

    params['B_y'].expr = 'B_r_center'

    params['C_r_amplitude'].value= -0.09
    params['C_b_amplitude'].value= -0.09

    # Initialize C center with parabola fit if wvl and profile provided
    if wvl is not None and profile is not None:
        params['C_r_center'].value = refine_center_with_parabola(wvl, profile, 4536.050)
    else:
        params['C_r_center'].value = 4536.050

    params['C_r_center'].min = params['C_r_center'].value - 0.005 # in A
    params['C_r_center'].max = params['C_r_center'].value + 0.005 # in A
    params['C_r_center'].vary = True
    params['C_b_center'].expr = 'C_r_center'

    params['C_r_sigma'].value= 0.03
    params['C_b_sigma'].value= 0.03
    params['C_r_sigma'].max= 0.05
    params['C_b_sigma'].max= 0.05
    params['C_r_sigma'].min= 0.01
    params['C_b_sigma'].min= 0.01

    params['C_r_gamma'].value= 0.03
    params['C_b_gamma'].value= 0.03
    params['C_r_gamma'].min = 0.001
    params['C_r_gamma'].max = 0.09
    params['C_b_gamma'].min = 0.001
    params['C_b_gamma'].max = 0.09
    params['C_y'].expr = 'C_r_center'

    params['D_r_amplitude'].value= -0.09
    params['D_b_amplitude'].value= -0.09

    # Initialize D center with parabola fit if wvl and profile provided
    if wvl is not None and profile is not None:
        params['D_r_center'].value = refine_center_with_parabola(wvl, profile, 4535.9)
    else:
        params['D_r_center'].value = 4535.9 # in A

    params['D_r_center'].min = params['D_r_center'].value - 0.005 # in A
    params['D_r_center'].max = params['D_r_center'].value + 0.005 # in A
    params['D_r_center'].vary = True
    params['D_b_center'].expr = 'D_r_center'

    params['D_r_sigma'].value= 0.03
    params['D_b_sigma'].value= 0.03
    params['D_r_sigma'].max= 0.05
    params['D_b_sigma'].max= 0.05
    params['D_r_sigma'].min= 0.01
    params['D_b_sigma'].min= 0.01

    params['D_r_gamma'].value= 0.03
    params['D_b_gamma'].value= 0.03
    params['D_r_gamma'].min = 0.001
    params['D_r_gamma'].max = 0.09
    params['D_b_gamma'].min = 0.001
    params['D_b_gamma'].max = 0.09
    params['D_y'].expr = 'D_r_center'

    params['c_c'].value= continuum
    params['c_c'].vary = False
    params['c_c'].min = continuum*0.9
    params['c_c'].max = continuum*1.1

    return(params)


def _piecewise_voigt_func(x, y, r_amplitude, r_center, r_sigma, r_gamma,
                           b_amplitude, b_center, b_sigma, b_gamma,
                           transition_width=0.005):
    # Two independent Voigt profiles, selected pointwise by wavelength x
    # relative to the split point y (blue side: x < y, red side: x >= y)
    voigt_r = voigt_lineshape(x, r_amplitude, r_center, r_sigma, r_gamma)
    voigt_b = voigt_lineshape(x, b_amplitude, b_center, b_sigma, b_gamma)
    return np.where(x < y, voigt_b, voigt_r)

def piecewise_voigt(prefix='line'): # fit y, the point where the split should appear
    model = Model(_piecewise_voigt_func, independent_vars=['x'], prefix=prefix+'_',
                  param_names=['y', 'r_amplitude', 'r_center', 'r_sigma', 'r_gamma',
                               'b_amplitude', 'b_center', 'b_sigma', 'b_gamma'],
                  )
    return model


def piecewise_model():

    voigt_ti = piecewise_voigt(prefix='ti')  # Ti I
    pars = voigt_ti.make_params()
    voigtB = piecewise_voigt(prefix='B')
    pars.update(voigtB.make_params())
    voigtC = piecewise_voigt(prefix='C')
    pars.update(voigtC.make_params())
    voigtD = piecewise_voigt(prefix='D')
    pars.update(voigtD.make_params())

    const = ConstantModel(prefix='c_')
    pars.update(const.make_params())

    model = voigt_ti + voigtB + voigtC + voigtD + const

    return model, pars

def add_overlap_weights(wvl, params, prefixes=None, base_weight=1.0, boost=3, boost_width=0.008): # keep boost_width small
    """Build a weights array that upweights the region around each component's
    split point y, where blue/red wings and neighboring components overlap
    the most. Use as `weights=` in model.fit().

    Args:
        wvl: Wavelength array
        params: Model parameters
        prefixes: List of component prefixes to weight (e.g., ['ti', 'B', 'C', 'D'] or ['sr', 'B'])
                 If None, defaults to ['ti', 'B', 'C', 'D'] for TI line
        base_weight: Base weight for all wavelengths
        boost: Weight multiplier for overlap regions
        boost_width: Width around each split point to boost (in Angstrom)
    """
    if prefixes is None:
        prefixes = ['ti', 'B', 'C', 'D']

    weights = np.full_like(wvl, base_weight, dtype=float)
    for prefix in prefixes:
        y_split = params[prefix+'_y'].value
        mask = np.abs(wvl - y_split) <= boost_width
        weights[mask] = np.maximum(weights[mask], boost)
    return weights

# Helper function for multiprocessing with shared memory (must be at module level)
def _fit_single_pixel_shared(args):
    """Fit a single pixel for multiprocessing using shared memory."""
    wvl_shm_name, line_data_shm_name, continuum_data_shm_name, wvl_shape, line_data_shape, continuum_data_shape, slit_idx, scan_idx = args
    
    try:
        # Access shared memory
        from multiprocessing import shared_memory
        
        # Connect to shared memory blocks
        wvl_shm = shared_memory.SharedMemory(name=wvl_shm_name)
        line_data_shm = shared_memory.SharedMemory(name=line_data_shm_name)
        continuum_data_shm = shared_memory.SharedMemory(name=continuum_data_shm_name)
        
        # Create numpy arrays from shared memory
        wvl = np.ndarray(wvl_shape, dtype=np.float64, buffer=wvl_shm.buf)
        line_data = np.ndarray(line_data_shape, dtype=np.float64, buffer=line_data_shm.buf)
        continuum_data = np.ndarray(continuum_data_shape, dtype=np.float64, buffer=continuum_data_shm.buf)
        
        # Extract profile and continuum for this pixel
        profile = line_data[:, slit_idx, scan_idx].copy()  # Copy to avoid keeping shared memory reference
        continuum_pixel = robust_continuum_mean(continuum_data[:, slit_idx, scan_idx])
        wvl_copy = wvl.copy()  # Copy wavelength
        
        # Close shared memory access
        wvl_shm.close()
        line_data_shm.close()
        continuum_data_shm.close()
        
        # Fit the pixel
        result = fit_ti_pixel(wvl_copy, profile, continuum_pixel)
        return (slit_idx, scan_idx, result, None)
    except Exception as e:
        return (slit_idx, scan_idx, None, str(e))

# Helper function for multiprocessing (must be at module level)
def _fit_single_pixel(args):
    """Fit a single pixel for multiprocessing."""
    wvl, profile, continuum, slit_idx, scan_idx = args
    try:
        result = fit_ti_pixel(wvl, profile, continuum)
        return (slit_idx, scan_idx, result, None)
    except Exception as e:
        return (slit_idx, scan_idx, None, str(e))


def fit_ti_pixel(wvl, profile, continuum):
    """
    Fit a single pixel spectrum using the piecewise Voigt model.

    Args:
        wvl: Wavelength array
        profile: Intensity profile (line + residual)
        continuum: Continuum level (fixed). If None or not finite, uses profile[-1]

    Returns:
        dict: Dictionary containing:
            - 'best_fit': Best fit profile
            - 'component_ti': Ti component
            - 'component_residual': Residual component (B + C + D)
            - 'params': Fitted parameters
    """
    # Continuum fallback logic
    if continuum is None or not np.isfinite(continuum):
        continuum = profile[-1]

    # Initialize model and parameters
    model, pars = piecewise_model()
    pars = initialize_piecewise_parameter(pars, continuum, wvl=wvl, profile=profile)

    # Add weights
    weights = add_overlap_weights(wvl, pars)

    # Fit
    out = model.fit(profile, pars, x=wvl, weights=weights)

    # Evaluate best fit and components
    best_fit = model.eval(out.params, x=wvl)
    components = model.eval_components(params=out.params, x=wvl)

    # Extract components
    component_ti = components['ti_'] + components['c_']
    component_residual = components['B_'] + components['C_'] + components['D_'] + components['c_']

    # Calculate line levels (may fail for bad data)
    line_center = out.params['ti_r_center'].value
    try:
        levels = line_levels(wvl, component_ti, line_center, continuum)
    except Exception:
        levels = None

    return {
        'best_fit': best_fit,
        'component_ti': component_ti,
        'component_residual': component_residual,
        'params': out.params,
        'levels': levels
    }

def robust_continuum_mean(continuum_data):
    """
    Calculate robust continuum mean using 80% of data (removing outliers).
    
    Takes the central 80% of the data after sorting, which effectively removes
    the top and bottom 10% outliers.
    
    Args:
        continuum_data: Continuum data array (1D or can be sliced to 1D)
        
    Returns:
        float: Robust mean of the continuum data
    """
    continuum_flat = continuum_data.flatten()
    continuum_sorted = np.sort(continuum_flat)
    # Take central 80% (remove top and bottom 10%)
    n = len(continuum_sorted)
    start_idx = int(n * 0.1)
    end_idx = int(n * 0.9)
    continuum_central = continuum_sorted[start_idx:end_idx]
    return np.mean(continuum_central)

def line_levels(wvl, line_profile, line_center, continuum, n_levels=25):
    """
    Calculate line levels (intensity at different widths) following run_voigt_as.pro logic.

    Splits the fitted profile into blue and red wings at the line center,
    treats each wing independently (asymmetric), and for each target width
    finds the intensity at which the combined width (blue + red) equals the
    target, using root-finding — matching the IDL niv() approach.

    Args:
        wvl: Wavelength array (in Angstrom)
        line_profile: Fitted line component profile (e.g., component_ti or component_sr)
        line_center: Line center wavelength from fit parameters (in Angstrom)
        continuum: Continuum level for this pixel
        n_levels: Number of levels to calculate (default: 25)

    Returns:
        dict: Dictionary containing:
            - 'widths': Array of widths from line center (in Angstrom)
            - 'intensities': Array of intensities at those widths
            - 'continuum_level': The continuum level used (95% of continuum)
            - 'intensity_core': Intensity evaluated exactly at line_center
            - 'width_continuum': Total width at continuum level (blue + red)
            Returns None if calculation fails
    """
    from scipy import interpolate
    from scipy.optimize import brentq

    try:
        # Use 95% of continuum as reference level
        continuum_level = 0.95 * continuum

        # Sanity check: ensure 95% continuum is below the maximum of the fitted data
        if continuum_level >= np.max(line_profile):
            continuum_level = np.max(line_profile) * 0.99

        # Evaluate core intensity exactly at line_center via interpolation
        interp_full = interpolate.interp1d(wvl, line_profile, kind='linear',
                                           fill_value='extrapolate')
        intensity_core = float(interp_full(line_center))

        # Split profile into blue and red wings at line_center
        blue_mask = wvl < line_center
        red_mask = wvl >= line_center

        if np.sum(blue_mask) < 2 or np.sum(red_mask) < 2:
            return None

        wvl_blue = wvl[blue_mask]
        prof_blue = line_profile[blue_mask]
        wvl_red = wvl[red_mask]
        prof_red = line_profile[red_mask]

        # Distance from center for each wing (positive)
        dist_blue = line_center - wvl_blue
        dist_red = wvl_red - line_center

        # Sort by distance (closest to center first)
        blue_sort = np.argsort(dist_blue)
        red_sort = np.argsort(dist_red)
        dist_blue = dist_blue[blue_sort]
        prof_blue = prof_blue[blue_sort]
        dist_red = dist_red[red_sort]
        prof_red = prof_red[red_sort]

        # Prepend the core point (distance=0, intensity=intensity_core) to each wing
        # so that inv_interp(core_intensity) = 0 for both wings
        dist_blue = np.concatenate([[0.0], dist_blue])
        prof_blue = np.concatenate([[intensity_core], prof_blue])
        dist_red = np.concatenate([[0.0], dist_red])
        prof_red = np.concatenate([[intensity_core], prof_red])

        # Enforce monotonic increase of intensity with distance (remove non-monotonic points)
        # This ensures the inverse interpolation is well-defined
        def _make_monotonic(dist, prof):
            sort_idx = np.argsort(prof)
            prof_s = prof[sort_idx]
            dist_s = dist[sort_idx]
            unique_mask = np.concatenate([[True], np.diff(prof_s) > 1e-12])
            return dist_s[unique_mask], prof_s[unique_mask]

        dist_blue_m, prof_blue_m = _make_monotonic(dist_blue, prof_blue)
        dist_red_m, prof_red_m = _make_monotonic(dist_red, prof_red)

        if len(prof_blue_m) < 2 or len(prof_red_m) < 2:
            return None

        # Create inverse interpolation: distance as function of intensity
        inv_interp_blue = interpolate.interp1d(prof_blue_m, dist_blue_m, kind='linear',
                                               fill_value=(0.0, dist_blue_m[-1]),
                                               bounds_error=False)
        inv_interp_red = interpolate.interp1d(prof_red_m, dist_red_m, kind='linear',
                                              fill_value=(0.0, dist_red_m[-1]),
                                              bounds_error=False)

        # Width at continuum level for each wing (independently)
        blue_width_cont = float(inv_interp_blue(continuum_level))
        red_width_cont = float(inv_interp_red(continuum_level))
        width_continuum = blue_width_cont + red_width_cont

        if width_continuum <= 0 or not np.isfinite(width_continuum):
            return None

        # N equally-spaced widths from 0 to width_continuum
        widths = np.linspace(0, width_continuum, n_levels)

        # For each width, find intensity I such that
        #   inv_interp_blue(I) + inv_interp_red(I) = target_width
        # This matches IDL's niv() which uses zbrent_tc on func2.
        intensities = np.full(n_levels, np.nan)
        intensities[0] = intensity_core

        Imin = min(intensity_core, prof_blue_m[0], prof_red_m[0])
        Imax = continuum_level

        for i in range(1, n_levels):
            target_width = widths[i]

            def combined_width(I, tw=target_width):
                return float(inv_interp_blue(I)) + float(inv_interp_red(I)) - tw

            try:
                val = brentq(combined_width, Imin, Imax, xtol=1e-10)
                intensities[i] = val
            except (ValueError, RuntimeError):
                intensities[i] = np.nan

        if np.any(np.isnan(intensities)):
            return None

        # Compute per-wing distances at each level intensity (for plotting/storage)
        blue_dists = np.array([float(inv_interp_blue(I)) for I in intensities])
        red_dists = np.array([float(inv_interp_red(I)) for I in intensities])

        return {
            'intensities': intensities,
            'widths': widths,
            'continuum_level': continuum_level,
            'intensity_core': intensity_core,
            'width_continuum': width_continuum,
            'blue_dists': blue_dists,
            'red_dists': red_dists
        }
    except Exception:
        return None

def fit_ti_spectrum(line_data, line_wvl, continuum_data):
    """
    Fit Ti line for all spatial pixels in a spectrum.

    Args:
        line_data: Line + residual data array (shape: wavelength, slit, scan)
        line_wvl: Wavelength array (in Angstrom)
        continuum_data: Continuum data array (shape: n_wvl_cont, n_slit, n_scan)
                        Per-pixel continuum level is computed as mean over wavelength

    Returns:
        dict: Dictionary containing fitted arrays:
            - 'best_fit': Best fit for all pixels (same shape as line_data)
            - 'component_ti': Ti component for all pixels
            - 'component_residual': Residual component for all pixels
            - 'levels': 3D array of intensity levels (shape: n_levels, n_slit, n_scan)
            - 'widths': 1D array of widths (in Angstrom) from line core
            - 'metadata': Dictionary with continuum_level, intensity_core, width_continuum per pixel
    """
    n_wvl, n_slit, n_scan = line_data.shape
    best_fit = np.zeros_like(line_data)
    component_ti = np.zeros_like(line_data)
    component_residual = np.zeros_like(line_data)
    
    # Initialize levels array (will store intensities at each level for each pixel)
    # First, get n_levels from a test pixel
    test_profile = line_data[:, 0, 0]
    test_continuum = robust_continuum_mean(continuum_data[:, 0, 0])
    test_center = line_wvl[np.argmin(test_profile)]
    test_levels = line_levels(line_wvl, test_profile, test_center, test_continuum)
    if test_levels is not None:
        n_levels = len(test_levels['intensities'])
        levels_array = np.zeros((n_levels, n_slit, n_scan))
        widths = test_levels['widths']  # Same for all pixels
    else:
        levels_array = None
        widths = None
    
    # Store metadata per pixel
    continuum_levels = np.zeros((n_slit, n_scan))
    intensity_cores = np.zeros((n_slit, n_scan))
    width_continuums = np.zeros((n_slit, n_scan))
    blue_dists_arr = np.full((n_levels, n_slit, n_scan), np.nan) if levels_array is not None else None
    red_dists_arr = np.full((n_levels, n_slit, n_scan), np.nan) if levels_array is not None else None

    total_pixels = n_slit * n_scan

    # Check if running in interactive environment (Jupyter/IPython)
    # If so, disable multiprocessing to avoid pickling issues
    import sys
    use_multiprocessing = total_pixels > 100 and not hasattr(sys, 'ps1')

    # Use multiprocessing for large datasets (only in non-interactive mode)
    if use_multiprocessing:
        from multiprocessing import shared_memory
        import uuid
        
        n_workers = min(os.cpu_count(), total_pixels)  # Use all available CPU cores
        print(f"Using {n_workers} workers for {total_pixels} pixels with shared memory...")

        # Ensure float64 dtype to match worker's fixed dtype assumption when
        # reconstructing arrays from shared memory (avoids buffer size mismatch)
        line_wvl_f64 = np.ascontiguousarray(line_wvl, dtype=np.float64)
        line_data_f64 = np.ascontiguousarray(line_data, dtype=np.float64)
        continuum_data_f64 = np.ascontiguousarray(continuum_data, dtype=np.float64)

        # Create shared memory blocks sized exactly for float64 data
        wvl_shm = shared_memory.SharedMemory(create=True, size=line_wvl_f64.nbytes)
        line_data_shm = shared_memory.SharedMemory(create=True, size=line_data_f64.nbytes)
        continuum_data_shm = shared_memory.SharedMemory(create=True, size=continuum_data_f64.nbytes)
        
        # Copy data to shared memory
        wvl_shared = np.ndarray(line_wvl_f64.shape, dtype=np.float64, buffer=wvl_shm.buf)
        line_data_shared = np.ndarray(line_data_f64.shape, dtype=np.float64, buffer=line_data_shm.buf)
        continuum_data_shared = np.ndarray(continuum_data_f64.shape, dtype=np.float64, buffer=continuum_data_shm.buf)
        
        wvl_shared[:] = line_wvl_f64[:]
        line_data_shared[:] = line_data_f64[:]
        continuum_data_shared[:] = continuum_data_f64[:]

        # Prepare arguments for all pixels (only indices, not data)
        args_list = []
        for slit_idx in range(n_slit):
            for scan_idx in range(n_scan):
                args_list.append((
                    wvl_shm.name, line_data_shm.name, continuum_data_shm.name,
                    line_wvl_f64.shape, line_data_f64.shape, continuum_data_f64.shape,
                    slit_idx, scan_idx
                ))

        try:
            # Process with multiprocessing using shared memory
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_fit_single_pixel_shared, args): args for args in args_list}

                with tqdm(total=total_pixels, desc="Fitting TI pixels") as pbar:
                    for future in as_completed(futures):
                        slit_idx, scan_idx, result, error = future.result()
                        if error is not None:
                            print(f"Failed to fit pixel (slit={slit_idx}, scan={scan_idx}): {error}")
                            profile = line_data[:, slit_idx, scan_idx]
                            best_fit[:, slit_idx, scan_idx] = profile
                            component_ti[:, slit_idx, scan_idx] = profile
                            component_residual[:, slit_idx, scan_idx] = 0
                        else:
                            best_fit[:, slit_idx, scan_idx] = result['best_fit']
                            component_ti[:, slit_idx, scan_idx] = result['component_ti']
                            component_residual[:, slit_idx, scan_idx] = result['component_residual']
                            # Store levels if available
                            if result['levels'] is not None and levels_array is not None:
                                levels_array[:, slit_idx, scan_idx] = result['levels']['intensities']
                                continuum_levels[slit_idx, scan_idx] = result['levels']['continuum_level']
                                intensity_cores[slit_idx, scan_idx] = result['levels']['intensity_core']
                                width_continuums[slit_idx, scan_idx] = result['levels']['width_continuum']
                                blue_dists_arr[:, slit_idx, scan_idx] = result['levels']['blue_dists']
                                red_dists_arr[:, slit_idx, scan_idx] = result['levels']['red_dists']
                        pbar.update(1)
        finally:
            # Clean up shared memory
            wvl_shm.close()
            wvl_shm.unlink()
            line_data_shm.close()
            line_data_shm.unlink()
            continuum_data_shm.close()
            continuum_data_shm.unlink()
    else:
        # Sequential processing (for small datasets or interactive environments)
        with tqdm(total=total_pixels, desc="Fitting pixels") as pbar:
            for slit_idx in range(n_slit):
                for scan_idx in range(n_scan):
                    profile = line_data[:, slit_idx, scan_idx]
                    continuum_pixel = robust_continuum_mean(continuum_data[:, slit_idx, scan_idx])

                    try:
                        result = fit_ti_pixel(line_wvl, profile, continuum_pixel)
                        best_fit[:, slit_idx, scan_idx] = result['best_fit']
                        component_ti[:, slit_idx, scan_idx] = result['component_ti']
                        component_residual[:, slit_idx, scan_idx] = result['component_residual']
                        # Store levels if available
                        if result['levels'] is not None and levels_array is not None:
                            levels_array[:, slit_idx, scan_idx] = result['levels']['intensities']
                            continuum_levels[slit_idx, scan_idx] = result['levels']['continuum_level']
                            intensity_cores[slit_idx, scan_idx] = result['levels']['intensity_core']
                            width_continuums[slit_idx, scan_idx] = result['levels']['width_continuum']
                            blue_dists_arr[:, slit_idx, scan_idx] = result['levels']['blue_dists']
                            red_dists_arr[:, slit_idx, scan_idx] = result['levels']['red_dists']
                    except Exception as e:
                        print(f"Failed to fit pixel (slit={slit_idx}, scan={scan_idx}): {e}")
                        # Use original data as fallback
                        best_fit[:, slit_idx, scan_idx] = profile
                        component_ti[:, slit_idx, scan_idx] = profile
                        component_residual[:, slit_idx, scan_idx] = 0

                    pbar.update(1)
    return {
        'best_fit': best_fit,
        'component_ti': component_ti,
        'component_residual': component_residual,
        'levels': levels_array,
        'widths': widths,
        'metadata': {
            'continuum_levels': continuum_levels,
            'intensity_cores': intensity_cores,
            'width_continuums': width_continuums,
            'blue_dists': blue_dists_arr,
            'red_dists': red_dists_arr
        }
    }

#----------------------------------------------------------------------
# SR Line Fitting Functions
#----------------------------------------------------------------------

SR_CENTER_WL = 4607.34  # Sr I line center in Angstrom
SR_SPEC_REGION_WIDTH = 0.16  # Width of spectral region for fitting (Angstrom)
B_CENTER_WL = 4607.65  # B component center in Angstrom

def _sr_piecewise_model():
    """Create the full piecewise model for SR line (SR + B + constant)."""
    voigt_sr = piecewise_voigt(prefix='sr')
    pars = voigt_sr.make_params()
    voigtB = piecewise_voigt(prefix='B')
    pars.update(voigtB.make_params())

    const = ConstantModel(prefix='c_')
    pars.update(const.make_params())

    model = voigt_sr + voigtB + const
    return model, pars

def _initialize_sr_parameter(params, continuum, wvl=None, profile=None):
    """Initialize parameters for SR line fitting."""
    # SR component parameters
    params['sr_r_amplitude'].value = -0.08
    params['sr_r_amplitude'].min = -0.1
    params['sr_r_amplitude'].max = -0.03

    params['sr_b_amplitude'].value = -0.08
    params['sr_b_amplitude'].min = -0.1
    params['sr_b_amplitude'].max = -0.03

    # Initialize SR center with parabola fit if wvl and profile provided
    if wvl is not None and profile is not None:
        sr_center_refined = refine_center_with_parabola(wvl, profile, SR_CENTER_WL)
        params['sr_r_center'].value = sr_center_refined
    else:
        params['sr_r_center'].value = SR_CENTER_WL

    params['sr_r_center'].min = params['sr_r_center'].value - 0.005
    params['sr_r_center'].max = params['sr_r_center'].value + 0.005
    params['sr_r_center'].vary = True
    params['sr_b_center'].expr = 'sr_r_center'

    params['sr_r_sigma'].value = 0.03
    params['sr_b_sigma'].value = 0.03
    params['sr_r_sigma'].max = 0.05
    params['sr_b_sigma'].max = 0.05
    params['sr_r_sigma'].min = 0.01
    params['sr_b_sigma'].min = 0.01

    params['sr_b_gamma'].value = 0.03
    params['sr_r_gamma'].value = 0.03
    params['sr_r_gamma'].min = 0.001
    params['sr_r_gamma'].max = 0.09
    params['sr_b_gamma'].min = 0.001
    params['sr_b_gamma'].max = 0.09

    params['sr_y'].expr = 'sr_r_center'

    # B component parameters
    params['B_r_amplitude'].value = -0.09
    params['B_b_amplitude'].value = -0.09
    params['B_r_amplitude'].max = -0.03
    params['B_r_amplitude'].min = -0.1
    params['B_b_amplitude'].max = -0.03
    params['B_b_amplitude'].min = -0.1

    # Use parabola-based initialization for B component center if wvl and profile are provided
    if wvl is not None and profile is not None:
        b_center_refined = refine_center_with_parabola(wvl, profile, B_CENTER_WL, search_width=0.07)
        params['B_r_center'].value = b_center_refined
    else:
        params['B_r_center'].value = B_CENTER_WL

    params['B_r_center'].vary = True
    params['B_b_center'].expr = 'B_r_center'

    params['B_r_center'].min = params['B_r_center'].value - 0.005
    params['B_r_center'].max = params['B_r_center'].value + 0.005

    params['B_r_sigma'].value = 0.03
    params['B_b_sigma'].value = 0.03
    params['B_r_sigma'].max = 0.05
    params['B_b_sigma'].max = 0.05
    params['B_r_sigma'].min = 0.01
    params['B_b_sigma'].min = 0.01

    params['B_r_gamma'].value = 0.03
    params['B_b_gamma'].value = 0.03
    params['B_r_gamma'].min = 0.001
    params['B_r_gamma'].max = 0.09
    params['B_b_gamma'].min = 0.001
    params['B_b_gamma'].max = 0.09

    params['B_y'].expr = 'B_r_center'

    # Constant continuum
    params['c_c'].value = continuum
    params['c_c'].vary = False
    params['c_c'].min = continuum * 0.9
    params['c_c'].max = continuum * 1.1

    return params


# Helper function for multiprocessing with shared memory (must be at module level)
def _fit_sr_single_pixel_shared(args):
    """Fit a single SR pixel for multiprocessing using shared memory."""
    wvl_shm_name, line_data_shm_name, continuum_data_shm_name, wvl_shape, line_data_shape, continuum_data_shape, slit_idx, scan_idx = args
    
    try:
        # Access shared memory
        from multiprocessing import shared_memory
        
        # Connect to shared memory blocks
        wvl_shm = shared_memory.SharedMemory(name=wvl_shm_name)
        line_data_shm = shared_memory.SharedMemory(name=line_data_shm_name)
        continuum_data_shm = shared_memory.SharedMemory(name=continuum_data_shm_name)
        
        # Create numpy arrays from shared memory
        wvl = np.ndarray(wvl_shape, dtype=np.float64, buffer=wvl_shm.buf)
        line_data = np.ndarray(line_data_shape, dtype=np.float64, buffer=line_data_shm.buf)
        continuum_data = np.ndarray(continuum_data_shape, dtype=np.float64, buffer=continuum_data_shm.buf)
        
        # Extract profile and continuum for this pixel
        profile = line_data[:, slit_idx, scan_idx].copy()  # Copy to avoid keeping shared memory reference
        continuum_pixel = robust_continuum_mean(continuum_data[:, slit_idx, scan_idx])
        wvl_copy = wvl.copy()  # Copy wavelength
        
        # Close shared memory access
        wvl_shm.close()
        line_data_shm.close()
        continuum_data_shm.close()
        
        # Fit the pixel
        result = fit_sr_pixel(wvl_copy, profile, continuum_pixel)
        return (slit_idx, scan_idx, result, None)
    except Exception as e:
        return (slit_idx, scan_idx, None, str(e))

# Helper function for multiprocessing (must be at module level)
def _fit_sr_single_pixel(args):
    """Fit a single SR pixel for multiprocessing."""
    wvl, profile, continuum, slit_idx, scan_idx = args
    try:
        result = fit_sr_pixel(wvl, profile, continuum)
        return (slit_idx, scan_idx, result, None)
    except Exception as e:
        return (slit_idx, scan_idx, None, str(e))

def fit_sr_pixel(wvl, profile, continuum):
    """
    Fit a single pixel spectrum using the SR piecewise Voigt model.

    Args:
        wvl: Wavelength array (in Angstrom)
        profile: Intensity profile (line + residual)
        continuum: Continuum level (fixed). If None or not finite, uses profile[-1]

    Returns:
        dict: Dictionary containing fitted arrays:
            - 'best_fit': Best fit profile
            - 'component_sr': Sr component
            - 'component_residual': Residual component (B)
            - 'params': Fitted parameters
    """
    # Continuum fallback
    if continuum is None:
        continuum = profile[-1]
    if not np.isfinite(continuum):
        continuum = np.nanmean(profile)

    # Initialize model and parameters (parabola refinement happens inside _initialize_sr_parameter)
    model, pars = _sr_piecewise_model()
    pars = _initialize_sr_parameter(pars, continuum, wvl=wvl, profile=profile)

    # Add weights for overlap regions
    weights = add_overlap_weights(wvl, pars, prefixes=['sr', 'B'])

    # Fit the model
    out = model.fit(profile, pars, x=wvl, weights=weights, method='least_squares', max_nfev=50)  # Limits total function evaluations)

    # Evaluate the best fit and components
    best_fit = model.eval(out.params, x=wvl)
    components = model.eval_components(params=out.params, x=wvl)

    # Extract components
    component_sr = components['sr_'] + components['c_']
    component_residual = components['B_'] + components['c_']

    # Calculate line levels (may fail for bad data)
    line_center = out.params['sr_r_center'].value
    try:
        levels = line_levels(wvl, component_sr, line_center, continuum)
    except Exception:
        levels = None

    return {
        'best_fit': best_fit,
        'component_sr': component_sr,
        'component_residual': component_residual,
        'params': out.params,
        'levels': levels
    }

def fit_sr_spectrum(line_data, line_wvl, continuum_data):
    """
    Fit SR line for all pixels in the spectrum.

    Args:
        line_data: 3D array of intensity data (n_wvl, n_slit, n_scan)
        line_wvl: Wavelength array (in Angstrom)
        continuum_data: Continuum data array (shape: n_wvl_cont, n_slit, n_scan)
                        Per-pixel continuum level is computed as mean over wavelength

    Returns:
        dict: Dictionary containing fitted arrays:
            - 'best_fit': Best fit for all pixels (same shape as line_data)
            - 'component_sr': Sr component for all pixels
            - 'component_residual': Residual component for all pixels
            - 'levels': 3D array of intensity levels (shape: n_levels, n_slit, n_scan)
            - 'widths': 1D array of widths (in Angstrom) from line core
            - 'metadata': Dictionary with continuum_level, intensity_core, width_continuum per pixel
    """
    n_wvl, n_slit, n_scan = line_data.shape
    best_fit = np.zeros_like(line_data)
    component_sr = np.zeros_like(line_data)
    component_residual = np.zeros_like(line_data)
    
    # Initialize levels array (will store intensities at each level for each pixel)
    # First, get n_levels from a test pixel
    test_profile = line_data[:, 0, 0]
    test_continuum = robust_continuum_mean(continuum_data[:, 0, 0])
    test_center = line_wvl[np.argmin(test_profile)]
    test_levels = line_levels(line_wvl, test_profile, test_center, test_continuum)
    if test_levels is not None:
        n_levels = len(test_levels['intensities'])
        levels_array = np.zeros((n_levels, n_slit, n_scan))
        widths = test_levels['widths']  # Same for all pixels
    else:
        levels_array = None
        widths = None
    
    # Store metadata per pixel
    continuum_levels = np.zeros((n_slit, n_scan))
    intensity_cores = np.zeros((n_slit, n_scan))
    width_continuums = np.zeros((n_slit, n_scan))
    blue_dists_arr = np.full((n_levels, n_slit, n_scan), np.nan) if levels_array is not None else None
    red_dists_arr = np.full((n_levels, n_slit, n_scan), np.nan) if levels_array is not None else None

    total_pixels = n_slit * n_scan

    # Check if running in interactive environment (Jupyter/IPython)
    import sys
    use_multiprocessing = total_pixels > 100 and not hasattr(sys, 'ps1')

    # Use multiprocessing for large datasets (only in non-interactive mode)
    if use_multiprocessing:
        from multiprocessing import shared_memory
        
        n_workers = min(os.cpu_count(), total_pixels)
        print(f"Using {n_workers} workers for {total_pixels} pixels with shared memory...")

        # Ensure float64 dtype to match worker's fixed dtype assumption when
        # reconstructing arrays from shared memory (avoids buffer size mismatch)
        line_wvl_f64 = np.ascontiguousarray(line_wvl, dtype=np.float64)
        line_data_f64 = np.ascontiguousarray(line_data, dtype=np.float64)
        continuum_data_f64 = np.ascontiguousarray(continuum_data, dtype=np.float64)

        # Create shared memory blocks sized exactly for float64 data
        wvl_shm = shared_memory.SharedMemory(create=True, size=line_wvl_f64.nbytes)
        line_data_shm = shared_memory.SharedMemory(create=True, size=line_data_f64.nbytes)
        continuum_data_shm = shared_memory.SharedMemory(create=True, size=continuum_data_f64.nbytes)
        
        # Copy data to shared memory
        wvl_shared = np.ndarray(line_wvl_f64.shape, dtype=np.float64, buffer=wvl_shm.buf)
        line_data_shared = np.ndarray(line_data_f64.shape, dtype=np.float64, buffer=line_data_shm.buf)
        continuum_data_shared = np.ndarray(continuum_data_f64.shape, dtype=np.float64, buffer=continuum_data_shm.buf)
        
        wvl_shared[:] = line_wvl_f64[:]
        line_data_shared[:] = line_data_f64[:]
        continuum_data_shared[:] = continuum_data_f64[:]

        # Prepare arguments for all pixels (only indices, not data)
        args_list = []
        for slit_idx in range(n_slit):
            for scan_idx in range(n_scan):
                args_list.append((
                    wvl_shm.name, line_data_shm.name, continuum_data_shm.name,
                    line_wvl_f64.shape, line_data_f64.shape, continuum_data_f64.shape,
                    slit_idx, scan_idx
                ))

        try:
            # Process with multiprocessing using shared memory
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_fit_sr_single_pixel_shared, args): args for args in args_list}

                with tqdm(total=total_pixels, desc="Fitting SR pixels") as pbar:
                    for future in as_completed(futures):
                        slit_idx, scan_idx, result, error = future.result()
                        if error is not None:
                            print(f"Failed to fit pixel (slit={slit_idx}, scan={scan_idx}): {error}")
                            profile = line_data[:, slit_idx, scan_idx]
                            best_fit[:, slit_idx, scan_idx] = profile
                            component_sr[:, slit_idx, scan_idx] = profile
                            component_residual[:, slit_idx, scan_idx] = 0
                        else:
                            best_fit[:, slit_idx, scan_idx] = result['best_fit']
                            component_sr[:, slit_idx, scan_idx] = result['component_sr']
                            component_residual[:, slit_idx, scan_idx] = result['component_residual']
                            # Store levels if available
                            if result['levels'] is not None and levels_array is not None:
                                levels_array[:, slit_idx, scan_idx] = result['levels']['intensities']
                                continuum_levels[slit_idx, scan_idx] = result['levels']['continuum_level']
                                intensity_cores[slit_idx, scan_idx] = result['levels']['intensity_core']
                                width_continuums[slit_idx, scan_idx] = result['levels']['width_continuum']
                                blue_dists_arr[:, slit_idx, scan_idx] = result['levels']['blue_dists']
                                red_dists_arr[:, slit_idx, scan_idx] = result['levels']['red_dists']
                        pbar.update(1)
        finally:
            # Clean up shared memory
            wvl_shm.close()
            wvl_shm.unlink()
            line_data_shm.close()
            line_data_shm.unlink()
            continuum_data_shm.close()
            continuum_data_shm.unlink()
    else:
        # Sequential processing (for small datasets or interactive environments)
        with tqdm(total=total_pixels, desc="Fitting SR pixels") as pbar:
            for slit_idx in range(n_slit):
                for scan_idx in range(n_scan):
                    profile = line_data[:, slit_idx, scan_idx]
                    continuum_pixel = robust_continuum_mean(continuum_data[:, slit_idx, scan_idx])

                    try:
                        result = fit_sr_pixel(line_wvl, profile, continuum_pixel)
                        best_fit[:, slit_idx, scan_idx] = result['best_fit']
                        component_sr[:, slit_idx, scan_idx] = result['component_sr']
                        component_residual[:, slit_idx, scan_idx] = result['component_residual']
                        # Store levels if available
                        if result['levels'] is not None and levels_array is not None:
                            levels_array[:, slit_idx, scan_idx] = result['levels']['intensities']
                            continuum_levels[slit_idx, scan_idx] = result['levels']['continuum_level']
                            intensity_cores[slit_idx, scan_idx] = result['levels']['intensity_core']
                            width_continuums[slit_idx, scan_idx] = result['levels']['width_continuum']
                            blue_dists_arr[:, slit_idx, scan_idx] = result['levels']['blue_dists']
                            red_dists_arr[:, slit_idx, scan_idx] = result['levels']['red_dists']
                    except Exception as e:
                        print(f"Failed to fit pixel (slit={slit_idx}, scan={scan_idx}): {e}")
                        # Use original data as fallback
                        best_fit[:, slit_idx, scan_idx] = profile
                        component_sr[:, slit_idx, scan_idx] = profile
                        component_residual[:, slit_idx, scan_idx] = 0

                    pbar.update(1)

    return {
        'best_fit': best_fit,
        'component_sr': component_sr,
        'component_residual': component_residual,
        'levels': levels_array,
        'widths': widths,
        'metadata': {
            'continuum_levels': continuum_levels,
            'intensity_cores': intensity_cores,
            'width_continuums': width_continuums,
            'blue_dists': blue_dists_arr,
            'red_dists': red_dists_arr
        }
    }

# Predefined split configurations for each line (wavelength ranges in nm)
SPLIT_CONFIGS = {
    'sr': {
        'continuum': (460.716, 460.719),
        'line': (460.719, 460.746),
        'residual': (460.746, 460.783),
    },
    'ti': {
        'continuum': (453.654, 453.66),
        'line': (453.631, 453.654),
        'residual': (453.58, 453.632),
    },
}

def crop_to_array(crop_dict):
    """
    Convert crop dictionary to array format for internal use.

    Args:
        crop_dict: Dictionary with keys 'slit', 'scan', 'wavelength',
                   each containing a tuple (start, end).

    Returns:
        list: Array [scan_start, scan_end, slit_start, slit_end, wl_start, wl_end]
    """
    return [
        crop_dict['scan'][0], crop_dict['scan'][1],
        crop_dict['slit'][0], crop_dict['slit'][1],
        crop_dict['wavelength'][0], crop_dict['wavelength'][1]
    ]

def split_spectrum(intensity, wavelength, split_config):
    """
    Split intensity and wavelength data into continuum, line, and residual parts
    based on provided wavelength ranges, and normalize by the continuum mean.

    Args:
        intensity: Dictionary mapping line -> position -> intensity array
        wavelength: Dictionary mapping line -> wavelength array
        split_config: Dictionary mapping line -> split configuration,
                      where each split config has keys 'continuum', 'line', 'residual',
                      each containing a tuple (wl_min, wl_max) in nm

    Returns:
        dict: Dictionary with same structure as intensity, but each intensity array
              is replaced by a dict with keys 'continuum', 'line', 'residual',
              each containing a tuple (intensity_part, wavelength_part).
              All intensity parts are normalized by the scan and spectral mean
              of the continuum region.
    """
    result = {}

    for line in intensity.keys():
        result[line] = {}
        line_split_config = split_config[line]
        wl = wavelength[line]

        for position in intensity[line].keys():
            intensity_data = intensity[line][position]
            result[line][position] = {}

            # First, extract continuum to compute normalization factor
            continuum_wl_min, continuum_wl_max = line_split_config['continuum']
            mask = (wl >= continuum_wl_min) & (wl <= continuum_wl_max)
            wl_indices = np.where(mask)[0]

            if len(wl_indices) == 0:
                print(f"Warning: No pixels found for {line} {position} continuum range {continuum_wl_min}-{continuum_wl_max} nm")
                continuum_mean = 1.0
            else:
                wl_start, wl_end = wl_indices[0], wl_indices[-1] + 1

                # Slice continuum intensity
                if intensity_data.ndim == 1:
                    continuum_intensity = intensity_data[wl_start:wl_end]
                    continuum_mean = np.mean(continuum_intensity)
                elif intensity_data.ndim == 2:
                    continuum_intensity = intensity_data[:, wl_start:wl_end]
                    continuum_mean = np.mean(continuum_intensity)
                elif intensity_data.ndim == 3:
                    continuum_intensity = intensity_data[:, :, wl_start:wl_end]
                    continuum_mean = np.mean(continuum_intensity)
                else:
                    raise ValueError(f"Unsupported intensity shape: {intensity_data.shape}")

            # Now split all parts and normalize
            for part_name, (wl_min, wl_max) in line_split_config.items():
                # Find pixel indices corresponding to wavelength range
                mask = (wl >= wl_min) & (wl <= wl_max)
                wl_indices = np.where(mask)[0]

                if len(wl_indices) == 0:
                    print(f"Warning: No pixels found for {line} {position} {part_name} range {wl_min}-{wl_max} nm")
                    result[line][position][part_name] = (None, None)
                    continue

                wl_start, wl_end = wl_indices[0], wl_indices[-1] + 1

                # Slice wavelength
                wl_part = wl[wl_start:wl_end]

                # Slice intensity (last dimension is wavelength)
                if intensity_data.ndim == 1:
                    intensity_part = intensity_data[wl_start:wl_end] / continuum_mean
                elif intensity_data.ndim == 2:
                    intensity_part = intensity_data[:, wl_start:wl_end] / continuum_mean
                elif intensity_data.ndim == 3:
                    intensity_part = intensity_data[:, :, wl_start:wl_end] / continuum_mean
                else:
                    raise ValueError(f"Unsupported intensity shape: {intensity_data.shape}")

                result[line][position][part_name] = (intensity_part, wl_part)

    return result


class SpectrumPart:
    """
    Class to hold a single part of a split spectrum (continuum, line, or residual).

    Attributes:
        data: Intensity array (normalized), organized: wavelength, direction along the slit, scan direction (slit positions)
        wvl: Wavelength array
        fit_result: Optional fit result dictionary
    """
    def __init__(self, data, wvl, fit_result=None):
        self.data = data
        self.wvl = wvl
        self.fit_result = fit_result

    def __repr__(self):
        if self.data is None:
            return f"SpectrumPart(data=None, wvl=None)"
        return f"SpectrumPart(data.shape={self.data.shape}, wvl.shape={self.wvl.shape})"

    def save(self, filepath, config_path=None, sequence=None, python_file=None):
        """
        Save the spectrum part to a FITS file.

        Wavelength is converted from Angstrom to nm (divided by 10) before saving.

        Args:
            filepath: Path to save the FITS file
            config_path: Path to the configuration file (saved in header)
            sequence: Sequence number used (saved in header)
            python_file: Name of the Python file that created this file (saved in header)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Create primary HDU with data
        primary_hdu = fits.PrimaryHDU(data=self.data)

        # Convert wavelength from Angstrom to nm for saving
        wvl_nm = self.wvl / 10.0
        wavelength_hdu = fits.ImageHDU(data=wvl_nm, name='WAVELENGTH')

        # Add header information
        header = primary_hdu.header
        header['CONFIG'] = str(config_path) if config_path else 'unknown'
        header['DATE'] = datetime.now().isoformat()
        header['SEQUENCE'] = str(sequence) if sequence is not None else 'unknown'
        header['PYTHON'] = str(python_file) if python_file else 'unknown'

        # Create HDU list
        hdul = fits.HDUList([primary_hdu, wavelength_hdu])
        hdul.writeto(filepath, overwrite=True)

    @classmethod
    def load(cls, filepath):
        """
        Load a spectrum part from a FITS file.

        Wavelength is converted from nm to Angstrom (multiplied by 10) after loading.

        Args:
            filepath: Path to the FITS file

        Returns:
            SpectrumPart object
        """
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            wvl = hdul['WAVELENGTH'].data
            # Convert wavelength from nm to Angstrom
            wvl_angstrom = wvl * 10.0

            return cls(data, wvl_angstrom, fit_result=None)

    def plot(self, ax=None, scan_idx=None, slit_idx=None, **kwargs):
        """
        Plot the spectrum part.

        Args:
            ax: Optional matplotlib axes. If None, creates a new figure.
            scan_idx: Optional scan index (slit positions) to plot specific pixel. If None, averages over scan.
            slit_idx: Optional slit index (direction along the slit) to plot specific pixel. If None, averages over slit.
            **kwargs: Additional arguments passed to ax.plot

        Returns:
            ax: The matplotlib axes object
        """
        if self.data is None or self.wvl is None:
            print("Cannot plot: data or wvl is None")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Handle different data dimensions
        # Data is now in (wavelength, slit, scan) order
        if self.data.ndim == 1:
            ax.plot(self.wvl, self.data, **kwargs)
        elif self.data.ndim == 2:
            # (wavelength, slit)
            if slit_idx is not None:
                # Plot specific slit
                ax.plot(self.wvl, self.data[:, slit_idx], **kwargs)
            else:
                # Average over slit
                slit_avg = np.mean(self.data, axis=1)
                ax.plot(self.wvl, slit_avg, **kwargs)
        elif self.data.ndim == 3:
            # (wavelength, slit, scan)
            if scan_idx is not None and slit_idx is not None:
                # Plot specific pixel
                ax.plot(self.wvl, self.data[:, slit_idx, scan_idx], **kwargs)
            elif scan_idx is not None:
                # Average over slit for specific scan
                ax.plot(self.wvl, np.mean(self.data[:, :, scan_idx], axis=1), **kwargs)
            elif slit_idx is not None:
                # Average over scan for specific slit
                ax.plot(self.wvl, np.mean(self.data[:, slit_idx, :], axis=1), **kwargs)
            else:
                # Average over slit and scan
                slit_avg = np.mean(self.data, axis=(1, 2))
                ax.plot(self.wvl, slit_avg, **kwargs)
        else:
            raise ValueError(f"Unsupported data shape: {self.data.shape}")

        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity (normalized)")
        ax.grid(True, alpha=0.3)

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax


def save_line_levels(filepath, levels, config_path=None, sequence=None, python_file=None):
    """
    Save line levels (OUTPUT of the line-levels calculation) to a dedicated
    FITS file, separate from the fitted line profile (INPUT).

    Format: LEVELS is the PRIMARY HDU (3D image, n_levels x n_slit x n_scan),
    matching run_voigt_as.pro (`writefits, outfile, Mval0[...], h` is the
    first, non-/APPEND write, i.e. Mval0 is the primary HDU there too).
    WIDTHS (1D image) and LEVELS_META (binary table) are extensions.

    Args:
        filepath: Path to save the FITS file (e.g. '{base}_levels.fits')
        levels: Dictionary with keys 'intensities' (3D array), 'widths' (1D array),
                'metadata' (dict with 'continuum_levels', 'intensity_cores', 'width_continuums')
        config_path: Path to the configuration file (saved in header)
        sequence: Sequence number used (saved in header)
        python_file: Name of the Python file that created this file (saved in header)
    """
    from astropy.table import Table

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    levels_array = levels['intensities']  # (n_levels, n_slit, n_scan)
    widths = levels['widths']  # 1D array
    metadata = levels['metadata']

    # LEVELS is the primary HDU (Mval0 format from run_voigt_as.pro)
    primary_hdu = fits.PrimaryHDU(data=levels_array)
    header = primary_hdu.header
    header['EXTNAME'] = 'LEVELS'
    header['CONFIG'] = str(config_path) if config_path else 'unknown'
    header['DATE'] = datetime.now().isoformat()
    header['SEQUENCE'] = str(sequence) if sequence is not None else 'unknown'
    header['PYTHON'] = str(python_file) if python_file else 'unknown'
    header['NLEVELS'] = len(widths)
    header['NSLIT'] = levels_array.shape[1]
    header['NSCAN'] = levels_array.shape[2]

    widths_hdu = fits.ImageHDU(data=widths, name='WIDTHS')

    blue_dists = metadata.get('blue_dists')
    red_dists = metadata.get('red_dists')

    hdul_list = [primary_hdu, widths_hdu]

    if blue_dists is not None:
        hdul_list.append(fits.ImageHDU(data=blue_dists, name='BLUE_DISTS'))
    if red_dists is not None:
        hdul_list.append(fits.ImageHDU(data=red_dists, name='RED_DISTS'))

    meta_table = Table()
    meta_table['continuum_levels'] = metadata['continuum_levels'].flatten()
    meta_table['intensity_cores'] = metadata['intensity_cores'].flatten()
    meta_table['width_continuums'] = metadata['width_continuums'].flatten()
    meta_hdu = fits.BinTableHDU(meta_table, name='LEVELS_META')

    hdul_list.append(meta_hdu)
    hdul = fits.HDUList(hdul_list)
    hdul.writeto(filepath, overwrite=True)


def load_line_levels(filepath):
    """
    Load line levels (OUTPUT of the line-levels calculation) from a
    dedicated FITS file, as saved by save_line_levels().

    Args:
        filepath: Path to the FITS file (e.g. '{base}_levels.fits')

    Returns:
        dict: Dictionary with keys 'intensities' (3D array), 'widths' (1D array),
              'metadata' (dict with 'continuum_levels', 'intensity_cores', 'width_continuums')
    """
    with fits.open(filepath) as hdul:
        levels_array = hdul['LEVELS'].data
        widths = hdul['WIDTHS'].data
        meta_table = hdul['LEVELS_META'].data
        n_slit = hdul['LEVELS'].header['NSLIT']
        n_scan = hdul['LEVELS'].header['NSCAN']

        blue_dists = hdul['BLUE_DISTS'].data if 'BLUE_DISTS' in hdul else None
        red_dists = hdul['RED_DISTS'].data if 'RED_DISTS' in hdul else None

        return {
            'intensities': levels_array,
            'widths': widths,
            'metadata': {
                'continuum_levels': meta_table['continuum_levels'].reshape(n_slit, n_scan),
                'intensity_cores': meta_table['intensity_cores'].reshape(n_slit, n_scan),
                'width_continuums': meta_table['width_continuums'].reshape(n_slit, n_scan),
                'blue_dists': blue_dists,
                'red_dists': red_dists
            }
        }


class Spectrum:
    """
    Class to hold split spectrum data for a specific line and position.

    Attributes:
        continuum: SpectrumPart for continuum region
        line: SpectrumPart for line region
        residual: SpectrumPart for residual region
        fit_sum: SpectrumPart for full fit result (fitted data array)
        fit_line: SpectrumPart for line component only (Ti for TI line, Sr for SR line)
    """
    def __init__(self, continuum, line, residual, line_name=None):
        self.continuum = continuum
        self.line = line
        self.residual = residual
        self.line_name = line_name  # 'ti' or 'sr'
        self.fit_sum = None  # Will store full fit result after fitting
        self.fit_line = None  # Will store line component after fitting
        self.levels = None  # Will store line levels dict (3D intensities array + widths + metadata)
        self._load_path = None  # Store original load path for default save

    def __repr__(self):
        return f"Spectrum(continuum, line, residual)"

    def _has_3d_levels(self):
        """Check whether self.levels holds a valid 3D intensities array (n_levels, n_slit, n_scan)."""
        return (self.levels is not None
                and isinstance(self.levels, dict)
                and 'intensities' in self.levels
                and isinstance(self.levels['intensities'], np.ndarray)
                and self.levels['intensities'].ndim == 3)

    def _merge_pixel_levels(self, slit, scan, pixel_levels, n_slit, n_scan):
        """
        Merge a single pixel's line levels into the shared 3D levels array,
        preserving any previously fitted pixels (loaded or fitted earlier).

        Args:
            slit, scan: pixel indices to update
            pixel_levels: dict returned by line_levels() for this pixel
                          (keys: 'intensities', 'widths', 'continuum_level',
                          'intensity_core', 'width_continuum'), or None
            n_slit, n_scan: full spatial dimensions of the spectrum

        Returns:
            None. Updates self.levels in place.
        """
        if pixel_levels is None:
            return

        if self._has_3d_levels():
            # Reuse existing 3D levels array, update only this pixel
            levels_array = self.levels['intensities']
            widths = self.levels['widths']
            metadata = self.levels['metadata']
            # Ensure blue_dists/red_dists arrays exist (may be missing from old files)
            n_levels = levels_array.shape[0]
            if 'blue_dists' not in metadata or metadata['blue_dists'] is None:
                metadata['blue_dists'] = np.full((n_levels, n_slit, n_scan), np.nan)
            if 'red_dists' not in metadata or metadata['red_dists'] is None:
                metadata['red_dists'] = np.full((n_levels, n_slit, n_scan), np.nan)
        else:
            # No existing levels array, create new one filled with NaN
            n_levels = len(pixel_levels['intensities'])
            levels_array = np.full((n_levels, n_slit, n_scan), np.nan)
            widths = pixel_levels['widths']
            metadata = {
                'continuum_levels': np.full((n_slit, n_scan), np.nan),
                'intensity_cores': np.full((n_slit, n_scan), np.nan),
                'width_continuums': np.full((n_slit, n_scan), np.nan),
                'blue_dists': np.full((n_levels, n_slit, n_scan), np.nan),
                'red_dists': np.full((n_levels, n_slit, n_scan), np.nan)
            }

        levels_array[:, slit, scan] = pixel_levels['intensities']
        metadata['continuum_levels'][slit, scan] = pixel_levels['continuum_level']
        metadata['intensity_cores'][slit, scan] = pixel_levels['intensity_core']
        metadata['width_continuums'][slit, scan] = pixel_levels['width_continuum']
        metadata['blue_dists'][:, slit, scan] = pixel_levels['blue_dists']
        metadata['red_dists'][:, slit, scan] = pixel_levels['red_dists']

        self.levels = {
            'intensities': levels_array,
            'widths': widths,
            'metadata': metadata
        }

    def reconstruct(self, parts=('residual', 'line', 'continuum')):
        """
        Reconstruct a combined spectrum from the given parts, concatenating
        their wavelength/data arrays and removing any duplicate wavelength
        pixels that arise where parts intentionally overlap at their boundary
        (e.g. residual/line share a few pixels for fit continuity).

        Args:
            parts: sequence of attribute names to combine, in any order,
                   e.g. ('residual', 'line') or ('line', 'continuum').

        Returns:
            (wvl, data): wavelength array (sorted, unique) and the
            corresponding data array (same axis-0 ordering), or (None, None)
            if none of the requested parts have data.
        """
        wvl_list = []
        data_list = []
        for part_name in parts:
            part = getattr(self, part_name, None)
            if part is None or part.data is None:
                continue
            wvl_list.append(part.wvl)
            data_list.append(part.data)

        if not wvl_list:
            return None, None

        wvl_concat = np.concatenate(wvl_list)
        data_concat = np.concatenate(data_list, axis=0)

        # Sort by wavelength, then keep only the first occurrence of each
        # unique wavelength value (duplicates come from intentionally
        # overlapping part boundaries, not floating-point noise, since all
        # parts are sliced from the same original wavelength array).
        sort_idx = np.argsort(wvl_concat)
        wvl_sorted = wvl_concat[sort_idx]
        data_sorted = data_concat[sort_idx]

        wvl_unique, unique_idx = np.unique(wvl_sorted, return_index=True)
        data_unique = data_sorted[unique_idx]

        return wvl_unique, data_unique

    def save(self, filepath=None, config_path=None, sequence=None, python_file=None):
        """
        Save each SpectrumPart to separate FITS files.

        Args:
            filepath: Base filepath (without extension). Each part will be saved as
                     filepath_continuum.fits, filepath_line.fits, filepath_residual.fits
                     If None, uses the path from which the spectrum was loaded.
            config_path: Path to the configuration file (saved in header)
            sequence: Sequence number used (saved in header)
            python_file: Name of the Python file that created this file (saved in header)
        """
        if filepath is None:
            if self._load_path is None:
                raise ValueError("No filepath provided and spectrum was not loaded from a file. "
                               "Please provide a filepath argument.")
            filepath = self._load_path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save each part separately
        self.continuum.save(f"{filepath}_continuum.fits", config_path, sequence, python_file)
        self.line.save(f"{filepath}_line.fits", config_path, sequence, python_file)
        self.residual.save(f"{filepath}_residual.fits", config_path, sequence, python_file)

        # Save fitted profiles if available
        if self.fit_sum is not None:
            self.fit_sum.save(f"{filepath}_fit_sum.fits", config_path, sequence, python_file)
        if self.fit_line is not None:
            # INPUT: fitted line profile only (no levels embedded)
            self.fit_line.save(f"{filepath}_fit_line.fits", config_path, sequence, python_file)
        if self.levels is not None:
            # OUTPUT: line levels saved to a dedicated file, separate from the fitted profile
            save_line_levels(f"{filepath}_levels.fits", self.levels, config_path, sequence, python_file)

    @classmethod
    def load(cls, filepath, line_name=None):
        """
        Load a split spectrum from separate FITS files.

        Args:
            filepath: Base filepath (without extension). Expects files
                     filepath_continuum.fits, filepath_line.fits, filepath_residual.fits
            line_name: Optional line name ('ti' or 'sr')

        Returns:
            Spectrum object
        """
        filepath = Path(filepath)
        continuum = SpectrumPart.load(f"{filepath}_continuum.fits")
        line = SpectrumPart.load(f"{filepath}_line.fits")
        residual = SpectrumPart.load(f"{filepath}_residual.fits")

        spectrum = cls(continuum, line, residual, line_name)
        spectrum._load_path = str(filepath)  # Store load path for default save

        # Load fitted profiles if they exist
        try:
            spectrum.fit_sum = SpectrumPart.load(f"{filepath}_fit_sum.fits")
        except FileNotFoundError:
            spectrum.fit_sum = None

        try:
            spectrum.fit_line = SpectrumPart.load(f"{filepath}_fit_line.fits")
        except FileNotFoundError:
            spectrum.fit_line = None

        # Load line levels (OUTPUT) from its own dedicated file, if present
        try:
            spectrum.levels = load_line_levels(f"{filepath}_levels.fits")
        except FileNotFoundError:
            spectrum.levels = None

        return spectrum

    def inspect_fit_quality(self, n_best=5, n_worst=5):
        """
        Inspect fit quality by plotting the best and worst pixel fits.

        Calculates residual sum of squares (RSS) for each pixel and plots
        the n_best and n_worst fits with their pixel indices in the legend.

        Args:
            n_best: Number of best fits to plot (default: 5)
            n_worst: Number of worst fits to plot (default: 5)

        Returns:
            fig: The matplotlib figure object
        """
        if self.fit_sum is None or self.fit_line is None:
            print("No fit results available. Run spectrum.fit() first.")
            return None

        # Get the concatenated line+residual data for comparison
        wvl_line = self.line.wvl
        wvl_residual = self.residual.wvl
        data_line = self.line.data
        data_residual = self.residual.data

        # Concatenate and sort (same as in _fit_ti)
        wvl_full = np.concatenate([wvl_line, wvl_residual])
        data_full = np.concatenate([data_line, data_residual], axis=0)
        sort_idx = np.argsort(wvl_full)
        wvl_full = wvl_full[sort_idx]
        data_full = data_full[sort_idx, :, :]

        # Calculate RSS for each pixel
        n_slit, n_scan = data_line.shape[1], data_line.shape[2]
        rss_values = []
        pixel_indices = []

        for slit_idx in range(n_slit):
            for scan_idx in range(n_scan):
                profile = data_full[:, slit_idx, scan_idx]
                fit_profile = self.fit_sum.data[:, slit_idx, scan_idx]
                rss = np.sum((profile - fit_profile) ** 2)
                rss_values.append(rss)
                pixel_indices.append((slit_idx, scan_idx))

        rss_values = np.array(rss_values)
        pixel_indices = np.array(pixel_indices)

        # Get best and worst indices
        best_indices = np.argsort(rss_values)[:n_best]
        worst_indices = np.argsort(rss_values)[-n_worst:]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot best fits
        ax_best = axes[0]
        for idx in best_indices:
            slit_idx, scan_idx = pixel_indices[idx]
            profile = data_full[:, slit_idx, scan_idx]
            fit_profile = self.fit_sum.data[:, slit_idx, scan_idx]
            line_profile = self.fit_line.data[:, slit_idx, scan_idx]
            ax_best.plot(wvl_full, profile, '+', alpha=0.7, markersize=4)
            ax_best.plot(wvl_full, fit_profile, '--', linewidth=1,
                        label=f'Fit (slit={slit_idx}, scan={scan_idx})')
            ax_best.plot(wvl_full, line_profile, ':', linewidth=1, color='orange')
        ax_best.set_title(f'Best {n_best} Fits (lowest RSS)')
        ax_best.set_xlabel('Wavelength (Angstrom)')
        ax_best.set_ylabel('Intensity')
        ax_best.legend()

        # Plot worst fits
        ax_worst = axes[1]
        for idx in worst_indices:
            slit_idx, scan_idx = pixel_indices[idx]
            profile = data_full[:, slit_idx, scan_idx]
            fit_profile = self.fit_sum.data[:, slit_idx, scan_idx]
            line_profile = self.fit_line.data[:, slit_idx, scan_idx]
            ax_worst.plot(wvl_full, profile, '+', alpha=0.7, markersize=4)
            ax_worst.plot(wvl_full, fit_profile, '--', linewidth=1,
                         label=f'Fit (slit={slit_idx}, scan={scan_idx})')
            ax_worst.plot(wvl_full, line_profile, ':', linewidth=1, color='orange')
        ax_worst.set_title(f'Worst {n_worst} Fits (highest RSS)')
        ax_worst.set_xlabel('Wavelength (Angstrom)')
        ax_worst.set_ylabel('Intensity')
        ax_worst.legend()

        #plt.tight_layout()
        return fig

    def fit(self, slit=None, scan=None):
        """
        Fit the spectrum using line-specific models.

        For TI line: Uses 4 Voigt profiles (A, B, C, D) + constant model.
        Fits each spatial pixel individually, or a single pixel if indices are provided.

        For SR line: Uses 2 Voigt profiles (SR, B) + constant model.
        Fits each spatial pixel individually, or a single pixel if indices are provided.

        Args:
            slit: Optional slit index to fit only that pixel. If None, fits all pixels.
            scan: Optional scan index to fit only that pixel. If None, fits all pixels.
                      Must be provided together with slit_idx.
        """
        if self.line_name == 'ti':
            self._fit_ti(slit=slit, scan=scan)
        elif self.line_name == 'sr':
            self._fit_sr(slit=slit, scan=scan)
        else:
            print(f"Unknown line: {self.line_name}")
            return

    def _fit_ti(self, slit=None, scan=None):
        """
        Fit TI line using piecewise Voigt model.

        Procedure:
        1. Concatenate residual+line for full spectrum
        2. Use per-pixel continuum as fixed continuum level
        3. Fit all spatial pixels separately using fit_ti_spectrum
        4. Store full fit and Ti component as SpectrumParts

        Args:
            slit: Optional slit index to fit only that pixel. If None, fits all pixels.
            scan: Optional scan index to fit only that pixel. If None, fits all pixels.
                  Must be provided together with slit.
        """
        # Use reconstruct to concatenate line and residual data
        wvl_full, data_full = self.reconstruct(parts=('line', 'residual'))

        # Get continuum data for per-pixel fitting
        continuum_data = self.continuum.data  # shape: (n_wvl_cont, n_slit, n_scan)

        # Determine which pixels to fit
        if slit is not None and scan is not None:
            # Fit single pixel
            import time
            profile = data_full[:, slit, scan]
            continuum_pixel = robust_continuum_mean(continuum_data[:, slit, scan])
            start_time = time.time()
            result = fit_ti_pixel(wvl_full, profile, continuum_pixel)
            elapsed = time.time() - start_time
            print(f"Single pixel fit time: {elapsed:.3f}s")

            # Preserve existing fit data if available, only update the single pixel
            n_wvl = wvl_full.shape[0]
            n_slit, n_scan = data_full.shape[1], data_full.shape[2]
            
            if self.fit_sum is not None and self.fit_line is not None:
                # Use existing fitted data, update only this pixel
                best_fit = self.fit_sum.data.copy()
                component_ti = self.fit_line.data.copy()
                best_fit[:, slit, scan] = result['best_fit']
                component_ti[:, slit, scan] = result['component_ti']
            else:
                # No existing fit data, create new arrays with only this pixel
                best_fit = np.zeros((n_wvl, n_slit, n_scan))
                component_ti = np.zeros((n_wvl, n_slit, n_scan))
                best_fit[:, slit, scan] = result['best_fit']
                component_ti[:, slit, scan] = result['component_ti']

            # Store levels data for plotting, preserving previously fitted pixels
            self._merge_pixel_levels(slit, scan, result['levels'], n_slit, n_scan)
        else:
            # Fit all pixels separately with per-pixel continuum
            fit_results = fit_ti_spectrum(data_full, wvl_full, continuum_data)
            best_fit = fit_results['best_fit']
            component_ti = fit_results['component_ti']
            # Store levels data for all pixels
            self.levels = {
                'intensities': fit_results['levels'],  # 3D array (n_levels, n_slit, n_scan)
                'widths': fit_results['widths'],  # 1D array
                'metadata': fit_results['metadata']
            }

        # Store results as SpectrumParts (wavelength is already in Angstrom)
        self.fit_sum = SpectrumPart(best_fit, wvl_full)
        self.fit_line = SpectrumPart(component_ti, wvl_full)

    def _fit_sr(self, slit=None, scan=None):
        """
        Fit SR line using piecewise Voigt model.

        Procedure:
        1. Concatenate residual+line for full spectrum
        2. Use per-pixel continuum as fixed continuum level
        3. Fit all spatial pixels separately using fit_sr_spectrum
        4. Store full fit and SR component as SpectrumParts

        Args:
            slit: Optional slit index to fit only that pixel. If None, fits all pixels.
            scan: Optional scan index to fit only that pixel. If None, fits all pixels.
                  Must be provided together with slit.
        """
        # Use reconstruct to concatenate line and residual data
        wvl_full, data_full = self.reconstruct(parts=('line', 'residual'))

        # Get continuum data for per-pixel fitting
        continuum_data = self.continuum.data  # shape: (n_wvl_cont, n_slit, n_scan)

        # Determine which pixels to fit
        if slit is not None and scan is not None:
            # Fit single pixel
            import time
            profile = data_full[:, slit, scan]
            continuum_pixel = robust_continuum_mean(continuum_data[:, slit, scan])
            start_time = time.time()
            result = fit_sr_pixel(wvl_full, profile, continuum_pixel)
            elapsed = time.time() - start_time
            print(f"Single pixel fit time: {elapsed:.3f}s")

            # Preserve existing fit data if available, only update the single pixel
            n_wvl = wvl_full.shape[0]
            n_slit, n_scan = data_full.shape[1], data_full.shape[2]
            
            if self.fit_sum is not None and self.fit_line is not None:
                # Use existing fitted data, update only this pixel
                best_fit = self.fit_sum.data.copy()
                component_sr = self.fit_line.data.copy()
                best_fit[:, slit, scan] = result['best_fit']
                component_sr[:, slit, scan] = result['component_sr']
            else:
                # No existing fit data, create new arrays with only this pixel
                best_fit = np.zeros((n_wvl, n_slit, n_scan))
                component_sr = np.zeros((n_wvl, n_slit, n_scan))
                best_fit[:, slit, scan] = result['best_fit']
                component_sr[:, slit, scan] = result['component_sr']

            # Store levels data for plotting, preserving previously fitted pixels
            self._merge_pixel_levels(slit, scan, result['levels'], n_slit, n_scan)
        else:
            # Fit all pixels separately with per-pixel continuum
            fit_results = fit_sr_spectrum(data_full, wvl_full, continuum_data)
            best_fit = fit_results['best_fit']
            component_sr = fit_results['component_sr']
            # Store levels data for all pixels
            self.levels = {
                'intensities': fit_results['levels'],  # 3D array (n_levels, n_slit, n_scan)
                'widths': fit_results['widths'],  # 1D array
                'metadata': fit_results['metadata']
            }

        # Store results as SpectrumParts (wavelength is already in Angstrom)
        self.fit_sum = SpectrumPart(best_fit, wvl_full)
        self.fit_line = SpectrumPart(component_sr, wvl_full)

    def plot(self, ax=None, show_continuum=True, show_line=True, show_residual=True,
              scan=None, slit=None, fit=False, show_levels=False):
        """
        Plot all parts of the split spectrum.

        Args:
            ax: Optional matplotlib axes. If None, creates a new figure.
            show_continuum: Whether to plot continuum
            show_line: Whether to plot line
            show_residual: Whether to plot residual
            scan_idx: Optional scan index to plot specific pixel. If None, averages over scan.
            slit_idx: Optional slit index to plot specific pixel. If None, averages over slit.
            fit: Whether to plot fit results (fit_sum, fit_line)
            show_levels: Whether to plot line levels (only available for single pixel fits)

        Returns:
            ax: The matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Helper function to extract spectrum from 3D array
        def extract_spectrum(data, scan, slit):
            if scan is not None and slit is not None:
                return data[:, slit, scan]
            elif scan is not None:
                return np.mean(data[:, :, scan], axis=1)
            elif slit is not None:
                return np.mean(data[:, slit, :], axis=1)
            else:
                return np.mean(data, axis=(1, 2))

        # Plot original data
        if show_continuum and self.continuum.data is not None:
            continuum_data = extract_spectrum(self.continuum.data, scan, slit)
            ax.plot(self.continuum.wvl, continuum_data, '+', label='Continuum', alpha=0.7)

        if show_line and self.line.data is not None:
            line_data = extract_spectrum(self.line.data, scan, slit)
            ax.plot(self.line.wvl, line_data, '+', label='Line', alpha=0.9)

        if show_residual and self.residual.data is not None:
            residual_data = extract_spectrum(self.residual.data, scan, slit)
            ax.plot(self.residual.wvl, residual_data, '+', label='Residual', alpha=0.7)

        # Plot fit results if available and requested
        if fit:
            if self.fit_sum is not None:
                fit_sum_data = extract_spectrum(self.fit_sum.data, scan, slit)
                ax.plot(self.fit_sum.wvl, fit_sum_data, '--', label='Full fit', alpha=0.8, linewidth=1, color='black')

            if self.fit_line is not None:
                fit_line_data = extract_spectrum(self.fit_line.data, scan, slit)
                ax.plot(self.fit_line.wvl, fit_line_data, '--', label='Line component', alpha=0.8, linewidth=1, color='orange')

        # Plot line levels if available and requested
        if show_levels and self.fit_line is not None and slit is not None and scan is not None:
            # Check if we have stored levels for this pixel (3D array format, NaN = not fitted)
            levels = None
            if self._has_3d_levels():
                intensities = self.levels['intensities'][:, slit, scan]
                if not np.all(np.isnan(intensities)):
                    widths = self.levels['widths']
                    continuum_level = self.levels['metadata']['continuum_levels'][slit, scan]
                    blue_dists = self.levels['metadata'].get('blue_dists')
                    red_dists = self.levels['metadata'].get('red_dists')
                    levels = {
                        'widths': widths,
                        'intensities': intensities,
                        'continuum_level': continuum_level,
                        'blue_dists': blue_dists[:, slit, scan] if blue_dists is not None else None,
                        'red_dists': red_dists[:, slit, scan] if red_dists is not None else None,
                    }
            
            # Plot levels if available
            if levels is not None:
                widths = levels['widths']
                intensities = levels['intensities']
                continuum_level = levels['continuum_level']
                blue_dists = levels.get('blue_dists')
                red_dists = levels.get('red_dists')

                # Get line center from fitted profile
                line_center = self.fit_line.wvl[np.argmin(self.fit_line.data[:, slit, scan])]

                # Plot horizontal lines and magenta dots at each level
                for i, (w, intensity) in enumerate(zip(widths, intensities)):
                    if np.isnan(intensity):
                        continue
                    if i % 5 == 0:
                        ax.axhline(y=intensity, color='gray', alpha=0.4, linestyle=':', linewidth=0.5)
                    else:
                        ax.axhline(y=intensity, color='gray', alpha=0.15, linestyle=':', linewidth=0.5)
                    # Two magenta dots at same intensity, asymmetric positions
                    if blue_dists is not None and red_dists is not None:
                        db = blue_dists[i]
                        dr = red_dists[i]
                        if np.isfinite(db) and np.isfinite(dr):
                            x_blue = line_center - db
                            x_red = line_center + dr
                            # Connecting segment shows the combined width
                            seg_alpha = 0.5 if i % 5 == 0 else 0.12
                            ax.plot([x_blue, x_red], [intensity, intensity], '-',
                                    color='magenta', alpha=seg_alpha, linewidth=1, zorder=4)
                            dot_alpha = 0.8 if i % 5 == 0 else 0.3
                            ax.plot(x_blue, intensity, 'o', color='magenta', markersize=3, alpha=dot_alpha, zorder=5)
                            ax.plot(x_red, intensity, 'o', color='magenta', markersize=3, alpha=dot_alpha, zorder=5)
                            # Annotate every 5th level with width and index
                            if i % 5 == 0:
                                w_mA = w * 1000  # Angstrom to mA
                                ax.annotate(f'#{i}: {w_mA:.0f} mA',
                                            xy=(x_red, intensity),
                                            xytext=(5, 0), textcoords='offset points',
                                            fontsize=6, color='magenta', va='center')

                # Add legend entries
                ax.axhline(y=intensities[0], color='gray', alpha=0.4, linestyle=':', linewidth=0.5, label='Line levels')
                if blue_dists is not None and red_dists is not None:
                    delta_mA = (widths[1] - widths[0]) * 1000 if len(widths) > 1 else 0
                    ax.plot([], [], 'o', color='magenta', markersize=3,
                            label=f'Level positions (Δ={delta_mA:.1f} mA)')

                # Plot continuum level with separate label
                ax.axhline(y=continuum_level, color='red', alpha=0.5, linestyle='--', linewidth=1, label='95% continuum')

        ax.legend(ncol=4)
        ax.set_title("Split Spectrum")

        if ax is None:
            plt.tight_layout()
            plt.show()

        return ax


class SpectrumContainer:
    """
    Container class to hold multiple Spectrum objects organized by line and position.

    Allows access via string keys: container['ti']['disk_center']
    """
    def __init__(self, spectra_dict):
        self._spectra = spectra_dict

    def __getitem__(self, key):
        return self._spectra[key]

    def __setitem__(self, key, value):
        self._spectra[key] = value

    def __repr__(self):
        lines = list(self._spectra.keys())
        n_spectra = sum(len(positions) for positions in self._spectra.values())
        return f"SpectrumContainer(lines={lines}, n_spectra={n_spectra})"

    def keys(self):
        return self._spectra.keys()

    def values(self):
        return self._spectra.values()

    def items(self):
        return self._spectra.items()

    def save_all(self, config_path=None):
        """
        Save all spectra to individual FITS files.

        Creates a directory structure: config_directories.reduced/spectra/line/position_*.fits
        Each SpectrumPart is saved separately as continuum, line, residual.

        Args:
            config_path: Path to the configuration file. If None, uses CONFIG_PATH.
                         This determines the base directory for saving files.
        """
        if config_path is None:
            config_path = CONFIG_PATH

        # Load config to get the reduced directory (use any line, directories are the same)
        config = get_config(line='ti', config_path=config_path, auto_discover_files=True, auto_create_dirs=False)
        base_path = Path(config.directories.reduced) / 'spectra'

        python_file = 'process_formation_height_line_levels'

        n_spectra = 0
        for line in self._spectra.keys():
            for position in self._spectra[line].keys():
                position_dir = base_path / line / position
                position_dir.mkdir(parents=True, exist_ok=True)
                filepath = position_dir / position
                sequence = POSITION_TO_SEQUENCE.get(position, 'unknown')
                self._spectra[line][position].save(filepath, config_path, sequence, python_file)
                n_spectra += 1

        print(f"Saved {n_spectra} spectra to {base_path}")

    def plot(self, line=None, position=None, fit=False, scan=None, slit=None,
             show_continuum=True, show_line=True, show_residual=True):
        """
        Plot a specific spectrum from the container.

        Args:
            line: Line name (e.g., 'ti', 'sr'). If None, uses first available line.
            position: Position name (e.g., 'disk_center'). If None, uses first available position.
            fit: Whether to plot fit results
            scan: Optional scan index for specific pixel
            slit: Optional slit index for specific pixel
            show_continuum: Whether to plot continuum
            show_line: Whether to plot line
            show_residual: Whether to plot residual
        """
        if line is None:
            line = list(self._spectra.keys())[0]
            print(f"Using default line: {line}")

        if position is None:
            position = list(self._spectra[line].keys())[0]
            print(f"Using default position: {position}")

        spectrum = self._spectra[line][position]
        return spectrum.plot(ax=None, show_continuum=show_continuum, show_line=show_line,
                           show_residual=show_residual, scan_idx=scan, slit_idx=slit, fit=fit)

    def plot_all(self, show_continuum=True, show_line=True, show_residual=True):
        """
        Plot all spectra in a grid layout.

        Args:
            show_continuum: Whether to plot continuum
            show_line: Whether to plot line
            show_residual: Whether to plot residual
        """
        lines = list(self._spectra.keys())
        n_lines = len(lines)

        fig, axes = plt.subplots(n_lines, 1, figsize=(12, 6 * n_lines))

        if n_lines == 1:
            axes = [axes]

        for idx, line in enumerate(lines):
            ax = axes[idx]
            positions = list(self._spectra[line].keys())
            for position in positions:
                spectrum = self._spectra[line][position]
                spectrum.plot(ax, show_continuum=show_continuum,
                            show_line=show_line, show_residual=show_residual,
                            label=position)
            ax.legend()
            ax.set_title(f"{line.upper()} - All Positions")

        plt.tight_layout()
        plt.show()

    @classmethod
    def load_all(cls, config_path=None):
        """
        Load all spectra from a directory structure.

        Expects directory structure: config_directories.reduced/spectra/line/position_*.fits

        Args:
            config_path: Path to the configuration file. If None, uses CONFIG_PATH.
                         This determines the base directory for loading files.

        Returns:
            SpectrumContainer with loaded spectra
        """
        if config_path is None:
            config_path = CONFIG_PATH

        # Load config to get the reduced directory (use any line, directories are the same)
        config = get_config(line='ti', config_path=config_path, auto_discover_files=True, auto_create_dirs=False)
        base_path = Path(config.directories.reduced) / 'spectra'
        spectra_dict = {}

        for line_dir in base_path.iterdir():
            if line_dir.is_dir():
                line = line_dir.name
                spectra_dict[line] = {}
                # Look for position directories (each containing continuum, line, residual FITS files)
                for position_dir in line_dir.iterdir():
                    if position_dir.is_dir():
                        position = position_dir.name
                        # Load Spectrum from the position directory
                        # Files are named: position_continuum.fits, position_line.fits, position_residual.fits
                        base_filepath = position_dir / position
                        spectra_dict[line][position] = Spectrum.load(base_filepath, line_name=line)

        n_spectra = sum(len(v) for v in spectra_dict.values())
        print(f"Loaded {n_spectra} spectra from {base_path}")
        return cls(spectra_dict)


def convert_to_spectrum(split_data):
    """
    Convert the dictionary output from split_spectrum to Spectrum objects.

    Reorders axes from (scan, slit, wavelength) to (wavelength, slit, scan).

    Args:
        split_data: Dictionary from split_spectrum function

    Returns:
        SpectrumContainer with Spectrum objects organized by line and position
    """
    result = {}
    for line in split_data.keys():
        result[line] = {}
        for position in split_data[line].keys():
            parts = split_data[line][position]

            # Reorder axes for each part: (scan, slit, wavelength) -> (wavelength, slit, scan)
            continuum_data = _reorder_axes(parts['continuum'][0])
            line_data = _reorder_axes(parts['line'][0])
            residual_data = _reorder_axes(parts['residual'][0])

            result[line][position] = Spectrum(
                SpectrumPart(continuum_data, parts['continuum'][1] * 10.0),
                SpectrumPart(line_data, parts['line'][1] * 10.0),
                SpectrumPart(residual_data, parts['residual'][1] * 10.0),
                line_name=line
            )
    return SpectrumContainer(result)


def _reorder_axes(data):
    """
    Reorder data axes from (scan, slit, wavelength) to (wavelength, slit, scan).

    Args:
        data: Intensity array with shape (n_scan, n_slit, n_wavelength) or similar

    Returns:
        Reordered array with shape (n_wavelength, n_slit, n_scan)
    """
    if data is None:
        return None
    if data.ndim == 1:
        return data  # Already 1D, no reordering needed
    elif data.ndim == 2:
        # (scan, wavelength) -> (wavelength, scan)
        return data.T
    elif data.ndim == 3:
        # (scan, slit, wavelength) -> (wavelength, slit, scan)
        return np.transpose(data, (2, 1, 0))
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

def get_formation_height_config(line: str, position: str):
    """
    Get configuration for a specific line and disk position.

    Args:
        line: Spectral line ('ti' or 'sr')
        position: Disk position ('disk_center', 'm40', etc.)

    Returns:
        int: Sequence number for the given line and position
    """
    return POSITION_TO_SEQUENCE[position]

def get_all_lines():
    """
    Get all available spectral lines.

    Returns:
        list: List of line identifiers (e.g., ['ti', 'sr'])
    """
    return AVAILABLE_LINES

def get_all_positions(line: str = None):
    """
    Get all available disk positions.

    Args:
        line: Spectral line (optional, kept for backward compatibility)
              Since positions are shared, this parameter is not used.

    Returns:
        list: List of position strings (e.g., ['disk_center', 'm40', 'm30', ...])
    """
    return list(POSITION_TO_SEQUENCE.keys())

def iterate_positions(line: str = None):
    """
    Iterate over all positions, yielding (position, sequence) pairs.

    Args:
        line: Spectral line (optional, kept for backward compatibility)
              Since positions are shared, this parameter is not used.

    Yields:
        tuple: (position, sequence) for each position
    """
    for position, sequence in POSITION_TO_SEQUENCE.items():
        yield position, sequence

def load_all_l4_for_line(line: str):
    """
    Load L4 data for all positions of a given line.

    Args:
        line: Spectral line ('ti' or 'sr')

    Returns:
        dict: Dictionary mapping position -> (l4_data, header)
    """
    results = {}
    for position, sequence in iterate_positions(line):
        l4_data, header = load_l4_for_position(line, position)
        results[position] = (l4_data, header)
        print(f"Loaded {line.upper()} at {position} (sequence {sequence})")
    return results

def load_l4_for_position(line: str, position: str):
    """
    Load L4 data for a specific line and disk position.

    Args:
        line: Spectral line ('ti' or 'sr')
        position: Disk position ('disk_center', '-40', etc.)

    Returns:
        tuple: (l4_data, header) from tio.read_any_file
    """
    sequence = get_formation_height_config(line, position)
    config = get_config(
        line=line,
        config_path=CONFIG_PATH,
        auto_discover_files=False,  # Don't auto-discover yet
        auto_create_dirs=False
    )
    # Override sequence for this position
    config.dataset['scan']['sequence'] = sequence

    # Now discover files with the correct sequence
    from themis.datasets.themis_datasets_2025 import _build_file_set
    config.dataset['scan']['files'] = _build_file_set(
        config.directories,
        config.dataset['scan'],
        config.data_types,
        config.reduction_levels,
        config.cam,
        config.line,
        yshift_calibration_date=config.yshift_calibration_date,
    )

    return tio.read_any_file(config, data_type='scan', status='l4')


def calculate_intensity(line: str, position: str, crop=None):
    """
    Calculate intensity for a specific line and position.

    Args:
        line: Spectral line ('ti' or 'sr')
        position: Disk position ('disk_center', '-40', etc.)
        crop: Optional crop specification. Can be:
              - None: no cropping
              - Array: [scan_start, scan_end, slit_start, slit_end, wl_start, wl_end]
              - Dict: {'slit': (start, end), 'scan': (start, end), 'wavelength': (start, end)}

    Returns:
        numpy array: Intensity values with shape (scan, slit, wavelength)
    """
    data, header = load_l4_for_position(line, position)

    # Check that only one map exists in the cycle set
    map_indices = sorted(set(k[2] for k in data.keys()))
    if len(map_indices) > 1:
        raise ValueError(f"Multiple maps found in cycle set for {line} at {position}: {map_indices}. "
                         f"Expected exactly one map.")
    map_idx = map_indices[0]

    if line == 'sr':
        result = tt.compute_polarimetry(data)
        # Collapse the map dimension (axis 1) to match TI shape
        sr_intensity = result.uml.I[:, map_idx, :, :]
        intensity = sr_intensity.squeeze(axis=1) if sr_intensity.shape[1] == 1 else sr_intensity
    elif line == 'ti': # use only lower state, as argued in the data reduction documentation
        # Since we verified only one map exists, stacking all is fine
        intensity = data.get_state('pQ').stack_all('lower')

    # Apply cropping if specified
    if crop is not None:
        # Convert dict to array if needed
        if isinstance(crop, dict):
            crop = crop_to_array(crop)
        scan_start, scan_end, slit_start, slit_end, wl_start, wl_end = crop
        intensity = intensity[scan_start:scan_end, slit_start:slit_end, wl_start:wl_end]

    return intensity
            
    
    

def calculate_intensity_for_all(crop=False):
    """
    Calculate intensity for all lines and positions and plot scan-averaged spectra.

    Args:
        crop: Boolean to control cropping. If True, uses predefined crop configurations
              from CROP_CONFIGS (line-dependent). If False, no cropping is applied.

    Returns:
        tuple: (intensity, wavelength) where intensity is a dictionary mapping
               line -> position -> full intensity array, and wavelength is a
               dictionary mapping line -> wavelength array
    """
    intensity = {}
    wavelength = {}
    scan_avg_intensity = {}  # For plotting only

    for line in get_all_lines():
        print(f"  Line: {line}")
        intensity[line] = {}
        wavelength[line] = {}
        scan_avg_intensity[line] = {}

        # Load wavelength once per line
        config = get_config(
            line=line,
            config_path=CONFIG_PATH,
            auto_discover_files=True,
            auto_create_dirs=False
        )
        wavelength[line], _ = tio.read_any_file(config, data_type='scan', status='wl')

        # Determine crop configuration for this line
        line_crop = CROP_CONFIGS[line] if crop else None

        # Apply wavelength cropping if crop is enabled
        if crop and line_crop is not None:
            wl_crop = line_crop['wavelength']
            wavelength[line] = wavelength[line][wl_crop[0]:wl_crop[1]]

        for position in get_all_positions(line):
            print(f"    Position: {position}")
            intensity_data = calculate_intensity(line, position, line_crop)

            # Store full intensity data
            intensity[line][position] = intensity_data

            # Calculate scan average (mean over scan axis) for plotting
            # Assuming shape is (n_scan, n_wavelength) or (n_scan, n_slit, n_wavelength)
            if intensity_data.ndim == 2:
                scan_avg = np.mean(intensity_data, axis=0)
            elif intensity_data.ndim == 3:
                scan_avg = np.mean(intensity_data, axis=(0, 1))
            else:
                scan_avg = intensity_data

            scan_avg_intensity[line][position] = scan_avg

    # Create figure with 2 subplots (one for each line)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for idx, line in enumerate(get_all_lines()):
        ax = axes[idx]
        wl = wavelength[line]
        for position in get_all_positions(line):
            spec = scan_avg_intensity[line][position]
            ax.plot(wl, spec, label=position)

            # Plot a random pixel spectrum (dashed line)
            intensity_data = intensity[line][position]
            # Get scan dimensions (excluding wavelength)
            if intensity_data.ndim == 2:
                # Shape is (n_scan, n_wavelength)
                random_scan_idx = np.random.randint(0, intensity_data.shape[0])
                pixel_spec = intensity_data[random_scan_idx, :]
            elif intensity_data.ndim == 3:
                # Shape is (n_scan, n_slit, n_wavelength)
                random_scan_idx = np.random.randint(0, intensity_data.shape[0])
                random_slit_idx = np.random.randint(0, intensity_data.shape[1])
                pixel_spec = intensity_data[random_scan_idx, random_slit_idx, :]
            else:
                # Already 1D, use as is
                pixel_spec = intensity_data

            ax.plot(wl, pixel_spec, '--', alpha=0.5, linewidth=0.8)

        # Add single legend entry for dashed lines
        ax.plot([], [], '--', alpha=0.5, linewidth=0.8, label='Random pixel')

        ax.set_title(f"{line.upper()} Scan-Averaged Spectrum")
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return intensity, wavelength

def display_one_position(position: str, crop=False):
    """
    Display intensity data for a single position for both lines.

    Args:
        position: Disk position ('disk_center', '-40', etc.)
        crop_sr: Optional crop specification for SR data (dict or array format)
        crop_ti: Optional crop specification for TI data (dict or array format)
    """
     # Determine crop configuration for this line
    line_crop = CROP_CONFIGS['sr'] if crop else None
    sr = calculate_intensity('sr', position, line_crop)
    line_crop = CROP_CONFIGS['ti'] if crop else None
    ti = calculate_intensity('ti', position, line_crop)
    data_plot = np.array([sr, ti])
    viewer = display_data( data_plot, ['states', 'spatial_y', 'spatial_x', 'spectral'],
                          title=position,
                          state_names=['Sr', 'Ti']
                          )

def run_all_preparation_steps(crop=True):

    intensity, wavelength = calculate_intensity_for_all(crop=crop)  # Use line-dependent crop configs
    split_data = split_spectrum(intensity, wavelength, SPLIT_CONFIGS) # also normalized to continuum per position
    
    spectra = convert_to_spectrum(split_data) # also converted to IDL expected wavelength, slit, scan
    # Access: spectra['ti']['disk_center'].continuum.data
    # Access: spectra['ti']['disk_center'].continuum.wvl
    # spectra['ti']['disk_center'].plot()
    # spectra['ti']['disk_center'].plot(scan_idx=10,slit_idx=300)
    spectra.save_all()
    return spectra

def run_all_fitting(spectra):
    """
    Fit and save all spectra in the container, except disk_center positions.

    Args:
        spectra: SpectrumContainer with spectra to fit
    """
    for line_name, positions in spectra.items():
        for position_name, spectrum in positions.items():
            if position_name == 'disk_center':
                print(f"Skipping {line_name} disk_center")
                continue

            print(f"Fitting {line_name} {position_name}...")
            try:
                spectrum.fit()
                spectrum.save()
                print(f"  Saved {line_name} {position_name}")
            except Exception as e:
                print(f"  Error fitting {line_name} {position_name}: {e}")


#%%
if __name__ == '__main__':

    # Use predefined crop configurations (easier to read and modify)

    # check out one position
    #display_one_position('m40', crop=True)
    
    # Prepare data into a SpectrumContainer class holding all information
    # spectra = run_all_preparation_steps(crop=True)
    # check out mean intensity
    # spectra['ti']['disk_center'].plot()

    # once the preparation is finished, you can just load the prepared data
    spectra = SpectrumContainer.load_all()

    # Fit Ti line for a specific spectrum
    spectrum = spectra['ti']['m30']
    #spectrum.fit(slit=0, scan=0)
    # Plot with fit results
    #spectrum.plot(fit=True, show_levels=True, slit=0, scan=0)
    spectrum.fit()
    spectrum.save()

    #Fit all other positions and lines in external terminal
    # uv run python /home/franziskaz/code/themis/scripts/process_formation_height_line_levels.py
    #run_all_fitting(spectra) # uncomment for running
# -----------------------------------------------------


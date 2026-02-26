#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 11:34:04 2025

@author: franziskaz

determine spatial scale of spectral cameras by using GREGOR/HiFI images

"""

import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
import sys


# Add path to import fitting functions
sys.path.append('/home/franziskaz/code/themis/tests')
from determine_image_shifts import fit_gaussians_along_axis, plot_gaussian_slice_fits

def load_context_image(filepath):
    """Load the context FITS file and return data + header."""
    print(f"Loading: {filepath}")
    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype('float32')
        header = hdul[0].header
    print(f"Image shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min/Max values: {data.min():.2f} / {data.max():.2f}")
    return data, header

def create_cutout(data, center=None, size=(300, 300)):
    """Create a cutout around a center point or from center of image."""
    ny, nx = data.shape
    
    if center is None:
        # Use center of image
        cy, cx = ny // 2, nx // 2
    else:
        cy, cx = center
    
    sy, sx = size
    y_start = max(0, cy - sy // 2)
    y_end = min(ny, cy + sy // 2)
    x_start = max(0, cx - sx // 2)
    x_end = min(nx, cx + sx // 2)
    
    cutout = data[y_start:y_end, x_start:x_end]
    print(f"Cutout shape: {cutout.shape} from ({y_start}:{y_end}, {x_start}:{x_end})")
    
    return cutout, (y_start, x_start)

def analyze_spatial_scale(data):
    """Analyze spatial scale using Gaussian fitting (vertical only)."""
    print("\n=== Analyzing spatial scale (vertical only) ===")
    
    # Fit along axis 0 (vertical)
    print("\nFitting along axis 0 (vertical slices)...")
    centers_v, fwhm_v, model_v, coords_v, fit_axis_v, sigmas_v = fit_gaussians_along_axis(
        data, axis=0, fit_min=None, fit_max=None
    )
    
    return {
        'vertical': {
            'centers': centers_v,
            'fwhm': fwhm_v,
            'model': model_v,
            'coords': coords_v,
            'fit_axis': fit_axis_v,
            'sigmas': sigmas_v
        }
    }

def main_target():
    """Main function to perform spatial scale determination."""
    
    # Step 1: Load the context image
    filepath = "/home/franziskaz/themis/data/rdata/2025-07-05/context/z01_20250705_11052254_b3.fits"
    data, header = load_context_image(filepath)
    
    # Step 2: Create a cutout for analysis
    # You can adjust the center and size as needed
    cutout, offset = create_cutout(data.mean(axis=0), center=(300,800), size=(400, 400))
    
    # Step 3: Analyze spatial scale using Gaussian fitting
    results = analyze_spatial_scale(cutout)
    
    # Step 4: Print summary statistics
    print("\n=== Spatial Scale Summary ===")
    fwhm_pixels = np.nanmean(results['vertical']['fwhm'])
    print(f"Vertical direction:")
    print(f"  Mean FWHM: {fwhm_pixels:.2f} pixels")
    print(f"  Std FWHM: {np.nanstd(results['vertical']['fwhm']):.2f} pixels")
    print(f"  Min/Max FWHM: {np.nanmin(results['vertical']['fwhm']):.2f} / {np.nanmax(results['vertical']['fwhm']):.2f} pixels")
    
    # Step 5: Create visualization plots using the same style as determine_image_shifts
    # Use plot_gaussian_slice_fits for vertical fitting only
    fig, axes = plot_gaussian_slice_fits(
        cutout, 
        axis=0,  # vertical fitting (slices along y, fit along x)
        fit_min=None, 
        fit_max=None
    )
    
    # Additional plots: FWHM and center positions as functions of pixel
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # FWHM as function of pixel position
    axes2[0].plot(results['vertical']['fwhm'], range(len(results['vertical']['fwhm'])), 'b-', linewidth=2)
    axes2[0].set_xlabel('FWHM [pixels]')
    axes2[0].set_ylabel('Y position [pixels]')
    axes2[0].set_title('FWHM vs Y Position')
    axes2[0].grid(True, alpha=0.3)
    
    # Center position as function of pixel position
    axes2[1].plot(results['vertical']['centers'], range(len(results['vertical']['centers'])), 'r-', linewidth=2)
    axes2[1].set_xlabel('Center position [pixels]')
    axes2[1].set_ylabel('Y position [pixels]')
    axes2[1].set_title('Gaussian Centers vs Y Position')
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    # reconstructed image:
    # Step 1: load reconstructed THEMIS context (630 nm) and HiFI TiO (705 nm) --> difference of 5% in terms of size
    
    #themis = iio.imread('/home/franziskaz/projects/themis/images/context-2025-07-05/deppng/z01_20250705_11022611_3.5s3k5ip100fr_decpsf.png')


    
    return data, cutout, results

if __name__ == "__main__":
    data, cutout, results = main_target() 
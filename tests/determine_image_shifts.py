#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 10:46:34 2025

@author: franziskaz
"""

#!/usr/bin/env python3
"""
Determine image transformation between upper and lower image.

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import shift as scipy_shift
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio
from spectroflat.smile import SmileInterpolator, OffsetMap
from spectroflat.utils.processing import MP

from spectator.controllers.app_controller import display_data # from spectator

import imreg_dft as ird

from themis.core import themis_data_reduction as tdr

#%%

def plot_alignment_diff(upper_data, lower_data_aligned, 
                        tvec, angle, scale, config, margin=50):
    """
    Plot differences between upper and aligned lower frames.
    
    Parameters
    ----------
    upper_data : ndarray
        L0 upper frame
    lower_data_aligned : ndarray
        L0 lower frame after alignment
    corrected_upper : ndarray
        Corrected upper frame
    corrected_lower_aligned : ndarray
        Corrected lower frame after alignment
    tvec : tuple
        Translation vector (y, x)
    angle : float
        Rotation angle in degrees
    scale : float
        Scale factor
    config : Config
        Configuration object
    margin : int
        Margin for percentile calculation
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    from datetime import datetime
    
    fig, axes = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle(f'Upper vs Lower Frame Alignment\nTranslation: ({tvec[0]:.2f}, {tvec[1]:.2f}) px, Rotation: {angle:.3f}°, Scale: {scale:.5f}', 
                 fontsize=14, fontweight='bold')
    
    # Left: L0 difference (upper - aligned lower)
    diff_l0 = upper_data - lower_data_aligned
    if margin > 0:
        vmax_l0 = np.percentile(np.abs(diff_l0[margin:-margin, margin:-margin]), 99)
    else:
        vmax_l0 = np.percentile(np.abs(diff_l0), 99)
    
    ax = axes[0]
    im = ax.imshow(diff_l0, aspect='auto', cmap='RdBu_r', vmin=-vmax_l0, vmax=vmax_l0)
    ax.set_title('L0: Upper - Aligned Lower')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Difference')
    

    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures_dir = Path(config.directories.figures)
    figures_dir.mkdir(exist_ok=True, parents=True)
    #output_plot = figures_dir / f'frame_alignment_diff_{timestamp}.png'
    #plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    #print(f"\n✓ Saved alignment difference plot: {output_plot}")
    
    plt.show()
    
    return fig



def fit_gaussians_along_axis(image, axis=1, fit_min=None, fit_max=None,
                             initial_sigma=5.0, initial_amplitude=None,
                             initial_center=None, initial_offset=None):
    """Fit a 1D Gaussian to each slice of a 2D image along a chosen axis.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    axis : int, {0,1}
        Axis along which to take slices:
        - axis=0: slice index is y, Gaussian is fitted along x
        - axis=1: slice index is x, Gaussian is fitted along y
    fit_min, fit_max : int or None
        Pixel indices along the *fitted* axis where the Gaussian will be fit.
        If None, use the full range.

    Returns
    -------
    centers : 1D ndarray
        Fitted Gaussian centers (pixel coordinates along the fit axis).
    sigmas : 1D ndarray
        Fitted Gaussian sigmas.
    model_image : 2D ndarray
        Image reconstructed from the fitted Gaussians.
    coord_along_slices, coord_fit_axis : 1D ndarrays
        Pixel coordinates along the slice index and fit axis, respectively.
    """

    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("image must be 2D")

    # Bring the fit axis to axis=1: (n_slices, n_fit)
    if axis == 0:
        slices = img
    elif axis == 1:
        slices = img.T
    else:
        raise ValueError("axis must be 0 or 1")

    n_slices, n_fit = slices.shape

    if fit_min is None:
        fit_min = 0
    if fit_max is None or fit_max > n_fit:
        fit_max = n_fit

    x = np.arange(fit_min, fit_max)
    centers = np.full(n_slices, np.nan, dtype=float)
    sigmas = np.full(n_slices, np.nan, dtype=float)

    model_slices = np.zeros_like(slices, dtype=float)

    def gaussian(x, A, x0, sigma, C):
        return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + C

    for i in range(n_slices):
        y = slices[i, fit_min:fit_max].astype(float)
        if not np.any(np.isfinite(y)):
            continue

        # Initial guesses: decide if feature is a peak or a dip
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        ymed = np.nanmedian(y)

        if initial_amplitude is None:
            # Compare distance of max/min from median to decide sign
            if (ymax - ymed) >= (ymed - ymin):
                # Positive peak
                A0 = ymax - ymed
            else:
                # Negative dip
                A0 = ymin - ymed  # negative amplitude
        else:
            A0 = initial_amplitude

        # Center guess: user-provided or from max/min
        if initial_center is not None:
            x0_0 = float(initial_center)
        else:
            if A0 >= 0:
                x0_0 = x[np.nanargmax(y)]
            else:
                x0_0 = x[np.nanargmin(y)]

        # Offset guess: user-provided or median
        C0 = ymed if initial_offset is None else float(initial_offset)

        p0 = (A0, x0_0, initial_sigma, C0)

        try:
            # Constrain parameters: sigma>=0, center within fit window.
            # Amplitude A is allowed to be positive or negative so we can
            # fit both emission-like peaks and absorption-like dips.
            popt, _ = curve_fit(
                gaussian,
                x,
                y,
                p0=p0,
                # bounds=([-np.inf, fit_min, 0.0, -np.inf],
                #         [ np.inf, fit_max, np.inf,  np.inf]),
                maxfev=5000,
            )
            A, x0, sigma, C = popt
            centers[i] = x0
            sigmas[i] = abs(sigma)

            # Build fitted slice over full length
            x_full = np.arange(n_fit)
            model_slices[i, :] = gaussian(x_full, A, x0, sigma, C)
        except Exception:
            continue

    # Transform model_slices back to original orientation
    if axis == 0:
        model_image = model_slices
        coord_along_slices = np.arange(img.shape[0])
        coord_fit_axis = np.arange(img.shape[1])
    else:
        model_image = model_slices.T
        coord_along_slices = np.arange(img.shape[1])
        coord_fit_axis = np.arange(img.shape[0])

    return centers, sigmas, model_image, coord_along_slices, coord_fit_axis


def plot_gaussian_slice_fits(image, axis=1, fit_min=None, fit_max=None,
                             initial_sigma=5.0, initial_amplitude=None,
    initial_center=None, initial_offset=None,
                             cmap='viridis'):
    """Fit Gaussians along `axis` and create a 4-panel diagnostic plot.

    Panels:
      1. Input image
      2. Fitted-model image
      3. Input image with Gaussian centers overplotted
      4. Input image with ±1σ curves overplotted
    """

    (centers, sigmas, model_image,
     coord_along_slices, coord_fit_axis) = fit_gaussians_along_axis(
        image,
        axis=axis,
        fit_min=fit_min,
        fit_max=fit_max,
        initial_sigma=initial_sigma,
        initial_amplitude=initial_amplitude,
        initial_center=initial_center,
        initial_offset=initial_offset,
    )

    img = np.asarray(image)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel 1: input image
    im1 = ax1.imshow(img, origin='lower', aspect='auto', cmap=cmap)
    ax1.set_title("Input image")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 2: fitted model image
    im2 = ax2.imshow(model_image, origin='lower', aspect='auto', cmap=cmap)
    ax2.set_title("Fitted Gaussian model image")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Prepare center and ±1σ coordinates
    if axis == 0:
        x_centers = centers
        y_centers = coord_along_slices
        x_minus = centers - sigmas
        x_plus = centers + sigmas
        y_line = coord_along_slices
    else:
        x_centers = coord_along_slices
        y_centers = centers
        x_line = coord_along_slices
        y_minus = centers - sigmas
        y_plus = centers + sigmas

    # Panel 3: centers over input image
    ax3.imshow(img, origin='lower', aspect='auto', cmap=cmap)
    ax3.plot(x_centers, y_centers, 'r.', ms=3, label='Gaussian center')
    ax3.set_title("Centers of fitted Gaussians")
    ax3.legend(loc='best')

    # Panel 4: ±1σ lines over input image
    ax4.imshow(img, origin='lower', aspect='auto', cmap=cmap)
    if axis == 0:
        ax4.plot(x_minus, y_line, 'r--', lw=1, label='center - 1σ')
        ax4.plot(x_plus,  y_line, 'b--', lw=1, label='center + 1σ')
    else:
        ax4.plot(x_line, y_minus, 'r--', lw=1, label='center - 1σ')
        ax4.plot(x_line, y_plus,  'b--', lw=1, label='center + 1σ')
    ax4.set_title("±1σ envelopes of fitted Gaussians")
    ax4.legend(loc='best')

    for ax in axes.ravel():
        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')

    # ------------------------------------------------------------
    # Additional 1D diagnostic: first slice data vs fitted Gaussian
    # ------------------------------------------------------------
    # Determine local fit window used for the 1D plot
    if axis == 0:
        n_fit = img.shape[1]
    else:
        n_fit = img.shape[0]

    fit_min_loc = 0 if fit_min is None else fit_min
    fit_max_loc = n_fit if fit_max is None or fit_max > n_fit else fit_max
    x_window = np.arange(fit_min_loc, fit_max_loc)

    if axis == 0:
        data_1d = img[0, fit_min_loc:fit_max_loc]
        model_1d = model_image[0, fit_min_loc:fit_max_loc]
    else:
        data_1d = img[fit_min_loc:fit_max_loc, 0]
        model_1d = model_image[fit_min_loc:fit_max_loc, 0]

    fig_slice, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x_window, data_1d, 'k.', label='data (slice 0)')
    ax.plot(x_window, model_1d, 'r-', label='Gaussian fit')

    # Also show the initial Gaussian guess used for the fit (slice 0)
    # Reconstruct the same initial p0 logic for this slice.
    y = data_1d.astype(float)
    if np.any(np.isfinite(y)):
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        ymed = np.nanmedian(y)

        if initial_amplitude is None:
            if (ymax - ymed) >= (ymed - ymin):
                A0 = ymax - ymed
            else:
                A0 = ymin - ymed
        else:
            A0 = initial_amplitude

        if initial_center is not None:
            x0_0 = float(initial_center)
        else:
            # Note: x_window is in the global fit-axis coordinates
            if A0 >= 0:
                x0_0 = x_window[np.nanargmax(y)]
            else:
                x0_0 = x_window[np.nanargmin(y)]

        C0 = ymed if initial_offset is None else float(initial_offset)

        def _gauss(x, A, x0, sigma, C):
            return A * np.exp(-0.5 * ((x - x0) / sigma) ** 2) + C

        y_init = _gauss(x_window, A0, x0_0, initial_sigma, C0)
        ax.plot(x_window, y_init, 'b--', label='initial guess')

    ax.set_xlabel('Pixel along fit axis')
    ax.set_ylabel('Intensity')
    ax.set_title('First slice: data and fitted Gaussian')
    ax.legend(loc='best')

    return fig, (centers, sigmas)


def reduce_all(config):
    lv0 = tdr.reduction_levels["l0"]
    
    data_types = ['dark', 'flat', 'flat_center', 'scan']
    for data_type in data_types:
       result = lv0.reduce(config, data_type=data_type, return_reduced = False)

    # lv1 = tdr.reduction_levels["l1"]
    # result = lv1.reduce(config, data_type='flat_center', return_reduced = False)
    return(0)
   
#%%

if __name__ == "__main__":
    """Main test function."""
    print("="*70)
    print("Calibration data: determine image shifts, rotation, scale")
    print("="*70)
    
    # Load configuration
    config = get_config(
        config_path='configs/calibration_data_sr_2025-07-05.toml',
        auto_discover_files=True
    )
    
    # reduce (only once)
    #result = reduce_all(config)
  #%%  
    scan, header = tio.read_any_file(config, 'scan', verbose=False, status='l1')
    pQ_scan = scan.get_state('pQ')
    upper_scan = pQ_scan.stack_all('upper')
    lower_scan = pQ_scan.stack_all('lower')
    
    #%%
    fig, (centers, sigmas) = plot_gaussian_slice_fits(
    upper_scan[0],
    axis=1,        # slices along y, fit along x
    fit_min=498,
    fit_max=532,initial_sigma=5, initial_amplitude=-8000,
                          initial_center=515, initial_offset=8700
)
    plt.show()
    
#%%
        # Crop edges to avoid border effects
    margin = 50
    upper_scan_norm = (upper_scan[0] / np.mean(upper_scan[0]))[margin:-margin, margin:-margin]
    lower_scan_norm = (lower_scan[0] / np.mean(lower_scan[0]))[margin:-margin, margin:-margin]
    # result = ird.similarity(
    #        upper_scan_norm, 
    #        lower_scan_norm, # preshifted for better results
    #        numiter=5,
    #        constraints={
    #        'tx': (-1,1),      # x translation bounds
    #        'ty': (-5, 5),      # y translation bounds  
    #        'angle': (-1,1),     # rotation bounds in degrees
    #        'scale': (0.99999, 1.00001),  # scale bounds
    #        }
    #    )
    lower_scan_aligned_rotated = ird.transform_img(lower_scan_norm,tvec=[-0.3,1.5], angle=-0.31, scale=1)
    #fig = ird.imreg.imshow(upper_scan_norm, lower_scan_norm, result['timg'], cmap ='plasma_r')
    plt.imshow(upper_scan_norm- gaussian_filter(lower_scan_aligned_rotated, sigma=1.3))
#%%
    data_plot = np.array([upper_scan[0],lower_scan[0] ])
    viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
                       title='target scan', 
                      state_names=['upper', 'lower'])
    
#%%

    
    # result = ird.similarity(
    #     upper_scan_norm, 
    #     lower_scan_norm, # preshifted for better results
    #     numiter=5,
    #     constraints={
    #     'tx': (-1,1),      # x translation bounds
    #     'ty': (-2, 2),      # y translation bounds  
    #     'angle': (-0.5,0.5),     # rotation bounds in degrees
    #     'scale': (0.9, 1.1),  # scale bounds
    #     }
    # )
    lower_scan_aligned = scipy_shift(lower_scan_norm, (0, -1), order=3, mode='constant')
    lower_scan_aligned_rotated = ird.transform_img(lower_scan_aligned,tvec=[0,0], angle=-0.5, scale=1)
    upper_scan_rotated = ird.transform_img(lower_scan_norm,tvec=[0,0], angle=-0.3, scale=1)
    # print(f"   ✓ Shift: ({result['tvec'][0]:.3f}, {result['tvec'][1]:.3f}) px")
    # print(f"     Rotation: {result['angle']:.4f}°")
    # print(f"     Scale: {result['scale']:.6f}")
   
    
    viewer = display_data( np.array([upper_scan_norm,lower_scan_aligned_rotated, upper_scan_norm-lower_scan_aligned_rotated]), 'states',  'spatial', 'spectral',
                       title='target scan', 
                      state_names=['upper', 'lower aligned', 'upper-lower aligned'])
    
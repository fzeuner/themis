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

def load_context_image(filepath, hdu=0):
    """Load the context FITS file and return data + header."""
    print(f"Loading: {filepath}")
    with fits.open(filepath) as hdul:
        data = hdul[hdu].data.astype('float32')
        header = hdul[hdu].header
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

def _normalize(img):
    """Normalize image to [0, 1] for display and correlation."""
    img = np.array(img, dtype=float)
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mx > mn:
        return (img - mn) / (mx - mn)
    return img - mn


def _resample_themis(data_themis, scale_themis, ref_shape, scale_ref):
    """Resample THEMIS image so its pixel scale matches the reference."""
    from scipy.ndimage import zoom as nd_zoom
    ratio = scale_themis / scale_ref  # pixels of THEMIS per pixel of reference
    zoomed = nd_zoom(data_themis, ratio, order=1)
    return zoomed


def _apply_transform(img, offset_x, offset_y, rotation_deg, scale, out_shape):
    """Shift, rotate and scale img, then crop/pad to out_shape."""
    from scipy.ndimage import rotate as nd_rotate, shift as nd_shift, zoom as nd_zoom
    # scale
    if scale != 1.0:
        img = nd_zoom(img, scale, order=1)
    # rotate
    if rotation_deg != 0.0:
        img = nd_rotate(img, rotation_deg, reshape=False, order=1, mode='constant', cval=np.nan)
    # shift
    img = nd_shift(img, [offset_y, offset_x], order=1, mode='constant', cval=np.nan)
    # crop or pad to out_shape
    oy, ox = out_shape
    cy, cx = img.shape
    # center crop/pad
    pad_y = max(0, oy - cy)
    pad_x = max(0, ox - cx)
    img = np.pad(img, ((pad_y//2, pad_y - pad_y//2), (pad_x//2, pad_x - pad_x//2)),
                 constant_values=np.nan)
    # crop if larger
    sy = (img.shape[0] - oy) // 2
    sx = (img.shape[1] - ox) // 2
    img = img[sy:sy+oy, sx:sx+ox]
    return img


def _cross_correlation(img_ref, img_mov):
    """Normalized cross-correlation coefficient (ignores NaNs)."""
    mask = np.isfinite(img_ref) & np.isfinite(img_mov)
    if mask.sum() < 10:
        return 0.0
    a = img_ref[mask] - img_ref[mask].mean()
    b = img_mov[mask] - img_mov[mask].mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def _interactive_alignment(data_ref, data_themis, pixel_scale_ref,
                            pixel_scale_themis_initial=0.096):
    """Interactive widget to align THEMIS image to a reference (GREGOR/HiFI).

    Sliders:
      - offset_x / offset_y : shift of THEMIS in pixels of the reference grid
      - scale     : pixel scale multiplier for THEMIS (final = initial * scale)
      - rotation  : rotation of THEMIS in degrees
      - alpha     : transparency of the THEMIS overlay

    The normalized cross-correlation is printed/displayed for every update.
    """
    from matplotlib.widgets import Slider

    # --- prepare reference image (2D) ---
    ref = data_ref.squeeze()
    if ref.ndim == 3:
        ref = ref.mean(axis=0)
    ref = _normalize(ref)

    # --- prepare THEMIS image (2D, mean over wavelength if needed) ---
    themis_raw = data_themis.squeeze()
    if themis_raw.ndim == 3:
        themis_raw = themis_raw.mean(axis=0)
    themis_raw = _normalize(themis_raw)

    # resample THEMIS to reference pixel scale as starting point
    themis_resampled = _resample_themis(themis_raw, pixel_scale_themis_initial,
                                        ref.shape, pixel_scale_ref)
    themis_resampled = _normalize(themis_resampled)

    out_shape = ref.shape

    # --- initial transform values ---
    init_offset_x = 0.0
    init_offset_y = 0.0
    init_scale    = 1.0
    init_rotation = 0.0
    init_alpha    = 0.5

    # --- build figure ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    plt.subplots_adjust(left=0.1, bottom=0.38, right=0.98, top=0.95)

    themis_t = _apply_transform(themis_resampled, init_offset_x, init_offset_y,
                                 init_rotation, init_scale, out_shape)

    im_ref   = ax.imshow(ref,      origin='lower', cmap='gray',  aspect='equal',
                         vmin=0, vmax=1, zorder=0)
    im_themis = ax.imshow(themis_t, origin='lower', cmap='gray',   aspect='equal',
                          alpha=init_alpha, vmin=0.2, vmax=0.8, zorder=1)

    cc0 = _cross_correlation(ref, themis_t)
    title = ax.set_title(f'Corr: {cc0:.4f}   |   pixel scale THEMIS: '
                         f'{pixel_scale_themis_initial * init_scale:.4f} arcsec/px')
    ax.set_xlabel('X [reference pixels]')
    ax.set_ylabel('Y [reference pixels]')

    # --- sliders ---
    max_shift = max(ref.shape) // 2

    ax_ox  = fig.add_axes([0.12, 0.28, 0.76, 0.03])
    ax_oy  = fig.add_axes([0.12, 0.23, 0.76, 0.03])
    ax_sc  = fig.add_axes([0.12, 0.18, 0.76, 0.03])
    ax_rot = fig.add_axes([0.12, 0.13, 0.76, 0.03])
    ax_al  = fig.add_axes([0.12, 0.08, 0.76, 0.03])

    sl_ox  = Slider(ax_ox,  'Offset X [px]', -max_shift, max_shift, valinit=init_offset_x, valstep=1)
    sl_oy  = Slider(ax_oy,  'Offset Y [px]', -max_shift, max_shift, valinit=init_offset_y, valstep=1)
    sl_sc  = Slider(ax_sc,  'Scale',          0.1,        2.0,       valinit=init_scale,    valstep=0.02)
    sl_rot = Slider(ax_rot, 'Rotation [deg]', -180,       180,       valinit=init_rotation, valstep=0.5)
    sl_al  = Slider(ax_al,  'Alpha',           0.0,        1.0,       valinit=init_alpha,    valstep=0.05)

    def _update(_):
        t = _apply_transform(themis_resampled,
                             sl_ox.val, sl_oy.val,
                             sl_rot.val, sl_sc.val, out_shape)
        im_themis.set_data(t)
        im_themis.set_alpha(sl_al.val)
        cc = _cross_correlation(ref, t)
        ps = pixel_scale_themis_initial * sl_sc.val
        title.set_text(f'Corr: {cc:.4f}   |   pixel scale THEMIS: {ps:.4f} arcsec/px')
        print(f'  offset=({sl_ox.val:.0f}, {sl_oy.val:.0f})  '
              f'scale={sl_sc.val:.3f}  rot={sl_rot.val:.1f}°  '
              f'→ pixel_scale={ps:.4f} arcsec/px  corr={cc:.4f}')
        fig.canvas.draw_idle()

    sl_ox.on_changed(_update)
    sl_oy.on_changed(_update)
    sl_sc.on_changed(_update)
    sl_rot.on_changed(_update)
    sl_al.on_changed(_update)

    plt.show()
    print(f'\nFinal pixel scale estimate: '
          f'{pixel_scale_themis_initial * sl_sc.val:.4f} arcsec/px')


def main_alignment():
    """Main function to perform alignment of GREGOR/HiFI with THEMIS."""

    # Step 1: Load the THEMIS context image (mean over wavelength)
    filepath = "/home/franziskaz/themis/data/rdata/2025-07-05/context/z01_20250705_11022611_b3.fits"
    data_themis, header = load_context_image(filepath)

    # Step 2: Load the two closest GREGOR/HiFI images
    pixel_scale_reference = 0.05  # arcsec per pixel (HiFI)
    filepath = "/home/franziskaz/themis/data/rdata/2025-07-05/sj_gregor_2025-07-05/hifiplus2_20250705_094857_sp.fts"
    data_g1, header_g1 = load_context_image(filepath, hdu=2)
    data_g1 = np.flip(data_g1)

    filepath = "/home/franziskaz/themis/data/rdata/2025-07-05/sj_gregor_2025-07-05/hifiplus2_20250705_100955_sp.fts"
    data_g2, header_g2 = load_context_image(filepath, hdu=2)
    data_g2 = np.flip(data_g2)

    # Step 3: Interactive alignment — choose which GREGOR image to use as reference
    print("\nStarting interactive alignment with data_g1 as reference.")
    print("Adjust sliders to match THEMIS to the GREGOR/HiFI image.")
    _interactive_alignment(data_g1, data_themis,
                           pixel_scale_ref=pixel_scale_reference,
                           pixel_scale_themis_initial=0.05)


if __name__ == "__main__":
    #data, cutout, results = main_target() 
    
    plot = main_alignment()
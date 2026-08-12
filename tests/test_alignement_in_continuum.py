#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script: spatial alignment check using two continuum regions per line.

For a given L4 scan sequence, this script extracts two "continuum" spatial
images per spectral line (Ti, Sr) by averaging over two disjoint spectral
pixel ranges that are known to be free of line absorption. It then uses
skimage.registration.phase_cross_correlation to measure the sub-pixel shift
between the two continuum images, separately for the upper (pQ) and lower
(mQ) detector halves. Comparing the pQ-derived shift with the mQ-derived
shift reveals whether there is a systematic alignment offset between the
two beams that depends on wavelength position within the passband.

Usage: run cell by cell 


This should resolve a test by Marianne that finds some shifts in the p image compared to
the m image
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from themis.datasets.themis_datasets_2025 import get_config, _build_file_set
from themis.core import themis_io as tio

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

CONFIG_PATH = 'configs/formation_dataset_2025-07-05.toml'

# Sequence to process. Start with sequence 18 (position m30, i.e. -30 deg).
SEQUENCE = 18

# Two disjoint continuum pixel ranges (spectral axis indices) per line.
# Ranges are [start, end] inclusive.
CONTINUUM_RANGES = {
    'ti': {
        'region_1': (103, 116),
        'region_2': (982, 1004),
    },
    'sr': {
        'region_1': (320, 337),
        'region_2': (801, 826),
    },
}

HALVES = ('upper', 'lower')  # upper -> pQ, lower -> mQ (paired per frame)
HALF_TO_POLSTATE = {'upper': 'pQ', 'lower': 'mQ'}

# Margin (pixels) trimmed from both ends of the slit direction to remove
# edge artifacts before computing profiles/shifts.
SLIT_MARGIN = 25

# Optional Fourier bandpass filter applied to the slit-direction profiles,
# defined as a spatial period range (period_low, period_high) in arcsec.
# Set to None to disable. Converted to pixels via config.cam.pixel_scale.
BANDPASS_ARCSEC = (0.1, 5.5) #None


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_l4_scan_cubes(line: str, sequence: int):
    """
    Load L4 scan data for a given line/sequence and return upper/lower cubes.

    Returns
    -------
    cubes : dict
        {'upper': ndarray (n_slit, spatial_y, spectral_x),
         'lower': ndarray (n_slit, spatial_y, spectral_x)}
    header : astropy FITS header of the underlying scan file (primary HDU)
    pixel_scale : float
        Spatial scale of the camera (arcsec/pixel), from config.cam.pixel_scale.
    """
    config = get_config(line=line, config_path=CONFIG_PATH,
                         auto_discover_files=False, auto_create_dirs=False)
    config.dataset['scan']['sequence'] = sequence
    config.dataset['scan']['files'] = _build_file_set(
        config.directories, config.dataset['scan'], config.data_types,
        config.reduction_levels, config.cam, config.line,
        yshift_calibration_date=config.yshift_calibration_date,
    )

    data, header = tio.read_any_file(config, data_type='scan', status='l4')

    states = sorted(set(k[0] for k in data.keys()))
    if len(states) != 1:
        print(f"  Warning: expected a single FRMSTATE, found {states}")

    cubes = {
        'upper': data.stack_all('upper'),  # (n_slit, spatial_y, spectral_x)
        'lower': data.stack_all('lower'),
    }
    return cubes, header, config.cam.pixel_scale


# ----------------------------------------------------------------------------
# Continuum image extraction and shift measurement
# ----------------------------------------------------------------------------

def continuum_image(cube: np.ndarray, px_range: tuple) -> np.ndarray:
    """
    Average the cube over a spectral pixel range to produce a 2D spatial image.

    Parameters
    ----------
    cube : ndarray, shape (n_slit, spatial_y, spectral_x)
    px_range : tuple (start, end), inclusive spectral pixel indices

    Returns
    -------
    ndarray, shape (n_slit, spatial_y)
    """
    lo, hi = px_range
    return cube[:, :, lo:hi + 1].mean(axis=2)


def apply_slit_margin(img: np.ndarray, margin: int = SLIT_MARGIN) -> np.ndarray:
    """
    Trim `margin` pixels from both ends of the slit direction (last axis)
    to remove edge artifacts.

    Parameters
    ----------
    img : ndarray, shape (..., spatial_y)
    margin : int

    Returns
    -------
    ndarray with the slit-direction axis trimmed by `margin` on both ends.
    """
    if margin <= 0:
        return img
    return img[..., margin:-margin]


def bandpass_arcsec_to_pixels(band_arcsec: tuple, pixel_scale: float) -> tuple:
    """
    Convert a spatial period range given in arcsec to a period range in pixels.

    Parameters
    ----------
    band_arcsec : tuple (period_low, period_high) in arcsec
    pixel_scale : float, arcsec/pixel

    Returns
    -------
    tuple (period_low_px, period_high_px) in pixels
    """
    period_low, period_high = band_arcsec
    return period_low / pixel_scale, period_high / pixel_scale


def fourier_bandpass_filter(data: np.ndarray, pixel_scale: float,
                             band_arcsec: tuple, axis: int = -1) -> np.ndarray:
    """
    Apply a Fourier bandpass filter along `axis`, keeping only spatial
    periods within `band_arcsec` (converted to pixels via `pixel_scale`).

    Parameters
    ----------
    data : ndarray
        Array to filter (e.g. a continuum image (n_slit, spatial_y)).
    pixel_scale : float, arcsec/pixel
    band_arcsec : tuple (period_low, period_high) in arcsec, or None to
        disable filtering (data returned unchanged).
    axis : int
        Axis along which to apply the filter (default: last axis, the
        slit direction).

    Returns
    -------
    ndarray, same shape as `data`, bandpass-filtered along `axis`.
    """
    if band_arcsec is None:
        return data

    period_low, period_high = band_arcsec
    if period_low <= 0 or period_high <= 0:
        raise ValueError(f"band_arcsec periods must be positive, got {band_arcsec}")
    if period_low >= period_high:
        raise ValueError(
            f"band_arcsec must be (period_low, period_high) with period_low < "
            f"period_high, got {band_arcsec}"
        )

    period_low_px, period_high_px = bandpass_arcsec_to_pixels(band_arcsec, pixel_scale)
    n = data.shape[axis]

    if period_low_px < 2:
        warnings.warn(
            f"bandpass period_low={period_low:.3f}\" ({period_low_px:.2f}px) is below "
            f"the Nyquist limit (2px = {2 * pixel_scale:.3f}\"); the effective high-"
            f"frequency cutoff will be capped at Nyquist instead."
        )
    if period_high_px > n:
        warnings.warn(
            f"bandpass period_high={period_high:.3f}\" ({period_high_px:.2f}px) exceeds "
            f"the slit length ({n}px = {n * pixel_scale:.3f}\"); no additional low-"
            f"frequency structure will be attenuated, but the DC/mean component is "
            f"still removed by this filter."
        )

    freq_low = 1.0 / period_high_px
    freq_high = 1.0 / period_low_px

    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = (freqs >= freq_low) & (freqs <= freq_high)

    fft_vals = np.fft.rfft(data, axis=axis)
    shape = [1] * data.ndim
    shape[axis] = mask.shape[0]
    fft_vals = fft_vals * mask.reshape(shape)
    return np.fft.irfft(fft_vals, n=n, axis=axis)


def slit_profile(continuum_img: np.ndarray) -> np.ndarray:
    """
    Average a continuum image over the scan (slit-position/raster) axis to
    obtain a 1D profile along the physical slit direction.

    Parameters
    ----------
    continuum_img : ndarray, shape (n_slit, spatial_y)

    Returns
    -------
    ndarray, shape (spatial_y,)
    """
    return continuum_img.mean(axis=0)


def measure_shift(profile1: np.ndarray, profile2: np.ndarray):
    """
    Measure the sub-pixel shift between two 1D slit profiles using phase
    cross-correlation.

    Returns
    -------
    shift : float, shift along the slit direction (pixels)
    error : float
    diffphase : float
    """
    shift, error, diffphase = phase_cross_correlation(profile1, profile2, upsample_factor=50)
    return float(shift[0]), error, diffphase


def measure_shift_vs_scan_position(img1: np.ndarray, img2: np.ndarray):
    """
    Measure the shift along the slit direction independently for every scan
    (raster) position, instead of averaging over all scan positions first.

    Parameters
    ----------
    img1, img2 : ndarray, shape (n_slit, spatial_y)
        Continuum images (already averaged over the spectral pixel range).

    Returns
    -------
    shifts : ndarray, shape (n_slit,)
        Shift along the slit direction (pixels) for each scan position.
    errors : ndarray, shape (n_slit,)
    """
    n_slit = img1.shape[0]
    shifts = np.full(n_slit, np.nan)
    errors = np.full(n_slit, np.nan)
    for i in range(n_slit):
        shift, error, _ = phase_cross_correlation(img1[i], img2[i], upsample_factor=50)
        shifts[i] = shift[0]
        errors[i] = error
    return shifts, errors


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------

def analyze_line(line: str, sequence: int, show_plots: bool = True,
                  bandpass_arcsec: tuple = BANDPASS_ARCSEC):
    print(f"\n{'='*70}")
    print(f"LINE: {line.upper()}  |  SEQUENCE: {sequence}")
    print(f"{'='*70}")

    cubes, header, pixel_scale = load_l4_scan_cubes(line, sequence)
    ranges = CONTINUUM_RANGES[line]

    bandpass_px = None
    if bandpass_arcsec is not None:
        bandpass_px = bandpass_arcsec_to_pixels(bandpass_arcsec, pixel_scale)
        print(f"  Bandpass: {bandpass_arcsec[0]:.3f}-{bandpass_arcsec[1]:.3f} arcsec "
              f"({bandpass_px[0]:.2f}-{bandpass_px[1]:.2f} px, "
              f"pixel_scale={pixel_scale:.4f} arcsec/px)")

    results = {}
    for half in HALVES:
        cube = cubes[half]
        img1 = continuum_image(cube, ranges['region_1'])
        img2 = continuum_image(cube, ranges['region_2'])
        img1 = apply_slit_margin(img1)
        img2 = apply_slit_margin(img2)
        img1 = fourier_bandpass_filter(img1, pixel_scale, bandpass_arcsec)
        img2 = fourier_bandpass_filter(img2, pixel_scale, bandpass_arcsec)
        profile1 = slit_profile(img1)
        profile2 = slit_profile(img2)

        shift, error, diffphase = measure_shift(profile1, profile2)
        shifts_per_scan, errors_per_scan = measure_shift_vs_scan_position(img1, img2)

        pol_state = HALF_TO_POLSTATE[half]
        results[half] = {
            'shift': shift,
            'error': error,
            'diffphase': diffphase,
            'img1': img1,
            'img2': img2,
            'profile1': profile1,
            'profile2': profile2,
            'shifts_per_scan': shifts_per_scan,
            'errors_per_scan': errors_per_scan,
        }
        print(f"  {half} ({pol_state}): shift along slit = {shift:+.4f} px, "
              f"error={error:.5f} "
              f"(per-scan mean={np.nanmean(shifts_per_scan):+.4f} px, "
              f"std={np.nanstd(shifts_per_scan):.4f} px)")

    # Difference between the two halves' shifts
    d_shift = results['upper']['shift'] - results['lower']['shift']
    print(f"  --> shift difference (upper - lower): {d_shift:+.4f} px")

    if bandpass_arcsec is not None:
        bp_str = (f", bandpass={bandpass_arcsec[0]:.2f}-{bandpass_arcsec[1]:.2f}\" "
                  f"({bandpass_px[0]:.1f}-{bandpass_px[1]:.1f}px)")
    else:
        bp_str = ""

    if show_plots:
        n_scan = cubes['upper'].shape[0]
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for ax, half in zip(axes, HALVES):
            r = results[half]
            ax.plot(r['profile1'], label='region 1')
            ax.plot(r['profile2'], label='region 2')
            ax.set_title(
                f"{line.upper()} seq={sequence}: {half} ({HALF_TO_POLSTATE[half]}) "
                f"slit profile (mean over {n_scan} scan positions, "
                f"margin={SLIT_MARGIN}px{bp_str})"
            )
            ax.set_xlabel('spatial_y (slit direction, px, margin-cropped)')
            ax.set_ylabel('mean intensity')
            ax.legend()
        plt.tight_layout()
        plt.show()

        # Shift vs. scan position for this line
        fig, ax = plt.subplots(figsize=(9, 4))
        for half in HALVES:
            r = results[half]
            color = ax._get_lines.get_next_color()
            ax.plot(r['shifts_per_scan'], marker='.', linestyle='-',
                    color=color,
                    label=f"{half} ({HALF_TO_POLSTATE[half]})")
            ax.axhline(r['shift'], linestyle='--', alpha=0.5, color=color,
                       label=f"{half} avg-first: {r['shift']:+.4f} px")
            mean_per_scan = np.nanmean(r['shifts_per_scan'])
            std_per_scan = np.nanstd(r['shifts_per_scan'])
            ax.axhline(mean_per_scan, linestyle=':', alpha=0.5, color=color,
                       label=f"{half} per-scan mean: {mean_per_scan:+.4f} \u00b1 {std_per_scan:.4f} px")
        ax.set_xlabel('scan (slit) index')
        ax.set_ylabel('shift along slit direction (px)')
        ax.set_title(f"{line.upper()} seq={sequence}: shift vs. scan position{bp_str}")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return results

#%%
if __name__ == '__main__':
    #%%
    all_results = {}
    for line in ('ti', 'sr'):
        all_results[line] = analyze_line(line, SEQUENCE, show_plots=False)

    #%%
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for line, res in all_results.items():
        for half in HALVES:
            s = res[half]['shift']
            print(f"{line.upper():4s} {half:5s} ({HALF_TO_POLSTATE[half]}): "
                  f"shift={s:+.4f} px, error={res[half]['error']:.5f}")

    #%%
    # Combined summary plot: shift vs. scan position for all lines / halves
    fig, axes = plt.subplots(len(all_results), 1, figsize=(9, 4 * len(all_results)),
                              sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (line, res) in zip(axes, all_results.items()):
        for half in HALVES:
            r = res[half]
            color = ax._get_lines.get_next_color()
            ax.plot(r['shifts_per_scan'], marker='.', linestyle='-',
                    color=color,
                    label=f"{half} ({HALF_TO_POLSTATE[half]})")
            ax.axhline(r['shift'], linestyle='--', alpha=0.5, color=color,
                       label=f"{half} avg-first: {r['shift']:+.4f} px")
            mean_per_scan = np.nanmean(r['shifts_per_scan'])
            std_per_scan = np.nanstd(r['shifts_per_scan'])
            ax.axhline(mean_per_scan, linestyle=':', alpha=0.5, color=color,
                       label=f"{half} per-scan mean: {mean_per_scan:+.4f} \u00b1 {std_per_scan:.4f} px")
        ax.set_ylabel('shift along slit (px)')
        if BANDPASS_ARCSEC is not None:
            bp_str = f", bandpass={BANDPASS_ARCSEC[0]:.2f}-{BANDPASS_ARCSEC[1]:.2f}\""
        else:
            bp_str = ""
        ax.set_title(f"{line.upper()} seq={SEQUENCE}: shift vs. scan position{bp_str}")
        ax.legend()
    axes[-1].set_xlabel('scan (slit) index')
    plt.tight_layout()
    plt.show()

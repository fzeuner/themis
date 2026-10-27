#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script: inter-camera alignment check using disk-center continuum.

For a given L4 scan sequence (disk center), this script extracts a "blue"
and a "red" continuum spatial image for each of the two cameras (Ti, Sr) by
averaging over spectral pixel ranges that are known to be free of line
absorption. It then uses skimage.registration.phase_cross_correlation to
measure the sub-pixel shift, along the slit direction, between the Ti and
the Sr camera for each continuum region (blue, red), separately for the
upper (pQ) and lower (mQ) detector halves.

The shift is computed within an optional Fourier bandpass (applied along
the slit direction) and independently for every scan (slit) index; the
final reported shift is the mean of these per-slit-index shifts.

Comparing the blue-derived Ti-vs-Sr shift with the red-derived Ti-vs-Sr
shift reveals whether there is a systematic alignment offset between the
two cameras that depends on wavelength position within the passband.

Usage: run cell by cell

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

# Sequence to process. Start with sequence 27 (disk center).
SEQUENCE = 27

# Two disjoint continuum pixel ranges (spectral axis indices) per line/camera.
# Ranges are [start, end] inclusive. 'region_1' is the blue continuum region,
# 'region_2' is the red continuum region.
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

REGION_LABELS = {'region_1': 'blue', 'region_2': 'red'}

HALVES = ('upper', 'lower')  # upper -> pQ, lower -> mQ (paired per frame)
HALF_TO_POLSTATE = {'upper': 'pQ', 'lower': 'mQ'}

# Margin (pixels) trimmed from both ends of the slit direction to remove
# edge artifacts before computing profiles/shifts.
SLIT_MARGIN = 25

# Optional Fourier bandpass filter applied to the slit-direction profiles,
# defined as a spatial period range (period_low, period_high) in arcsec.
# Set to None to disable. Converted to pixels via config.cam.pixel_scale.
BANDPASS_ARCSEC = (0.3, 8.5) #None

# Per-slit-index shifts with an absolute value above this threshold (pixels)
# are treated as outliers (e.g. failed cross-correlations) and discarded
# before averaging over slit indices. Set to None to disable.
MAX_ABS_SHIFT_PX = 10.0


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


def measure_shift_vs_scan_position(img1: np.ndarray, img2: np.ndarray):
    """
    Measure the shift along the slit direction independently for every scan
    (raster) position.

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


def load_both_cameras(sequence: int):
    """
    Load the L4 scan cubes for both cameras (Ti, Sr).

    Returns
    -------
    cubes : dict
        {'ti': {'upper': ndarray, 'lower': ndarray}, 'sr': {'upper': ndarray, 'lower': ndarray}}
    pixel_scales : dict
        {'ti': float, 'sr': float}, arcsec/pixel
    """
    cubes = {}
    pixel_scales = {}
    for line in ('ti', 'sr'):
        cubes[line], _header, pixel_scales[line] = load_l4_scan_cubes(line, sequence)
    return cubes, pixel_scales


def reject_outlier_shifts(shifts: np.ndarray, max_abs_shift: float) -> np.ndarray:
    """
    Mask per-slit-index shifts whose absolute value exceeds `max_abs_shift`.

    Parameters
    ----------
    shifts : ndarray, shape (n_slit,)
    max_abs_shift : float or None
        Shifts with |shift| > max_abs_shift are replaced with NaN. If None,
        `shifts` is returned unchanged.

    Returns
    -------
    ndarray, shape (n_slit,), with outlier entries set to NaN.
    """
    if max_abs_shift is None:
        return shifts
    cleaned = shifts.copy()
    cleaned[np.abs(cleaned) > max_abs_shift] = np.nan
    return cleaned


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------

def analyze_sequence(sequence: int, show_plots: bool = True,
                      bandpass_arcsec: tuple = BANDPASS_ARCSEC,
                      max_abs_shift_px: float = MAX_ABS_SHIFT_PX):
    """
    Compare the Ti and Sr cameras for a given disk-center sequence.

    For each detector half (upper/pQ, lower/mQ) and each continuum region
    (blue=region_1, red=region_2), the continuum image is extracted from
    both cameras, trimmed, bandpass-filtered along the slit direction, and
    the shift between the Ti and Sr camera is measured independently for
    every slit (scan) index. Per-slit-index shifts with |shift| >
    `max_abs_shift_px` are discarded as outliers before averaging; the
    final shift is the mean over the remaining slit indices.
    """
    print(f"\n{'='*70}")
    print(f"SEQUENCE: {sequence}  |  Ti camera vs. Sr camera")
    print(f"{'='*70}")

    cubes, pixel_scales = load_both_cameras(sequence)

    results = {}
    for half in HALVES:
        for region_key, region_label in REGION_LABELS.items():
            img_ti = continuum_image(cubes['ti'][half], CONTINUUM_RANGES['ti'][region_key])
            img_sr = continuum_image(cubes['sr'][half], CONTINUUM_RANGES['sr'][region_key])
            img_ti = apply_slit_margin(img_ti)
            img_sr = apply_slit_margin(img_sr)
            img_ti = fourier_bandpass_filter(img_ti, pixel_scales['ti'], bandpass_arcsec)
            img_sr = fourier_bandpass_filter(img_sr, pixel_scales['sr'], bandpass_arcsec)

            shifts_per_scan, errors_per_scan = measure_shift_vs_scan_position(img_ti, img_sr)
            shifts_clean = reject_outlier_shifts(shifts_per_scan, max_abs_shift_px)
            n_rejected = np.sum(np.isnan(shifts_clean) & ~np.isnan(shifts_per_scan))
            shift = np.nanmean(shifts_clean)
            std = np.nanstd(shifts_clean)

            pol_state = HALF_TO_POLSTATE[half]
            results[(half, region_key)] = {
                'shift': shift,
                'std': std,
                'img_ti': img_ti,
                'img_sr': img_sr,
                'shifts_per_scan': shifts_per_scan,
                'shifts_per_scan_clean': shifts_clean,
                'errors_per_scan': errors_per_scan,
                'n_rejected': n_rejected,
            }
            reject_str = (f" ({n_rejected} of {len(shifts_per_scan)} slit indices rejected, "
                          f"|shift| > {max_abs_shift_px} px)" if max_abs_shift_px is not None else "")
            print(f"  {half} ({pol_state}), {region_label}: "
                  f"Ti-vs-Sr shift (mean over scans) = {shift:+.4f} \u00b1 {std:.4f} px"
                  f"{reject_str}")

    # Difference between the blue- and red-derived shifts, per half.
    for half in HALVES:
        d_shift = (results[(half, 'region_1')]['shift']
                   - results[(half, 'region_2')]['shift'])
        print(f"  --> {half} ({HALF_TO_POLSTATE[half]}) shift difference "
              f"(blue - red): {d_shift:+.4f} px")

    if bandpass_arcsec is not None:
        bp_str = f", bandpass={bandpass_arcsec[0]:.2f}-{bandpass_arcsec[1]:.2f}\""
    else:
        bp_str = ""

    if show_plots:
        fig, ax = plt.subplots(figsize=(9, 4))
        for half in HALVES:
            for region_key, region_label in REGION_LABELS.items():
                r = results[(half, region_key)]
                color = ax._get_lines.get_next_color()
                label_base = f"{half} ({HALF_TO_POLSTATE[half]}), {region_label}"
                ax.plot(r['shifts_per_scan'], marker='.', linestyle='-',
                        color=color, label=label_base)
                ax.axhline(r['shift'], linestyle=':', alpha=0.5, color=color,
                           label=f"{label_base} mean: {r['shift']:+.4f} \u00b1 {r['std']:.4f} px")
        ax.set_xlabel('scan (slit) index')
        ax.set_ylabel('Ti-vs-Sr shift along slit direction (px)')
        ax.set_title(f"seq={sequence}: Ti-vs-Sr shift per slit index{bp_str}")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return results


def plot_camera_shift_images(cubes: dict, half: str = 'upper',
                              region_key: str = 'region_2',
                              shifts_px: tuple = (-3, 11),
                              margin: int = SLIT_MARGIN):
    """
    Show, as separate rows in one figure, the Ti and Sr continuum images and
    their difference, with and without an applied slit-direction shift.

    Rows:
      - Ti continuum image
      - Sr continuum image
      - Sr - Ti ("original difference")
      - Sr - roll(Ti, s) for each s in `shifts_px` (labelled "shift:s")

    `np.roll` wraps values around (pixels shifted off one end re-appear at
    the other end), so it is applied to the *full*, un-cropped Ti continuum
    image along the slit axis; the `margin` is only applied afterwards, for
    display, so that wrapped-around values never show up in the displayed
    region.

    Parameters
    ----------
    cubes : dict
        {'ti': {'upper': ndarray, 'lower': ndarray}, 'sr': {...}}, as
        returned by `load_both_cameras`.
    half : str, 'upper' or 'lower'
    region_key : str, 'region_1' (blue) or 'region_2' (red)
    shifts_px : tuple of int
        Slit-direction shifts (pixels) to apply to the Ti image via
        `np.roll` before differencing with Sr.
    margin : int
        Slit-direction margin cropped (from both ends) for display.
    """
    ti_full = continuum_image(cubes['ti'][half], CONTINUUM_RANGES['ti'][region_key])
    sr_full = continuum_image(cubes['sr'][half], CONTINUUM_RANGES['sr'][region_key])

    ti_crop = apply_slit_margin(ti_full, margin)
    sr_crop = apply_slit_margin(sr_full, margin)

    region_label = REGION_LABELS[region_key]
    rows = [
        (f"Ti ({region_label})", ti_crop, False),
        (f"Sr ({region_label})", sr_crop, False),
        ("original difference (Sr - Ti)", sr_crop - ti_crop, True),
    ]
    for s in shifts_px:
        ti_rolled_crop = apply_slit_margin(np.roll(ti_full, s, axis=1), margin)
        rows.append((f"shift:{s} (Sr - roll(Ti,{s}))", sr_crop - ti_rolled_crop, True))

    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.3 * len(rows)),
                              sharex=True, sharey=True)
    for ax, (title, img, is_diff) in zip(axes, rows):
        if is_diff:
            vmax = np.nanmax(np.abs(img))
            im = ax.imshow(img, aspect='auto', origin='lower',
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(img, aspect='auto', origin='lower', cmap='gray')
        ax.set_title(title)
        ax.set_ylabel('scan index')
        fig.colorbar(im, ax=ax)
    axes[-1].set_xlabel(f'slit position (px, margin={margin} cropped)')
    fig.suptitle(f"half={half} ({HALF_TO_POLSTATE[half]}), region={region_key} ({region_label})")
    plt.tight_layout()
    plt.show()

#%%
if __name__ == '__main__':
    #%%
    results = analyze_sequence(SEQUENCE, show_plots=True)

    #%%
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for half in HALVES:
        for region_key, region_label in REGION_LABELS.items():
            r = results[(half, region_key)]
            print(f"{half:5s} ({HALF_TO_POLSTATE[half]}) {region_label:4s}: "
                  f"shift={r['shift']:+.4f} \u00b1 {r['std']:.4f} px")

    #%%
    # Combined summary plot: Ti-vs-Sr shift per slit index, for all halves/regions
    fig, ax = plt.subplots(figsize=(9, 5))
    for half in HALVES:
        for region_key, region_label in REGION_LABELS.items():
            r = results[(half, region_key)]
            color = ax._get_lines.get_next_color()
            label_base = f"{half} ({HALF_TO_POLSTATE[half]}), {region_label}"
            ax.plot(r['shifts_per_scan'], marker='.', linestyle='-',
                    color=color, label=label_base)
            ax.axhline(r['shift'], linestyle=':', alpha=0.5, color=color,
                       label=f"{label_base} mean: {r['shift']:+.4f} \u00b1 {r['std']:.4f} px")
    ax.set_xlabel('scan (slit) index')
    ax.set_ylabel('Ti-vs-Sr shift along slit (px)')
    if BANDPASS_ARCSEC is not None:
        bp_str = f", bandpass={BANDPASS_ARCSEC[0]:.2f}-{BANDPASS_ARCSEC[1]:.2f}\""
    else:
        bp_str = ""
    ax.set_title(f"seq={SEQUENCE}: Ti-vs-Sr shift per slit index{bp_str}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    #%%
    # Visualize Ti/Sr continuum images and their difference for a given
    # shift candidate (e.g. upper half, red continuum region).
    cubes, _pixel_scales = load_both_cameras(SEQUENCE)
    plot_camera_shift_images(cubes, half='upper', region_key='region_1',
                              shifts_px=(-3, 11))

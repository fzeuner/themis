#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic validation test for the shift-measurement method used in
test_alignement_in_continuum.py (measure_shift_vs_scan_position, based on
skimage.registration.phase_cross_correlation).

Uses the real Sr pQ (upper) "region_1" continuum image (from sequence
SEQUENCE) as a "reference" image, then builds synthetic test images with
KNOWN sub-pixel shifts applied along the slit direction:

1) Zero shift: the test image IS the reference (measured shift should be
   ~0 for every scan position).
2) Constant 0.05 px shift: every scan position is shifted by the same
   +0.05 px along the slit.
3) 0.05 px + random shift: every scan position is shifted by 0.05 px plus
   an independent random component drawn from U(-0.1, +0.1) px.

For each case the script applies the same measure_shift_vs_scan_position
routine used in the real analysis and compares the measured shifts against
the known ground truth (bias, std of residuals, RMS error).

Usage: run cell by cell.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import fourier_shift

sys.path.insert(0, str(Path(__file__).parent))

from test_alignement_in_continuum import (
    CONFIG_PATH, SEQUENCE, CONTINUUM_RANGES, SLIT_MARGIN,
    load_l4_scan_cubes, continuum_image, apply_slit_margin,
    measure_shift_vs_scan_position,
)

np.random.seed(0)


# ----------------------------------------------------------------------------
# Reference image
# ----------------------------------------------------------------------------

def load_reference_image() -> np.ndarray:
    """
    Load the real Sr pQ (upper) "region_1" (left) continuum image, margin-
    cropped, to be used as the ground-truth reference for synthetic shift
    tests.

    Returns
    -------
    ndarray, shape (n_slit, spatial_y)
    """
    cubes, header, pixel_scale = load_l4_scan_cubes('sr', SEQUENCE)
    ref = continuum_image(cubes['upper'], CONTINUUM_RANGES['sr']['region_1'])
    ref = apply_slit_margin(ref, SLIT_MARGIN)
    return ref


# ----------------------------------------------------------------------------
# Synthetic shifted images
# ----------------------------------------------------------------------------

def shift_rows(img: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Apply an independent sub-pixel shift along the slit direction (last
    axis) to each row (scan position) of `img`, using an FFT-based (Fourier
    shift theorem) shift. This is numerically exact for band-limited
    signals, unlike spline-based interpolation (scipy.ndimage.shift),
    which has its own sub-pixel bias for small shifts and would otherwise
    contaminate the ground truth used in this test.

    Parameters
    ----------
    img : ndarray, shape (n_slit, spatial_y)
    shifts : ndarray, shape (n_slit,)
        Shift (px) to apply to each row.

    Returns
    -------
    ndarray, shape (n_slit, spatial_y)
    """
    out = np.empty_like(img)
    for i in range(img.shape[0]):
        f = np.fft.fft(img[i])
        f_shifted = fourier_shift(f, shifts[i])
        out[i] = np.fft.ifft(f_shifted).real
    return out


def make_test_cases(reference: np.ndarray):
    """
    Build the three synthetic test cases described in the module docstring.

    Returns
    -------
    dict: {case_name: (test_img, true_shifts)}
        true_shifts : ndarray, shape (n_slit,)
    """
    n_slit = reference.shape[0]

    true_shift_zero = np.zeros(n_slit)
    img_zero = reference.copy()

    true_shift_const = np.full(n_slit, 0.05)
    img_const = shift_rows(reference, true_shift_const)

    random_component = np.random.uniform(-0.1, 0.1, size=n_slit)
    true_shift_random = 0.05 + random_component
    img_random = shift_rows(reference, true_shift_random)

    return {
        '1) zero shift': (img_zero, true_shift_zero),
        '2) constant 0.05px shift': (img_const, true_shift_const),
        '3) 0.05px + U(-0.1,0.1)px random shift': (img_random, true_shift_random),
    }


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def evaluate_case(name: str, reference: np.ndarray, test_img: np.ndarray,
                   true_shifts: np.ndarray, nominal_constant_shift: float = None):
    """
    Run measure_shift_vs_scan_position(reference, test_img) and compare the
    measured shifts against the known true_shifts.

    Note on sign convention: `test_img` is constructed so that its content
    is displaced by +true_shift relative to `reference` (via an exact
    FFT-based shift). measure_shift_vs_scan_position(reference, test_img)
    calls phase_cross_correlation(reference, test_img), which returns the
    shift required to register `test_img` onto `reference`, i.e.
    approximately -true_shift. We therefore compare `measured` against
    `-true_shifts`.

    Parameters
    ----------
    nominal_constant_shift : float, optional
        The known constant (systematic) shift component that was used to
        build `true_shifts` (e.g. 0.05 px), on top of which an independent
        per-position random component may have been added. If provided,
        this checks whether averaging the measured shift over all scan
        positions recovers this nominal constant (i.e. whether the random
        per-position component averages out).
    """
    measured, errors = measure_shift_vs_scan_position(reference, test_img)
    expected = -true_shifts
    residual = measured - expected
    bias = np.nanmean(residual)
    std = np.nanstd(residual)
    rms = np.sqrt(np.nanmean(residual ** 2))

    print(f"\n{name}")
    print(f"  true shift (applied to test_img):  mean={np.mean(true_shifts):+.4f} px, "
          f"std={np.std(true_shifts):.4f} px")
    print(f"  expected measured (-true shift):   mean={np.mean(expected):+.4f} px, "
          f"std={np.std(expected):.4f} px")
    print(f"  measured shift:                    mean={np.nanmean(measured):+.4f} px, "
          f"std={np.nanstd(measured):.4f} px")
    print(f"  residual (measured - expected): bias={bias:+.4f} px, "
          f"std={std:.4f} px, rms={rms:.4f} px")

    if nominal_constant_shift is not None:
        mean_measured_as_true = -np.nanmean(measured)
        n = np.sum(np.isfinite(measured))
        sem = np.nanstd(measured) / np.sqrt(max(n, 1))
        print(f"  averaged-over-scan check: mean(measured) as true-shift "
              f"sign = {mean_measured_as_true:+.4f} px vs nominal constant "
              f"= {nominal_constant_shift:+.4f} px "
              f"(diff={mean_measured_as_true - nominal_constant_shift:+.4f} px, "
              f"SEM={sem:.4f} px, n={n})")

    return {'measured': measured, 'errors': errors, 'true': true_shifts,
            'expected': expected, 'residual': residual, 'bias': bias,
            'std': std, 'rms': rms}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
#%%
if __name__ == '__main__':
    #%%
    reference = load_reference_image()
    print(f"Reference image shape (n_slit, spatial_y): {reference.shape}")

    cases = make_test_cases(reference)

    #%%
    nominal_constants = {
        '1) zero shift': 0.0,
        '2) constant 0.05px shift': 0.05,
        '3) 0.05px + U(-0.1,0.1)px random shift': 0.05,
    }
    results = {}
    for name, (test_img, true_shifts) in cases.items():
        results[name] = evaluate_case(name, reference, test_img, true_shifts,
                                       nominal_constant_shift=nominal_constants[name])

    #%%
    fig, axes = plt.subplots(len(results), 1, figsize=(9, 3.5 * len(results)),
                              sharex=True)
    for ax, (name, res) in zip(axes, results.items()):
        mean_measured = np.nanmean(res['measured'])
        std_measured = np.nanstd(res['measured'])
        ax.plot(res['expected'], label='expected (-true shift)', linestyle='-',
                 marker=None)
        ax.plot(res['measured'], label='measured shift', linestyle='--',
                 marker='.')
        ax.axhline(mean_measured, color='tab:red', linewidth=1.2, linestyle=':',
                   label=f'measured mean: {mean_measured:+.4f} ± {std_measured:.4f} px')
        ax.set_title(f"{name}  (bias={res['bias']:+.4f}px, "
                     f"std={res['std']:.4f}px, rms={res['rms']:.4f}px)")
        ax.set_ylabel('shift (px)')
        ax.legend()
    axes[-1].set_xlabel('scan (slit) index')
    plt.tight_layout()
    plt.show()

    #%%
    fig, axes = plt.subplots(len(results), 1, figsize=(9, 3 * len(results)),
                              sharex=True)
    for ax, (name, res) in zip(axes, results.items()):
        mean_res = np.nanmean(res['residual'])
        std_res = np.nanstd(res['residual'])
        ax.axhline(0, color='k', linewidth=0.8)
        ax.plot(res['residual'], marker='.', linestyle='-')
        ax.axhline(mean_res, color='tab:red', linewidth=1.2, linestyle=':',
                   label=f'residual mean: {mean_res:+.4f} ± {std_res:.4f} px')
        ax.set_title(f"{name}: residual (measured - expected)")
        ax.set_ylabel('residual (px)')
        ax.legend()
    axes[-1].set_xlabel('scan (slit) index')
    plt.tight_layout()
    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: does using order=3 (cubic spline) instead of the pipeline default
order=1 (linear) in the L1->L2 y-shift correction step improve the
continuum-alignment shift measurement (test_alignement_in_continuum)?

IMPORTANT: This is a TEST-ONLY comparison. It does NOT modify the real
reduction pipeline default (order=1 remains the default in
themis_data_reduction._apply_yshift_to_half / _apply_yshift_map_to_half /
_apply_yshift_correction_to_half). This script only READS existing L1
scan data and existing flat_center auxiliary calibration files already
produced by the real pipeline; it NEVER writes/overwrites any pipeline
output file (L2/L3/L4 are only kept in-memory here).

Approach: replicate the L1->L2 (y-shift), L2->L3 (desmiling), and L3->L4
(flat_center corrections) steps in-memory for data_type='scan', using the
exact same internal helper functions as reduce_l1_to_l2 /
reduce_l2_to_l3 / _apply_flat_center_corrections_to_l3, but with the
y-shift interpolation order as a free parameter. Then run the same
continuum alignment shift analysis as test_alignement_in_continuum on
both resulting L4 CycleSets and compare the standard deviation of the
per-scan-position shifts.

Usage: run cell by cell.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from themis.core import themis_io as tio
from themis.core import themis_data_reduction as tdr
from themis.core import data_classes as dct
from themis.datasets.themis_datasets_2025 import get_config, _build_file_set

from test_alignement_in_continuum import (
    CONFIG_PATH, SEQUENCE, CONTINUUM_RANGES, SLIT_MARGIN, HALVES,
    HALF_TO_POLSTATE, continuum_image, apply_slit_margin, slit_profile,
    measure_shift, measure_shift_vs_scan_position,
)

LINE = 'sr'


# ----------------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------------

def load_config_for_line(line: str, sequence: int):
    config = get_config(line=line, config_path=CONFIG_PATH,
                         auto_discover_files=False, auto_create_dirs=False)
    config.dataset['scan']['sequence'] = sequence
    config.dataset['scan']['files'] = _build_file_set(
        config.directories, config.dataset['scan'], config.data_types,
        config.reduction_levels, config.cam, config.line,
        yshift_calibration_date=config.yshift_calibration_date,
    )
    # Also build the flat_center file set: needed to locate yshift_upper/
    # yshift_lower auxiliary files (_load_yshift_arrays) and the flat_center
    # L4 calibration auxiliary files (delta_offsets_upper, amended
    # illumination, z3cc offsets, continuum correction).
    config.dataset['flat_center']['files'] = _build_file_set(
        config.directories, config.dataset['flat_center'], config.data_types,
        config.reduction_levels, config.cam, config.line,
        yshift_calibration_date=config.yshift_calibration_date,
    )
    return config


# ----------------------------------------------------------------------------
# In-memory pipeline replication (read-only, never saves to disk)
# ----------------------------------------------------------------------------

def build_l2_cycleset(config, order: int):
    """
    Replicate reduce_l1_to_l2 for data_type='scan', in-memory, with a
    configurable interpolation order for the y-shift correction.
    """
    yshifts = tdr._load_yshift_arrays(config)
    if yshifts is None:
        raise RuntimeError("y-shift arrays not found; run the real pipeline once first.")

    l1_data, header = tio.read_any_file(config, 'scan', status='l1', verbose=False)

    l2 = dct.CycleSet()
    for key, frame in l1_data.items():
        dest = dct.Frame(f"scan_l2_{frame.name}")
        for half in ('upper', 'lower'):
            half_obj = frame.get_half(half)
            data_l1 = half_obj.data.astype('float32')
            data_l2 = tdr._apply_yshift_correction_to_half(data_l1, yshifts[half], order=order)
            if half_obj.pol_state:
                dest.set_half(half, data_l2.astype('float32'), half_obj.pol_state)
            else:
                dest.set_half(half, data_l2.astype('float32'))
        l2.add_frame(dest, key)
    return l2, header


def build_l3_cycleset(config, l2_cycleset):
    """
    Replicate reduce_l2_to_l3 (desmiling) for data_type='scan', in-memory.
    """
    offset_maps = tio._load_offset_maps(config, data_type='flat_center')
    if offset_maps is None:
        raise RuntimeError("offset maps not found; run flat_center L2->L3 first.")

    l3 = dct.CycleSet()
    for key, frame in l2_cycleset.items():
        dest = dct.Frame(f"scan_l3_{frame.name}")
        for half in ('upper', 'lower'):
            half_obj = frame.get_half(half)
            frame_2d = half_obj.data.astype('float32')
            desmiled = tdr._apply_desmiling(frame_2d, offset_maps[half])
            if half_obj.pol_state:
                dest.set_half(half, desmiled.astype('float32'), half_obj.pol_state)
            else:
                dest.set_half(half, desmiled.astype('float32'))
        l3.add_frame(dest, key)
    return l3


def load_flat_center_corrections(config):
    """
    Replicate the correction-loading portion of
    _apply_flat_center_corrections_to_l3 (reads pre-computed auxiliary
    calibration files only, never recomputes them).
    """
    fc_files = config.dataset.get('flat_center', {}).get('files')
    fc_aux = getattr(fc_files, 'auxiliary', {})

    delta_offsets_upper_file = fc_aux.get('delta_offsets_upper')
    with fits.open(delta_offsets_upper_file) as hdul:
        delta_offsets_upper = np.array(hdul[0].data)
        wl_upper = np.array(hdul['WAVELENGTH'].data)

    cont_norm_upper = None
    amended_illum_upper = fc_aux.get('amended_illumination_upper')
    if amended_illum_upper and Path(amended_illum_upper).exists():
        with fits.open(amended_illum_upper) as hdul:
            if len(hdul) > 1:
                cont_corr_upper = np.array(hdul[1].data)
                if cont_corr_upper.ndim == 3:
                    cont_corr_upper = np.mean(cont_corr_upper, axis=0)
                cont_norm_upper = cont_corr_upper / np.mean(cont_corr_upper)

    z3cc_file = fc_aux.get('z3cc_offsets_lower')
    with fits.open(z3cc_file) as hdul:
        wl_lower_z3cc = np.array(hdul['WAVELENGTH'].data)

    cont_norm_lower = None
    cont_corr_lower_file = fc_aux.get('continuum_correction_lower')
    if cont_corr_lower_file and Path(cont_corr_lower_file).exists():
        with fits.open(cont_corr_lower_file) as hdul:
            cont_norm_lower = np.array(hdul[0].data)

    return dict(delta_offsets_upper=delta_offsets_upper, wl_upper=wl_upper,
                cont_norm_upper=cont_norm_upper, wl_lower_z3cc=wl_lower_z3cc,
                cont_norm_lower=cont_norm_lower)


def build_l4_cycleset(l3_cycleset, corr: dict):
    """
    Replicate the per-frame correction loop of
    _apply_flat_center_corrections_to_l3, in-memory.
    """
    l4 = dct.CycleSet()
    for key, frame in l3_cycleset.items():
        upper_half = frame.get_half('upper')
        lower_half = frame.get_half('lower')

        upper_l3 = upper_half.data.astype('float32')
        lower_l3 = lower_half.data.astype('float32')

        upper_shifted = tdr._apply_desmiling(
            upper_l3, np.tile(corr['delta_offsets_upper'], (upper_l3.shape[0], 1)))
        if corr['cont_norm_upper'] is not None:
            upper_l4 = upper_shifted / corr['cont_norm_upper']
        else:
            upper_l4 = upper_shifted

        lower_z3cc_shifted = np.empty_like(lower_l3)
        for row in range(lower_l3.shape[0]):
            f_interp = interp1d(corr['wl_lower_z3cc'], lower_l3[row, :], kind='cubic',
                                 bounds_error=False, fill_value='extrapolate')
            lower_z3cc_shifted[row, :] = f_interp(corr['wl_upper'])

        if corr['cont_norm_lower'] is not None:
            lower_l4 = lower_z3cc_shifted * corr['cont_norm_lower']
        else:
            lower_l4 = lower_z3cc_shifted

        dest = dct.Frame(frame.name)
        dest.set_half('upper', upper_l4.astype('float32'), upper_half.pol_state)
        dest.set_half('lower', lower_l4.astype('float32'), lower_half.pol_state)
        l4.add_frame(dest, key)
    return l4


# ----------------------------------------------------------------------------
# Alignment analysis (mirrors test_alignement_in_continuum.analyze_line)
# ----------------------------------------------------------------------------

def run_alignment_analysis(l4_cycleset, line: str, label: str):
    ranges = CONTINUUM_RANGES[line]
    cubes = {
        'upper': l4_cycleset.stack_all('upper'),
        'lower': l4_cycleset.stack_all('lower'),
    }

    results = {}
    for half in HALVES:
        cube = cubes[half]
        img1 = continuum_image(cube, ranges['region_1'])
        img2 = continuum_image(cube, ranges['region_2'])
        img1 = apply_slit_margin(img1, SLIT_MARGIN)
        img2 = apply_slit_margin(img2, SLIT_MARGIN)
        profile1 = slit_profile(img1)
        profile2 = slit_profile(img2)
        shift, error, _ = measure_shift(profile1, profile2)
        shifts_per_scan, _ = measure_shift_vs_scan_position(img1, img2)
        results[half] = {'shift': shift, 'shifts_per_scan': shifts_per_scan}
        print(f"  [{label}] {half} ({HALF_TO_POLSTATE[half]}): "
              f"avg-first shift={shift:+.4f} px, "
              f"per-scan mean={np.nanmean(shifts_per_scan):+.4f} px, "
              f"per-scan std={np.nanstd(shifts_per_scan):.4f} px")
    return results


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
#%%
if __name__ == '__main__':
    #%%
    config = load_config_for_line(LINE, SEQUENCE)

    print(f"\n{'='*70}\nBuilding L2 (order=1, pipeline default)\n{'='*70}")
    l2_o1, header = build_l2_cycleset(config, order=1)
    print(f"\n{'='*70}\nBuilding L2 (order=3, TEST ONLY)\n{'='*70}")
    l2_o3, _ = build_l2_cycleset(config, order=3)

    #%%
    print(f"\n{'='*70}\nBuilding L3 (desmiling)\n{'='*70}")
    l3_o1 = build_l3_cycleset(config, l2_o1)
    l3_o3 = build_l3_cycleset(config, l2_o3)

    #%%
    print(f"\n{'='*70}\nLoading flat_center corrections + building L4\n{'='*70}")
    corr = load_flat_center_corrections(config)
    l4_o1 = build_l4_cycleset(l3_o1, corr)
    l4_o3 = build_l4_cycleset(l3_o3, corr)

    #%%
    print(f"\n{'='*70}\nSHIFT ANALYSIS COMPARISON ({LINE.upper()}, seq={SEQUENCE})\n{'='*70}")
    res_o1 = run_alignment_analysis(l4_o1, LINE, 'order=1 (pipeline default)')
    res_o3 = run_alignment_analysis(l4_o3, LINE, 'order=3 (TEST)')

    print(f"\n{'='*70}\nSUMMARY: per-scan-position std (order=1 vs order=3)\n{'='*70}")
    for half in HALVES:
        std1 = np.nanstd(res_o1[half]['shifts_per_scan'])
        std3 = np.nanstd(res_o3[half]['shifts_per_scan'])
        verdict = 'IMPROVED (smaller std)' if std3 < std1 else 'NOT IMPROVED'
        print(f"  {half} ({HALF_TO_POLSTATE[half]}): "
              f"std order=1={std1:.4f} px, std order=3={std3:.4f} px  -> {verdict}")

    #%%
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, half in zip(axes, HALVES):
        std1 = np.nanstd(res_o1[half]['shifts_per_scan'])
        std3 = np.nanstd(res_o3[half]['shifts_per_scan'])
        ax.plot(res_o1[half]['shifts_per_scan'], marker='.', linestyle='-',
                label=f"order=1 (std={std1:.4f}px)")
        ax.plot(res_o3[half]['shifts_per_scan'], marker='.', linestyle='-',
                label=f"order=3 (std={std3:.4f}px)")
        ax.set_title(f"{LINE.upper()} seq={SEQUENCE}: {half} ({HALF_TO_POLSTATE[half]}) "
                     f"shift vs scan position")
        ax.set_ylabel('shift along slit (px)')
        ax.legend()
    axes[-1].set_xlabel('scan (slit) index')
    plt.tight_layout()
    plt.show()

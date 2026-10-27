#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script: global stray-light removal (themis.core.stray_light).

Loads the L4 flat_center data of a camera, removes the global spectrograph
stray light from the upper and lower frame half with the levels from the
camera's atlas-fit config (input.stray_light for upper, input.stray_light_lower
for lower), and compares the spatially averaged spectrum before and after the
correction with the Hamburg disk-center (HHDC) atlas, convolved with the
instrumental FWHM from the atlas-fit config.

The normalization follows the atlas-fit normalized plot (atlas-fit/bin/prepare
and fitting/parameter_fit.py): the data are scaled so that their mean equals
the mean of the atlas over the same wavelength window, and both are divided by
the mean atlas continuum. Since the stray light is removed as an offset
(I - mean(I) * s/100), the corrected spectrum has deeper lines and should match
the atlas better. The atlas-fit error measure sum((data - atlas)^2 / data^2) is
printed for both cases.

A second test (test_remove_straylight_scan) loads L4 scan data off disk center
(default: m30, sequence 18), where the disk-center atlas is not a valid
reference. There, only the raw and the stray-light corrected spectra (averaged
over all scan positions and the slit) are compared, each normalized to its own
continuum level, together with the ratio corrected/raw.

Usage: run cell by cell.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'atlas-fit' / 'src'))

from themis.datasets.themis_datasets_2025 import get_config, _build_file_set
from themis.core import themis_io as tio
from themis.core import stray_light as sl

from atlas_fit.atlantes import atlas_factory
from atlas_fit.base.constants import NM2ANGSTROM
from atlas_fit.models.differential_convolution import FWHM, DifferentialSpectralPSF

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

CONFIG_PATH = 'configs/formation_dataset_2025-07-05.toml'
HALVES = ('upper', 'lower')

# Pixels trimmed from both ends of the slit (spatial) direction. The spectral
# window is the spectral part of the atlas-fit input.roi (as for the mean in
# the stray-light correction), since the L4 edge columns can contain artifacts.
SLIT_MARGIN = 25

# Align the data wavelength axis to the atlas by cross-correlation before the
# comparison (search range +-ALIGN_RANGE_NM). The found shift is printed: it
# reveals an offset of the L4 wavelength calibration w.r.t. the atlas.
ALIGN_TO_ATLAS = True
ALIGN_RANGE_NM = 0.1

# Off-disk-center scan used by test_remove_straylight_scan (m30 = -30 deg,
# see POSITION_TO_SEQUENCE in scripts/process_formation_height_line_levels.py)
SCAN_POSITION = 'm30'
SCAN_SEQUENCE = 18

# Continuum level of a spectrum (for the scan test) = this percentile of the
# spectrum within the spectral roi
CONTINUUM_PERCENTILE = 95


def load_flat_center(line: str):
    """Return (config, L4 flat_center FramesSet, wavelength [nm])."""
    config = get_config(line=line, config_path=CONFIG_PATH,
                        auto_discover_files=True, auto_create_dirs=False)
    data, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='l4')
    wl, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='wl')
    return config, data, np.asarray(wl, dtype='float64')


def load_scan(line: str, sequence: int):
    """Return (config, L4 scan CycleSet, wavelength [nm]) for a scan sequence."""
    config = get_config(line=line, config_path=CONFIG_PATH,
                        auto_discover_files=True, auto_create_dirs=False)
    config.dataset['scan']['sequence'] = sequence
    config.dataset['scan']['files'] = _build_file_set(
        config.directories, config.dataset['scan'], config.data_types,
        config.reduction_levels, config.cam, config.line,
        yshift_calibration_date=config.yshift_calibration_date,
    )
    data, _ = tio.read_any_file(config, 'scan', verbose=False, status='l4')
    wl, _ = tio.read_any_file(config, 'scan', verbose=False, status='wl')
    return config, data, np.asarray(wl, dtype='float64')


def scan_mean_spectrum(data_set, half: str) -> np.ndarray:
    """Average a frame half over all scan positions and the slit direction."""
    return np.mean([mean_spectrum(h.data) for h in data_set.get_all_halves(half)], axis=0)


def mean_spectrum(frame: np.ndarray) -> np.ndarray:
    """Average a frame half (spatial, spectral) over the slit direction."""
    return np.nanmean(frame[SLIT_MARGIN:-SLIT_MARGIN, :], axis=0)


def convolved_atlas(config, wl_min: float, wl_max: float):
    """
    Load the HHDC atlas in [wl_min, wl_max] nm and convolve it with the data
    FWHM from the atlas-fit config (as atlas-fit's convolve_atlas_profile).

    Returns (wl [nm], convolved intensity, mean continuum).
    """
    with open(config.cam.atlas_fit_config, 'r') as f:
        atlas_cfg = yaml.safe_load(f)
    atlas = atlas_factory('hhdc', wl_min, wl_max, conversion=NM2ANGSTROM)
    fwhm = FWHM(atlas=atlas.fwhm, data=atlas_cfg['input']['fwhm'])
    intensity = DifferentialSpectralPSF(fwhm).convolve(atlas.intensity, atlas.wl)
    return atlas.wl, intensity, np.mean(atlas.continuum)


def wavelength_shift_to_atlas(wl: np.ndarray, spec: np.ndarray, atlas_wl: np.ndarray,
                              atlas_int: np.ndarray, search_nm: float = ALIGN_RANGE_NM) -> float:
    """Shift [nm] to add to `wl` that maximizes the correlation with the atlas."""
    step = np.mean(np.diff(wl)) / 4
    shifts = np.arange(-search_nm, search_nm + step, step)
    corr = [np.corrcoef(np.interp(wl + s, atlas_wl, atlas_int), spec)[0, 1] for s in shifts]
    return float(shifts[int(np.argmax(corr))])


def normalize_to_atlas(spec: np.ndarray, atlas_on_data: np.ndarray, cont: float) -> np.ndarray:
    """Scale spectrum to the atlas mean and divide by the atlas continuum."""
    return spec / np.mean(spec) * np.mean(atlas_on_data) / cont


def atlas_fit_error(data: np.ndarray, atlas: np.ndarray) -> float:
    """atlas-fit global error: sum((data - atlas)^2 / data^2)."""
    return float(np.sum((data - atlas) ** 2 / data ** 2))


def test_remove_straylight(line: str = 'sr', show_plots: bool = True):
    config, data, wl = load_flat_center(line)
    levels = sl.get_stray_light_levels(config)
    roi = levels['spectral_roi']
    print(f"\n{line.upper()} flat_center: stray light upper={levels['upper']} %, "
          f"lower={levels['lower']} %, spectral roi={roi}")

    corrected = sl.remove_stray_light_from_set(data, levels)

    wl_cut = wl[roi]
    atlas_wl, atlas_int, atlas_cont = convolved_atlas(
        config, wl_cut.min() - ALIGN_RANGE_NM, wl_cut.max() + ALIGN_RANGE_NM)
    if ALIGN_TO_ATLAS:
        wl_shift = wavelength_shift_to_atlas(
            wl_cut, mean_spectrum(data[0]['upper'].data)[roi], atlas_wl, atlas_int)
        print(f"  Wavelength axis aligned to atlas: shift = {wl_shift:+.4f} nm "
              f"({wl_shift / np.mean(np.diff(wl_cut)):+.1f} px)")
        wl_cut = wl_cut + wl_shift
    atlas_on_data = np.interp(wl_cut, atlas_wl, atlas_int)
    atlas_norm = atlas_on_data / atlas_cont

    results = {}
    for half in HALVES:
        raw = mean_spectrum(data[0][half].data)[roi]
        cor = mean_spectrum(corrected[0][half].data)[roi]
        raw_n = normalize_to_atlas(raw, atlas_on_data, atlas_cont)
        cor_n = normalize_to_atlas(cor, atlas_on_data, atlas_cont)
        results[half] = {'raw': raw_n, 'corrected': cor_n,
                         'err_raw': atlas_fit_error(raw_n, atlas_norm),
                         'err_corrected': atlas_fit_error(cor_n, atlas_norm)}
        print(f"  {half:5s}: atlas-fit error raw={results[half]['err_raw']:.4e}, "
              f"corrected={results[half]['err_corrected']:.4e}")

    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})
        for col, half in enumerate(HALVES):
            r = results[half]
            ax = axes[0, col]
            ax.plot(wl_cut, atlas_norm, color='k', label='HHDC atlas (convolved)')
            ax.plot(wl_cut, r['raw'], '--', label='not stray-light corrected')
            ax.plot(wl_cut, r['corrected'], label=f"stray-light corrected ({levels[half]} %)")
            ax.axhline(1, linestyle='--', color='gray')
            ax.set_title(f"{line.upper()} flat_center {half}")
            ax.set_ylabel('Normalized I')
            ax.grid(linestyle=':')
            ax.legend()

            ax = axes[1, col]
            ax.plot(wl_cut, atlas_norm / r['raw'], '--', linewidth=0.7, label='raw')
            ax.plot(wl_cut, atlas_norm / r['corrected'], linewidth=0.7, label='corrected')
            ax.axhline(1, linestyle='--', color='gray')
            ax.set_xlabel(r'$\lambda$ [nm]')
            ax.set_ylabel('<atlas>/<data>')
            ax.grid(linestyle=':')
            ax.legend()
        plt.tight_layout()
        plt.show()

    return results


def test_remove_straylight_scan(line: str = 'sr', sequence: int = SCAN_SEQUENCE,
                                position: str = SCAN_POSITION, show_plots: bool = True):
    """
    Compare raw and stray-light corrected L4 scan spectra off disk center
    (no atlas reference). Spectra are averaged over all scan positions and the
    slit, and normalized to their own continuum (CONTINUUM_PERCENTILE).
    """
    config, data, wl = load_scan(line, sequence)
    levels = sl.get_stray_light_levels(config)
    roi = levels['spectral_roi']
    wl_cut = wl[roi]
    print(f"\n{line.upper()} scan {position} (seq={sequence}): stray light "
          f"upper={levels['upper']} %, lower={levels['lower']} %, spectral roi={roi}")

    # Raw spectra first, then correct in place (avoids copying the full scan)
    raw = {half: scan_mean_spectrum(data, half)[roi] for half in HALVES}
    sl.remove_stray_light_from_set(data, levels, inplace=True)
    cor = {half: scan_mean_spectrum(data, half)[roi] for half in HALVES}

    results = {}
    for half in HALVES:
        raw_n = raw[half] / np.percentile(raw[half], CONTINUUM_PERCENTILE)
        cor_n = cor[half] / np.percentile(cor[half], CONTINUUM_PERCENTILE)
        results[half] = {'raw': raw_n, 'corrected': cor_n}
        print(f"  {half:5s}: deepest line core raw={raw_n.min():.4f}, "
              f"corrected={cor_n.min():.4f}, "
              f"mean offset removed={np.mean(raw[half] - cor[half]):.1f} counts")

    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})
        for col, half in enumerate(HALVES):
            r = results[half]
            ax = axes[0, col]
            ax.plot(wl_cut, r['raw'], '--', label='not stray-light corrected')
            ax.plot(wl_cut, r['corrected'], label=f"stray-light corrected ({levels[half]} %)")
            ax.axhline(1, linestyle='--', color='gray')
            ax.set_title(f"{line.upper()} scan {position} (seq={sequence}) {half}")
            ax.set_ylabel(f'I / I_c ({CONTINUUM_PERCENTILE}th percentile)')
            ax.grid(linestyle=':')
            ax.legend()

            ax = axes[1, col]
            ax.plot(wl_cut, r['corrected'] / r['raw'], linewidth=0.7)
            ax.axhline(1, linestyle='--', color='gray')
            ax.set_xlabel(r'$\lambda$ [nm]')
            ax.set_ylabel('corrected/raw')
            ax.grid(linestyle=':')
        plt.tight_layout()
        plt.show()

    return results


#%%
if __name__ == '__main__':
    #%%
    results_sr = test_remove_straylight('sr')
    #%%
    results_ti = test_remove_straylight('ti')
    #%%
    # Off disk center (m30): no atlas comparison, only raw vs corrected
    results_scan_sr = test_remove_straylight_scan('sr')
    #%%
    results_scan_ti = test_remove_straylight_scan('ti')

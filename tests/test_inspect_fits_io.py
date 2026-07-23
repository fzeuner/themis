#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: franziskaz

Inspection / sanity-check script for the FITS files read and written by
process_formation_height_line_levels.py.

This script inspects two separate files:
  - INPUT  = `_fit_line.fits`: the fitted line profile (component_ti /
             component_sr, shape (wavelength, slit, scan)), written by
             SpectrumPart.save(). This is an intermediate fitting result,
             but it is the quantity that a run_voigt_as.pro-style level
             calculation (voigt_as) would consume to derive line levels.
  - OUTPUT = `_levels.fits`: the LEVELS / WIDTHS / LEVELS_META extensions,
             written by save_line_levels(). These are the line levels
             actually derived from the INPUT profile by line_levels(),
             stored in a dedicated file (separate from the fitted profile)
             precisely because computing them can take hours and the
             result must not be lost or tied to re-saving the fit.

run_voigt_as.pro reference (see idl/run_voigt_as.pro):
  - Input cube: readfits(file) -> array indexed as cube[wavelength, slit, scan]
    (same wavelength/slit/scan axis order as our Python fitted-line array).
  - Defaults: Nniv=25 (number of line levels), frac_Ic=0.95 (continuum
    fraction used as the outer level), NIc=1 (single continuum definition).
  - For each pixel: p = cube[lbda_min:lbda_max, i, k] normalized by its max
    (maxmap), then voigt_as(...) computes `val` = intensity at each of the
    Nniv levels from line core out to the width where the profile crosses
    frac_Ic * continuum. This is analogous to our line_levels() function
    applied to the fitted line profile (the "input" above).
  - Output Mval0 has IDL shape (Ny, ngood, Nniv) i.e. (slit, scan, levels).
    This is IDL/Fortran-style axis ordering; it is NOT the same as our
    Python-native array which we store as (n_levels, n_slit, n_scan) via
    astropy (self-consistent round-trip within Python, but the axis order
    is reversed relative to a literal IDL Mval0 dump). This script makes
    that distinction explicit so it is not accidentally missed.

Usage (interactive, e.g. Spyder/IPython, run cell by cell):
    %runcell -i 0 tests/test_inspect_fits_io.py   # imports/definitions
    %runcell -i 1 tests/test_inspect_fits_io.py   # edit LINE/POSITION/SLIT/SCAN below, then run

Usage (plain script, run with uv):
    uv run python tests/test_inspect_fits_io.py
    (edit the LINE / POSITION / SLIT / SCAN parameters below as needed)
"""

import sys
import os
from pathlib import Path

import numpy as np
from astropy.io import fits

# Add scripts directory to path for importing process_formation_height_line_levels
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from process_formation_height_line_levels import get_config, CONFIG_PATH

# Values documented in idl/run_voigt_as.pro defaults, for cross-checking
IDL_NNIV_DEFAULT = 25       # Nniv
IDL_FRAC_IC_DEFAULT = 0.95  # frac_Ic


def _fmt_shape(arr):
    return 'None' if arr is None else f"shape={arr.shape} dtype={arr.dtype}"


def print_fits_structure(filepath):
    """
    Generic, dependency-free pretty-printer for any FITS file: lists every
    HDU with its name, shape, dtype and a handful of header keywords.
    """
    print(f"\n--- FITS structure: {filepath} ---")
    try:
        with fits.open(filepath) as hdul:
            for i, hdu in enumerate(hdul):
                name = hdu.name or f"HDU{i}"
                shape = getattr(hdu.data, 'shape', None)
                dtype = getattr(hdu.data, 'dtype', None)
                print(f"  [{i}] {name}: shape={shape} dtype={dtype}")
                interesting_keys = ['NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
                                     'NLEVELS', 'NSLIT', 'NSCAN',
                                     'CONFIG', 'DATE', 'SEQUENCE', 'PYTHON']
                header_bits = []
                for key in interesting_keys:
                    if key in hdu.header:
                        header_bits.append(f"{key}={hdu.header[key]}")
                if header_bits:
                    print(f"      header: {', '.join(header_bits)}")
    except FileNotFoundError:
        print(f"  FILE NOT FOUND: {filepath}")


def inspect_fit_line_file(base_filepath):
    """
    Inspect the INPUT to the line-levels calculation: `_fit_line.fits`
    (fitted line profile only, written by SpectrumPart.save()). This is
    what a run_voigt_as.pro-style voigt_as() call would consume to derive
    line levels.
    """
    fit_line_path = f"{base_filepath}_fit_line.fits"
    print(f"\n{'='*70}\nINPUT -- _fit_line.fits: {base_filepath}\n{'='*70}")
    print_fits_structure(fit_line_path)

    try:
        with fits.open(fit_line_path) as hdul:
            data = hdul[0].data
            wvl_nm = hdul['WAVELENGTH'].data
            print(f"\n  data {_fmt_shape(data)} (wavelength, slit, scan)")
            print(f"  wavelength on disk: [{wvl_nm.min():.4f}, {wvl_nm.max():.4f}] nm "
                  f"-> Angstrom: [{wvl_nm.min()*10:.2f}, {wvl_nm.max()*10:.2f}]")
            if data is not None and data.ndim != 3:
                print(f"  WARNING: expected 3D data (wavelength, slit, scan), got ndim={data.ndim}")
            if 'LEVELS' in hdul:
                print("  WARNING: this file still has a LEVELS extension embedded -- "
                      "it looks like an OLD-format file. Run scripts/migrate_split_fit_line_levels.py.")
    except FileNotFoundError:
        print(f"  FILE NOT FOUND: {fit_line_path}")


def inspect_levels_file(base_filepath):
    """
    Inspect the OUTPUT of the line-levels calculation: `_levels.fits`
    (LEVELS/WIDTHS/LEVELS_META, written by save_line_levels()). Validates
    internal consistency against the shapes/values documented in
    run_voigt_as.pro.
    """
    levels_path = f"{base_filepath}_levels.fits"
    print(f"\n{'='*70}\nOUTPUT -- _levels.fits: {base_filepath}\n{'='*70}")
    print_fits_structure(levels_path)

    try:
        with fits.open(levels_path) as hdul:
            levels = hdul['LEVELS'].data          # Python shape: (n_levels, n_slit, n_scan)
            widths = hdul['WIDTHS'].data if 'WIDTHS' in hdul else None
            meta = hdul['LEVELS_META'].data if 'LEVELS_META' in hdul else None
            n_levels_hdr = hdul['LEVELS'].header.get('NLEVELS')
            n_slit_hdr = hdul['LEVELS'].header.get('NSLIT')
            n_scan_hdr = hdul['LEVELS'].header.get('NSCAN')

            print(f"\n  LEVELS array: python shape (n_levels, n_slit, n_scan) = {levels.shape}")
            print(f"  Header says: NLEVELS={n_levels_hdr}, NSLIT={n_slit_hdr}, NSCAN={n_scan_hdr}")

            ok = True
            if levels.shape != (n_levels_hdr, n_slit_hdr, n_scan_hdr):
                print("  MISMATCH: LEVELS.shape does not match (NLEVELS, NSLIT, NSCAN) header values!")
                ok = False

            if widths is None:
                print("  MISMATCH: WIDTHS extension missing.")
                ok = False
            else:
                if len(widths) != n_levels_hdr:
                    print(f"  MISMATCH: len(WIDTHS)={len(widths)} != NLEVELS={n_levels_hdr}")
                    ok = False
                if not np.all(np.diff(widths) >= 0):
                    print("  WARNING: WIDTHS is not monotonically increasing.")
                print(f"  WIDTHS range: [{widths.min():.5f}, {widths.max():.5f}] Angstrom "
                      f"(n_levels={len(widths)}; run_voigt_as.pro default Nniv={IDL_NNIV_DEFAULT})")

            if meta is None:
                print("  MISMATCH: LEVELS_META extension missing.")
                ok = False
            else:
                expected_len = n_slit_hdr * n_scan_hdr
                if len(meta) != expected_len:
                    print(f"  MISMATCH: LEVELS_META rows={len(meta)} != NSLIT*NSCAN={expected_len}")
                    ok = False
                cont = meta['continuum_levels']
                n_finite = np.sum(np.isfinite(cont))
                print(f"  LEVELS_META: {n_finite}/{len(cont)} pixels have a finite continuum_level "
                      f"(i.e. have been fitted)")

            print(f"\n  Consistency check: {'PASSED' if ok else 'FAILED'}")

            print("\n  NOTE on IDL compatibility (run_voigt_as.pro):")
            print("  - IDL Mval0 is written with shape (Ny, ngood, Nniv) i.e. (slit, scan, levels).")
            print("  - Our Python LEVELS array uses (n_levels, n_slit, n_scan) -- axis order is")
            print("    REVERSED relative to a literal IDL dump. This is a self-consistent")
            print("    Python-only round trip (astropy read/write), NOT a byte-for-byte IDL file.")
            print("    If direct IDL interoperability is required, transpose before/after IO:")
            print("    idl_style = np.transpose(levels, (1, 2, 0))  # -> (slit, scan, levels)")

    except FileNotFoundError:
        print(f"  FILE NOT FOUND: {levels_path} (no fit performed yet, or levels not saved)")


def inspect_base_filepath(base_filepath):
    """
    Inspect both the INPUT (`_fit_line.fits`) and OUTPUT (`_levels.fits`)
    files given a base filepath directly (no SpectrumContainer/Spectrum
    involved), i.e. the same base filepath passed to
    Spectrum.save()/Spectrum.load().
    """
    inspect_fit_line_file(str(base_filepath))
    inspect_levels_file(str(base_filepath))


###################
# Parameters -- edit these and re-run
###################

LINE = 'ti'                # e.g. 'ti' or 'sr'
POSITION = 'disk_center'   # e.g. 'disk_center'

###################
# Run
###################

# Resolve the base filepath the same way Spectrum.save()/load() does,
# without going through SpectrumContainer/Spectrum at all.
config = get_config(line=LINE, config_path=CONFIG_PATH, auto_discover_files=True, auto_create_dirs=False)
base_filepath = Path(config.directories.reduced) / 'spectra' / LINE / POSITION / POSITION

print(f"Inspecting line={LINE}, position={POSITION}\nBase filepath: {base_filepath}")

inspect_base_filepath(base_filepath)

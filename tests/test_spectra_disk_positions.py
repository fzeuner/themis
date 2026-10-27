#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script: compare spatially averaged spectra across disk positions.

For three disk positions (disk center, -30 deg, +30 deg), this script loads
the L4 scan cubes for both cameras (Ti, Sr) and both detector halves
(upper/pQ, lower/mQ), averages over the spatial dimensions (scan/slit index
and the slit direction) to obtain one 1D spectrum per
camera/half/position, and plots them for comparison. Each spectrum is
normalized to the mean of its "blue" continuum region, and only the inner
spectral range [200:-200] is shown.

Usage: run cell by cell.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from test_alignement_in_continuum import load_l4_scan_cubes

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Positions to load and their scan sequences (see
# scripts/process_formation_height_line_levels.py POSITION_TO_SEQUENCE).
POSITION_TO_SEQUENCE = {
    'disk_center': 27,
    'm30': 18,
    '30': 50,
}

HALVES = ('upper', 'lower')  # upper -> pQ, lower -> mQ (paired per frame)
HALF_TO_POLSTATE = {'upper': 'pQ', 'lower': 'mQ'}

# Margin (pixels) trimmed from both ends of the slit direction before
# averaging, to remove edge artifacts.
SLIT_MARGIN = 25

# "Blue" continuum spectral pixel ranges used for normalization
# ([start, end] inclusive; same as region_1 in the camera alignment test).
BLUE_CONTINUUM_RANGE = {
    'ti': (103, 116),
    'sr': (320, 337),
}

# Spectral pixels trimmed from both ends for display.
SPEC_CUT = 200


def spatially_averaged_spectrum(cube: np.ndarray, margin: int = SLIT_MARGIN) -> np.ndarray:
    """
    Average an L4 scan cube over the spatial dimensions to get a 1D spectrum.

    Parameters
    ----------
    cube : ndarray, shape (n_slit, spatial_y, spectral_x)
    margin : int
        Pixels trimmed from both ends of the slit direction before averaging.

    Returns
    -------
    ndarray, shape (spectral_x,)
    """
    if margin > 0:
        cube = cube[:, margin:-margin, :]
    return cube.mean(axis=(0, 1))


def normalize_to_blue_continuum(spectrum: np.ndarray, line: str) -> np.ndarray:
    """
    Normalize a spectrum by the mean intensity of its blue continuum region.

    Parameters
    ----------
    spectrum : ndarray, shape (spectral_x,)
    line : str, 'ti' or 'sr'

    Returns
    -------
    ndarray, shape (spectral_x,), normalized so the blue continuum mean = 1.
    """
    lo, hi = BLUE_CONTINUUM_RANGE[line]
    return spectrum / spectrum[lo:hi + 1].mean()


#%%
if __name__ == '__main__':
    #%%
    # Load all positions/cameras and compute the spatially averaged spectra,
    # normalized to the blue continuum (normalization on the full spectrum,
    # before the display cut).
    spectra = {}  # spectra[position][line][half] -> 1D ndarray (normalized)
    for position, sequence in POSITION_TO_SEQUENCE.items():
        spectra[position] = {}
        for line in ('ti', 'sr'):
            cubes, _header, _px = load_l4_scan_cubes(line, sequence)
            spectra[position][line] = {
                half: normalize_to_blue_continuum(
                    spatially_averaged_spectrum(cubes[half]), line)
                for half in HALVES
            }

    #%%
    # Plot: one subplot per camera x detector half, one line per position.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex='col')
    for row, line in enumerate(('ti', 'sr')):
        for col, half in enumerate(HALVES):
            ax = axes[row, col]
            for position in POSITION_TO_SEQUENCE:
                spec = spectra[position][line][half]
                x = np.arange(spec.size)[SPEC_CUT:-SPEC_CUT]
                ax.plot(x, spec[SPEC_CUT:-SPEC_CUT],
                        label=f"{position} (seq={POSITION_TO_SEQUENCE[position]})")
            ax.set_title(f"{line.upper()} camera, {half} ({HALF_TO_POLSTATE[half]})")
            ax.set_ylabel('normalized intensity')
            ax.legend()
    axes[-1, 0].set_xlabel('spectral pixel')
    axes[-1, 1].set_xlabel('spectral pixel')
    fig.suptitle('Spectra normalized to blue continuum, spatially averaged '
                 f'(slit margin={SLIT_MARGIN}px, spectral cut={SPEC_CUT}px)')
    plt.tight_layout()
    plt.show()

#!/usr/bin/env python3
"""
Display the continuum correction from the amended illumination pattern (upper)
and from the ratio polyfit (lower).

Usage:
    python scripts/analysis/plot_continuum_correction.py
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path


# --- Configuration ---
data_dir = Path('/home/franziskaz/data/themis/pdata/2025-07-07')
prefix = 'sr_flat_center_t023'

# amend_spectroflat continuum corrections
amend_frames = {
    'upper (amend_spectroflat)': data_dir / f'{prefix}_amended_illumination_upper.fits',
    'lower (amend_spectroflat)': data_dir / f'{prefix}_amended_illumination_lower.fits',
}

# ratio polyfit continuum correction for lower frame
ratio_polyfit_path = data_dir / f'{prefix}_continuum_correction_lower.fits'


def plot_continuum(ax_row, label, cont_norm, wl=None):
    """Plot 2D image, spatial profile, and spectral profile."""
    # Panel 1: 2D image
    ax = ax_row[0]
    im = ax.imshow(cont_norm, aspect='auto', origin='lower', cmap='RdBu_r',
                   vmin=0.95, vmax=1.05)
    ax.set_title(f'{label} — continuum correction')
    ax.set_xlabel('Column [px]')
    ax.set_ylabel('Row [px]')
    plt.colorbar(im, ax=ax, label='Correction factor')

    # Panel 2: spatial profile (column-averaged)
    ax = ax_row[1]
    spatial_profile = np.mean(cont_norm, axis=1)
    ax.plot(spatial_profile)
    ax.set_title(f'{label} — spatial profile')
    ax.set_xlabel('Row [px]')
    ax.set_ylabel('Mean correction')
    ax.axhline(1.0, color='k', ls='--', alpha=0.3)

    # Panel 3: spectral profile (row-averaged)
    ax = ax_row[2]
    spectral_profile = np.mean(cont_norm, axis=0)
    if wl is not None and len(wl) == len(spectral_profile):
        ax.plot(wl, spectral_profile)
        ax.set_xlabel('Wavelength [nm]')
    else:
        ax.plot(spectral_profile)
        ax.set_xlabel('Column [px]')
    ax.set_title(f'{label} — spectral profile')
    ax.set_ylabel('Mean correction')
    ax.axhline(1.0, color='k', ls='--', alpha=0.3)


# --- Count how many rows we need ---
n_rows = 0
for fpath in amend_frames.values():
    if fpath.exists():
        n_rows += 1
if ratio_polyfit_path.exists():
    n_rows += 1  # ratio polyfit lower
    n_rows += 1  # raw ratio (for diagnostics)

if n_rows == 0:
    print('No continuum correction files found.')
    exit()

fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows), squeeze=False)

row_idx = 0

# --- amend_spectroflat corrections ---
for label, fpath in amend_frames.items():
    if not fpath.exists():
        print(f'  ⚠ File not found: {fpath.name}')
        continue

    with fits.open(fpath) as hdul:
        print(f'\n{label} — {fpath.name}')
        print(f'  HDUs: {len(hdul)}')
        for j, h in enumerate(hdul):
            shape = h.data.shape if h.data is not None else None
            print(f'  HDU {j}: {h.__class__.__name__}  shape={shape}')

        # HDU 0: amended soft flat, HDU 1: continuum correction, HDU 2: wavelength
        cont = np.array(hdul[1].data)
        if cont.ndim == 3:
            cont = np.mean(cont, axis=0)  # average over states

        wl = None
        if len(hdul) > 2 and hdul[2].data is not None:
            wl = np.array(hdul[2].data)

    cont_norm = cont / np.mean(cont)
    print(f'  Continuum correction shape: {cont.shape}')
    print(f'  Range (normalised): {cont_norm.min():.6f} – {cont_norm.max():.6f}')

    plot_continuum(axes[row_idx], label, cont_norm, wl)
    row_idx += 1

# --- Ratio polyfit continuum correction (lower) ---
if ratio_polyfit_path.exists():
    with fits.open(ratio_polyfit_path) as hdul:
        print(f'\nRATIO POLYFIT LOWER — {ratio_polyfit_path.name}')
        print(f'  HDUs: {len(hdul)}')
        for j, h in enumerate(hdul):
            shape = h.data.shape if h.data is not None else None
            print(f'  HDU {j}: {h.__class__.__name__}  shape={shape}')

        cont_surface = np.array(hdul[0].data)
        ratio_raw = np.array(hdul['RATIO'].data) if 'RATIO' in hdul else None

    print(f'  Fitted surface shape: {cont_surface.shape}')
    print(f'  Range: {cont_surface.min():.6f} – {cont_surface.max():.6f}')

    plot_continuum(axes[row_idx], 'lower (ratio polyfit)', cont_surface)
    row_idx += 1

    # Also show the raw ratio for diagnostics
    if ratio_raw is not None:
        print(f'  Raw ratio shape: {ratio_raw.shape}')
        print(f'  Raw ratio range: {np.nanmin(ratio_raw):.4f} – {np.nanmax(ratio_raw):.4f}')
        plot_continuum(axes[row_idx], 'lower (raw ratio upper/lower)', ratio_raw)
        row_idx += 1
else:
    print(f'\n  ⚠ Ratio polyfit file not found: {ratio_polyfit_path.name}')

plt.tight_layout()
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script: expected center-to-limb intensity along the slit at the observed
telescope positions.

The observed positions are heliographic (Stonyhurst) latitudes on the central
meridian (longitude 0), see scripts/measurement_procedure.py. For each
latitude, the heliocentric angle theta (mu = cos(theta)) is computed with
spherical geometry including the solar B0 angle (heliographic latitude of the
disk center) of the observing date:

    mu = cos(theta) = sin(B0) sin(lat) + cos(B0) cos(lat) cos(lon)

This treats the observer as being at infinite distance; the finite Sun-Earth
distance changes theta by less than the solar angular radius (~0.27 deg).

The slit is oriented north-south along the central meridian. Around each
position, the part of the disk covered by the slit (SLIT_N_PIXELS pixels of
SLIT_PIXEL_SCALE arcsec) is cut out: a point at the projected distance y from
disk center (along the meridian, positive = north) has mu = sqrt(1 - (y/R)^2),
R = solar angular radius, and the slit center is at y = R sin(lat - B0).

The limb darkening I(mu)/I_center (polynomial below) is normalized to its mean
within each slit window, since the measurements cannot be normalized
absolutely. The resulting (discontinuous) curve and the local intensity
gradient along the slit are what can be compared with the measured slit
profiles to check how well the telescope was positioned.

Data comparison: the saved continuum parts of the prepared spectra
(spectra/<line>/<position>/<position>_continuum.fits, written by
scripts/process_formation_height_line_levels.py, axes (wavelength, slit, scan))
are averaged over the continuum wavelengths and over N_SCAN_SEGMENTS groups of
scan positions. Each resulting profile along the slit is normalized to its
mean and fitted with a straight line; the slope per slit pixel is compared
with the model gradient. Which slit end is north is unknown, so the
orientation is derived from the data (sign of the agreement with the model,
whose gradient changes sign between the southern and northern positions).

Usage: run cell by cell.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from sunpy.coordinates import sun

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from themis.datasets.themis_datasets_2025 import get_config
from themis.core.cam_config import cam

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

OBS_DATE = '2025-07-05'  # observing date of configs/formation_dataset_2025-07-05.toml

# Observed positions -> heliographic latitude [deg] (central meridian)
POSITION_TO_LATITUDE = {
    'disk_center': 0,
    'm40': -40,
    'm30': -30,
    '30': 30,
    '40': 40,
    '45': 45,
    'm45': -45,
    'm55': -55,
    '55': 55,
}

# Slit sampling (Sr camera, see cam_config) and number of slit pixels used
# (cropped slit, CROP_CONFIGS in scripts/process_formation_height_line_levels.py)
SLIT_PIXEL_SCALE = 0.096  # arcsec/pixel
SLIT_N_PIXELS = 811

# Data comparison
CONFIG_PATH = 'configs/formation_dataset_2025-07-05.toml'
LINES = ('sr',)  # tuple: note the comma for a single line
N_SCAN_SEGMENTS = 4  # scan positions are split into this many groups
FIT_MARGIN = 10      # slit pixels excluded at both ends for the linear fit
MODEL_LAT_TOLERANCE = 2.0  # deg, model also shown for latitude +- this value

# Limb-darkening polynomial coefficients a_k of I(mu)/I_center = sum a_k mu^k
# Formula from Neckel (2005), Solar Physics 229, 13:
# https://ui.adsabs.harvard.edu/abs/2005SoPh..229...13N/abstract
LIMB_DARKENING_COEFFS = (0.1601, 2.1384, -3.9482, 6.0645, -5.0416, 1.6268)


def limb_darkening(mu):
    """
    Relative intensity I(mu)/I_center.

    I(mu)/I_center = 0.1601 + 2.1384 mu - 3.9482 mu^2 + 6.0645 mu^3
                     - 5.0416 mu^4 + 1.6268 mu^5
    (Neckel 2005, https://ui.adsabs.harvard.edu/abs/2005SoPh..229...13N/abstract)

    Parameters
    ----------
    mu : float or ndarray
        cos(theta), theta = heliocentric angle.
    """
    return np.polynomial.polynomial.polyval(mu, LIMB_DARKENING_COEFFS)


def b0_angle(date: str = OBS_DATE) -> float:
    """Solar B0 angle [deg] (heliographic latitude of the disk center) at `date`."""
    return sun.B0(date).to_value(u.deg)


def solar_radius_arcsec(date: str = OBS_DATE) -> float:
    """Apparent solar radius [arcsec] at `date`."""
    return sun.angular_radius(date).to_value(u.arcsec)


def mu_from_latitude(latitude_deg, longitude_deg=0.0, b0_deg=0.0):
    """
    mu = cos(heliocentric angle) of a point at heliographic latitude/longitude
    (longitude relative to the central meridian), for a disk center at
    heliographic latitude b0_deg. Points on the far side give mu < 0.
    """
    lat, lon, b0 = np.deg2rad(latitude_deg), np.deg2rad(longitude_deg), np.deg2rad(b0_deg)
    return np.sin(b0) * np.sin(lat) + np.cos(b0) * np.cos(lat) * np.cos(lon)


def intensity_from_latitude(latitude_deg, longitude_deg=0.0, b0_deg=0.0):
    """Relative intensity I/I_center at a heliographic latitude/longitude."""
    return limb_darkening(mu_from_latitude(latitude_deg, longitude_deg, b0_deg))


def slit_window(latitude_deg, b0_deg, radius_arcsec,
                n_pixels=SLIT_N_PIXELS, pixel_scale=SLIT_PIXEL_SCALE):
    """
    Limb darkening along a north-south slit centered on the central meridian
    at heliographic latitude `latitude_deg`, normalized to its mean.

    Returns
    -------
    dict with
        'offset_arcsec': position along the slit relative to its center
                         (positive = north)
        'y_arcsec': projected distance from disk center (positive = north)
        'mu': mu along the slit
        'intensity': I/I_center along the slit
        'intensity_norm': intensity normalized to its mean within the slit
        'gradient_per_arcsec', 'gradient_per_pixel': slope of a linear fit of
                         intensity_norm vs. position (towards north)
    """
    offset = (np.arange(n_pixels) - (n_pixels - 1) / 2) * pixel_scale
    y = radius_arcsec * np.sin(np.deg2rad(latitude_deg - b0_deg)) + offset
    mu = np.sqrt(np.clip(1 - (y / radius_arcsec) ** 2, 0, None))
    intensity = limb_darkening(mu)
    intensity_norm = intensity / intensity.mean()
    slope = np.polyfit(offset, intensity_norm, 1)[0]
    return {
        'offset_arcsec': offset,
        'y_arcsec': y,
        'mu': mu,
        'intensity': intensity,
        'intensity_norm': intensity_norm,
        'gradient_per_arcsec': slope,
        'gradient_per_pixel': slope * pixel_scale,
    }


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_continuum(line: str, position: str):
    """
    Load the saved continuum part of a prepared spectrum.

    Returns (data with axes (wavelength, slit, scan), file DATE header).
    """
    config = get_config(line=line, config_path=CONFIG_PATH,
                        auto_discover_files=False, auto_create_dirs=False)
    path = Path(config.directories.reduced) / 'spectra' / line / position / f'{position}_continuum.fits'
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype='float64'), hdul[0].header.get('DATE', 'unknown')


def segment_profiles(continuum: np.ndarray, n_segments: int = N_SCAN_SEGMENTS):
    """
    Average the continuum over wavelength and over `n_segments` groups of scan
    positions; each profile along the slit is normalized to its mean.

    Returns list of (scan indices, normalized profile along the slit).
    """
    slit_scan = continuum.mean(axis=0)  # (slit, scan)
    segments = np.array_split(np.arange(slit_scan.shape[1]), n_segments)
    return [(idx, slit_scan[:, idx].mean(axis=1) / slit_scan[:, idx].mean()) for idx in segments]


def fit_line(profile: np.ndarray, margin: int = FIT_MARGIN):
    """Linear fit of a slit profile vs. pixel index; returns (slope/px, intercept)."""
    x = np.arange(profile.size)
    sel = slice(margin, profile.size - margin)
    slope, intercept = np.polyfit(x[sel], profile[sel], 1)
    return slope, intercept


def analyze_line(line: str, b0: float, radius: float):
    """
    Measured continuum gradients along the slit (per scan segment) and the
    model gradients for all positions of one camera.

    Returns dict position -> {'segments': [(idx, profile, slope, intercept)],
    'slopes', 'model' (slit_window dict), 'date'}.
    """
    pixel_scale = cam[line].pixel_scale
    results = {}
    for position, lat in sorted(POSITION_TO_LATITUDE.items(), key=lambda kv: kv[1]):
        try:
            continuum, date = load_continuum(line, position)
        except FileNotFoundError:
            print(f"  {line} {position}: no continuum file, skipped")
            continue
        segments = [(idx, prof, *fit_line(prof)) for idx, prof in segment_profiles(continuum)]
        n_pix = continuum.shape[1]
        results[position] = {
            'segments': segments,
            'slopes': np.array([s[2] for s in segments]),
            'model': slit_window(lat, b0, radius, n_pix, pixel_scale),
            # model for latitude -/+ MODEL_LAT_TOLERANCE
            'model_range': [slit_window(lat + d, b0, radius, n_pix, pixel_scale)
                            for d in (-MODEL_LAT_TOLERANCE, MODEL_LAT_TOLERANCE)],
            'date': date,
        }
    return results


def latitude_from_gradient(gradient_per_pixel, b0: float, radius: float, pixel_scale: float,
                           n_pixels: int = SLIT_N_PIXELS):
    """
    Invert the model: heliographic latitude [deg] (central meridian) whose
    slit gradient (towards north, per pixel) equals `gradient_per_pixel`.
    The model gradient decreases monotonically with latitude.
    """
    lat_grid = np.linspace(-75, 75, 1501)
    grad_grid = np.array([slit_window(lat, b0, radius, n_pixels, pixel_scale)['gradient_per_pixel']
                          for lat in lat_grid])
    return np.interp(gradient_per_pixel, grad_grid[::-1], lat_grid[::-1])


def slit_orientation(results: dict) -> int:
    """
    +1 if the slit pixel index increases towards north (measured slopes agree
    with the model gradient), -1 if it increases towards south.
    """
    agreement = sum(np.mean(r['slopes']) * r['model']['gradient_per_pixel']
                    for p, r in results.items() if p != 'disk_center')
    return 1 if agreement >= 0 else -1


#%%
if __name__ == '__main__':
    #%%
    b0 = b0_angle(OBS_DATE)
    radius = solar_radius_arcsec(OBS_DATE)
    print(f"{OBS_DATE}: B0 = {b0:+.3f} deg, solar radius = {radius:.1f} arcsec")
    print(f"Slit: {SLIT_N_PIXELS} px x {SLIT_PIXEL_SCALE} arcsec/px = "
          f"{SLIT_N_PIXELS * SLIT_PIXEL_SCALE:.1f} arcsec")
    print(f"Check: I(mu=1)/I_center = {limb_darkening(1.0):.4f}")

    windows = {}
    print(f"\n{'position':12s} {'lat':>6s} {'mu center':>9s} {'mu range':>15s} "
          f"{'I/I_c':>7s} {'grad [1/arcsec]':>16s} {'grad [1/px]':>12s} {'I_norm range':>15s}")
    for position, lat in sorted(POSITION_TO_LATITUDE.items(), key=lambda kv: kv[1]):
        w = slit_window(lat, b0, radius)
        windows[position] = w
        mu_c = mu_from_latitude(lat, b0_deg=b0)
        print(f"{position:12s} {lat:6.1f} {mu_c:9.4f} "
              f"{w['mu'].min():7.4f}-{w['mu'].max():.4f} {limb_darkening(mu_c):7.4f} "
              f"{w['gradient_per_arcsec']:+16.3e} {w['gradient_per_pixel']:+12.3e} "
              f"{w['intensity_norm'].min():7.4f}-{w['intensity_norm'].max():.4f}")

    #%%
    # Limb darkening within each slit window, normalized to the window mean
    fig, ax = plt.subplots(figsize=(9, 6))
    for position, w in windows.items():
        line, = ax.plot(w['mu'], w['intensity_norm'], linewidth=2)
        i_c = np.argmin(np.abs(w['offset_arcsec']))
        ax.annotate(f"{position}\n{w['gradient_per_arcsec']:+.2e}/\"",
                    (w['mu'][i_c], w['intensity_norm'][i_c]),
                    xytext=(0, 8) if POSITION_TO_LATITUDE[position] <= 0 else (0, -28),
                    textcoords='offset points', fontsize=8, ha='center',
                    color=line.get_color())
    ax.axhline(1, linestyle='--', color='gray')
    ax.set_xlabel(r'$\mu = \cos\theta$')
    ax.set_ylabel(r'$I / \langle I \rangle_{slit}$')
    ax.set_title(f'Limb darkening (Neckel 2005) within the slit '
                 f'({SLIT_N_PIXELS} px x {SLIT_PIXEL_SCALE}"), normalized per position')
    ax.grid(linestyle=':')
    plt.tight_layout()
    plt.show()

    #%%
    # Data: continuum gradient along the slit per scan segment vs. model
    data_results = {}
    for line in LINES:
        res = analyze_line(line, b0, radius)
        orient = slit_orientation(res)
        data_results[line] = (res, orient)
        print(f"\n{line.upper()} (pixel scale {cam[line].pixel_scale}\"/px): slit pixel index "
              f"increases towards {'north' if orient > 0 else 'south'}")
        print(f"{'position':12s} {'model [1/px]':>13s} {'measured mean [1/px]':>21s} "
              f"{'std':>10s} {'meas/model':>10s} {'implied lat [deg]':>18s}  "
              f"segments [1/px]                     file date")
        for position, r in res.items():
            model = orient * r['model']['gradient_per_pixel']
            mean, std = r['slopes'].mean(), r['slopes'].std()
            ratio = mean / model if position != 'disk_center' else np.nan
            # measured slope in the "towards north" convention of the model
            lat_impl = latitude_from_gradient(orient * np.array([mean - std, mean, mean + std]),
                                              b0, radius, cam[line].pixel_scale,
                                              r['model']['intensity_norm'].size)
            r['implied_latitude'] = lat_impl[1]
            r['implied_latitude_range'] = (min(lat_impl[0], lat_impl[2]), max(lat_impl[0], lat_impl[2]))
            lat_str = f"{lat_impl[1]:+6.1f} ({min(lat_impl[0], lat_impl[2]):+.0f}..{max(lat_impl[0], lat_impl[2]):+.0f})"
            segs = ' '.join(f"{s:+.2e}" for s in r['slopes'])
            print(f"{position:12s} {model:+13.3e} {mean:+21.3e} {std:10.2e} {ratio:10.2f} "
                  f"{lat_str:>18s}  {segs}  {r['date'][:10]}")

    #%%
    # Linear fits along the slit per scan segment and model (+- MODEL_LAT_TOLERANCE)
    def _oriented(profile, orient):
        return profile if orient > 0 else profile[::-1]

    for line, (res, orient) in data_results.items():
        positions = list(res)
        n_cols = 3
        n_rows = int(np.ceil(len(positions) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows), squeeze=False)
        for ax, position in zip(axes.flat, positions):
            r = res[position]
            x = np.arange(r['model']['intensity_norm'].size)
            for idx, prof, slope, intercept in r['segments']:
                ax.plot(x, slope * x + intercept, '--',
                        label=f"fit scan {idx[0]}-{idx[-1]}: {slope:+.2e}/px")
            lo, hi = (_oriented(m['intensity_norm'], orient) for m in r['model_range'])
            ax.fill_between(x, np.minimum(lo, hi), np.maximum(lo, hi), color='gray', alpha=0.3,
                            label=f"model lat ±{MODEL_LAT_TOLERANCE:g}°: "
                                  + '..'.join(f"{orient * m['gradient_per_pixel']:+.2e}"
                                              for m in r['model_range']) + '/px')
            ax.plot(x, _oriented(r['model']['intensity_norm'], orient), 'k', linewidth=2,
                    label=f"model: {orient * r['model']['gradient_per_pixel']:+.2e}/px")
            lo_lat, hi_lat = r['implied_latitude_range']
            ax.set_title(f"{line.upper()} {position} (lat {POSITION_TO_LATITUDE[position]:+d}°): "
                         f"implied lat {r['implied_latitude']:+.1f}° ({lo_lat:+.0f}..{hi_lat:+.0f}°)")
            ax.set_xlabel('slit pixel')
            ax.set_ylabel(r'$I / \langle I \rangle$')
            ax.grid(linestyle=':')
            ax.legend(fontsize=7)
        for ax in axes.flat[len(positions):]:
            ax.set_visible(False)
        fig.suptitle(f"{line.upper()}: linear fits of the continuum along the slit "
                     f"({N_SCAN_SEGMENTS} scan segments) vs. limb-darkening model "
                     f"(±{MODEL_LAT_TOLERANCE:g}°, pixel index towards "
                     f"{'north' if orient > 0 else 'south'})")
        plt.tight_layout()
        plt.show()

    #%%
    # Summary: measured vs. model gradient
    fig, ax = plt.subplots(figsize=(7, 7))
    for line, (res, orient) in data_results.items():
        model = np.array([orient * r['model']['gradient_per_pixel'] for r in res.values()])
        model_range = np.array([[orient * m['gradient_per_pixel'] for m in r['model_range']]
                                for r in res.values()])
        xerr = np.abs(np.array([model - model_range.min(axis=1), model_range.max(axis=1) - model]))
        slopes = np.array([r['slopes'] for r in res.values()])
        eb = ax.errorbar(model, slopes.mean(axis=1), xerr=xerr, yerr=slopes.std(axis=1), fmt='o',
                         capsize=3, label=f"{line.upper()} (x error: lat ±{MODEL_LAT_TOLERANCE:g}°)")
        for position, m, s in zip(res, model, slopes.mean(axis=1)):
            ax.annotate(f"{position}: {res[position]['implied_latitude']:+.1f}°", (m, s),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color=eb[0].get_color())
    lim = np.max(np.abs(ax.get_xlim() + ax.get_ylim()))
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8, label='measured = model')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('model gradient [1/px]')
    ax.set_ylabel(f'measured gradient [1/px] (mean ± std of {N_SCAN_SEGMENTS} scan segments)')
    ax.set_title('Continuum gradient along the slit (labels: implied latitude)')
    ax.grid(linestyle=':')
    ax.legend()
    plt.tight_layout()
    plt.show()

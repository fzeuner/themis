#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script: Fe camera (cam3, ~630.2 nm) L4 scan data — first step towards
Doppler-shift analysis.

Loads the L4 scan data of the Fe camera for the positions +-30, +-45 and +-55
deg heliographic latitude (observation date 2025-07-05) with
tio.read_any_file(config, 'scan', status='l4'), the same reading path used by
reduce_l3_to_4 in themis_data_reduction.py. Positions whose L4 file does not
exist yet are skipped with a message, so the script can be re-run while more
data is being reduced.

For every position the mean spectrum of the UPPER frame is computed by
averaging over the full scan and the slit (leaving out SLIT_MARGIN pixels at
both slit ends), normalized to its own continuum. The spectral window is
restricted to the spectral part of the atlas-fit input.roi (cols 5:-5)
because the L4 edge columns of the upper frame contain huge
continuum-correction artifacts (values up to ~1e7).

Each mean spectrum is then fitted with four lines plus a constant background
(model from the original fit script at the bottom of this file's history):
the two Fe I lines (6301.5 / 6302.5 A) as plain Gaussians and the two
telluric O2 lines (6302.0 / 6302.8 A) as a Lorentzian convolved with a
rectangular slit function whose width is a free parameter shared between
both tellurics. The disk_center spectrum (fallback: flat_center L4 while the
disk_center scan has no L4 yet) is fitted the same way. One figure is made
per position: the data, the fit, the fitted line centers (values in the
legend), and the disk_center data and its fit drawn faintly for comparison.
The fitted line centers are printed as a table, including the offset to the
disk_center fit in mA and km/s.

Usage: run cell by cell.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'atlas-fit' / 'src'))

from themis.datasets.themis_datasets_2025 import get_config, _build_file_set
from themis.core import themis_io as tio
from themis.core import stray_light as sl

import yaml
from atlas_fit.atlantes import atlas_factory
from atlas_fit.base.constants import NM2ANGSTROM
from atlas_fit.models.differential_convolution import FWHM, DifferentialSpectralPSF

from lmfit.models import GaussianModel, ConstantModel
from lmfit import Model

import astropy.units as u
import astropy.constants as const

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

CONFIG_PATH = 'configs/formation_dataset_2025-07-05.toml'
LINE = 'fe'

# Positions and their scan sequences
# (see POSITION_TO_SEQUENCE in scripts/process_formation_height_line_levels.py)
POSITION_TO_SEQUENCE = {
    'm55': 92,
    'm45': 48,
    'm30': 18,
    '30': 50,
    'm40': 24,
    '45': 38,
    '55': 86,
}

HALF = 'upper'   # only the upper frame is used

# disk_center scan sequence (fallback: flat_center L4 while it has no L4 yet)
DISK_CENTER_SEQUENCE = 27

# Slit (spatial) rows trimmed at both ends when averaging over the slit, to
# avoid averaging over edge artifacts
SLIT_MARGIN = 10

# Continuum level of a mean spectrum = this percentile within the spectral roi
CONTINUUM_PERCENTILE = 95

# Apply the global stray-light correction (levels from atlas_fit_config_cam3:
# upper 0.38 %, lower 0.4 %)
REMOVE_STRAY_LIGHT = False

# Line names in the fit, and colors for the center markers
LINE_NAMES = ('fei_1', 'telluric_1', 'fei_2', 'telluric_2')
LINE_COLORS = {'fei_1': 'tab:blue', 'telluric_1': 'tab:orange',
               'fei_2': 'tab:green', 'telluric_2': 'tab:red'}

C_KMS = 299792.458   # speed of light [km/s], for the Doppler offset printout

# Solar B0 angle on 2025-07-05 (see test_telescope_position.py)
B0_DEG = 3.303

# Sidereal differential rotation law
#   omega(lat) = A + B sin^2(lat) + C sin^4(lat)   [urad/s]
# spectroscopic/photospheric values, Snodgrass & Ulrich (1990)
ROTATION_COEFFS_URADS = (2.902, -0.464, -0.328)

# --- Sun panel --------------------------------------------------------------
OBS_DATE = '2025-07-05'
R_SOLAR_ARCSEC = 943.5            # apparent solar radius on OBS_DATE
BOX_HEIGHT_ARCSEC = 77.0          # slit length N-S (as in measurement_procedure)
BOX_WIDTH_ARCSEC = 141 * 0.33     # scan width E-W (141 steps x 0.33" slit width)

# Implied heliographic latitude (mean +- half-range over the 4 scan segments)
# from the continuum-gradient analysis in test_telescope_position.py
IMPLIED_LATITUDE = {
    'm55': (-56.5, 1.5), 'm45': (-47.8, 0.5), 'm40': (-40.8, 1.0),
    'm30': (-26.3, 5.5), '30': (38.3, 6.0), '40': (2.7, 8.5),
    '45': (43.0, 3.5), '55': (53.7, 2.5),
}


def differential_rotation(lat_deg) -> u.Quantity:
    """Sidereal rotation rate omega(lat) at heliographic latitude [rad/s]."""
    s2 = np.sin(np.deg2rad(lat_deg)) ** 2
    a, b, c = ROTATION_COEFFS_URADS
    return (a + b * s2 + c * s2 ** 2) * 1e-6 * (u.rad / u.s)


def implied_longitude_offset(v_los_kms, lat_deg):
    """
    Longitude offset [deg] from the central meridian that would produce the
    measured line-of-sight velocity through solar (differential) rotation:

        v_los = omega(lat) * R_sun * cos(lat) * cos(B0) * sin(dlon)

    On the central meridian (dlon = 0) rotation gives no line-of-sight
    component, so a non-zero Doppler shift implies an east/west pointing
    offset. Positive dlon = displaced toward the receding side of the disk.
    |v_los| larger than the projected rotation speed is clipped to +-90 deg
    with a warning (the shift cannot come from rotation alone).
    """
    omega = differential_rotation(lat_deg).to_value(u.rad / u.s)   # [rad/s]
    v_rot = (omega * const.R_sun.to_value(u.km)
             * np.cos(np.deg2rad(lat_deg)) * np.cos(np.deg2rad(B0_DEG)))
    ratio = np.asarray(v_los_kms) / v_rot   # [km/s] / [km/s]
    if np.any(np.abs(ratio) > 1):
        print('  Warning: |v_los| exceeds the projected rotation velocity at '
              f'lat {lat_deg} deg — cannot be explained by rotation alone, '
              'dlon clipped to +-90 deg')
    return np.rad2deg(np.arcsin(np.clip(ratio, -1, 1)))


def projected_rotation(lat_deg):
    """Projected equatorial rotation velocity omega*R*cos(lat)*cos(B0) [km/s]."""
    omega = differential_rotation(lat_deg).to_value(u.rad / u.s)
    return (omega * const.R_sun.to_value(u.km)
            * np.cos(np.deg2rad(lat_deg)) * np.cos(np.deg2rad(B0_DEG)))


def separation_stderr(result) -> dict:
    """1-sigma error of each fei_n - telluric_n separation [A], propagated
    from the lmfit stderrs of the two fitted centers."""
    out = {}
    for pair, fei, tel in (('fei_1-telluric_1', 'fei_1', 'telluric_1'),
                           ('fei_2-telluric_2', 'fei_2', 'telluric_2')):
        e_f = result.params[fei + '_center'].stderr or np.nan
        e_t = result.params[tel + '_center'].stderr or np.nan
        out[pair] = np.hypot(e_f, e_t)
    return out


def implied_position(position: str, result, ref_result,
                     centers: dict = None, ref_centers: dict = None):
    """
    Implied measurement center for one position: the longitude offset is the
    mean over the two Fe-telluric pairs (relative to the reference fit
    `ref_result`, e.g. disk_center), the latitude is taken from
    IMPLIED_LATITUDE (test_telescope_position.py), falling back to the
    nominal latitude from the position name. `centers`/`ref_centers` are the
    (parabola-refined) line centers used for the separations; they default to
    the fitted centers of `result`/`ref_result`. Returns a dict with lon/lat
    and their 1-sigma errors [deg].
    """
    ref_centers = ref_centers or line_centers(ref_result)
    pairs = tuple(line_separations(ref_centers))
    fei_of_pair = {'fei_1-telluric_1': 'fei_1', 'fei_2-telluric_2': 'fei_2'}
    nominal_lat = (0 if position == 'disk_center'
                   else int(position.replace('m', '-')))
    seps = line_separations(centers or line_centers(result))
    seps_ref = line_separations(ref_centers)
    sep_err = separation_stderr(result)
    sep_err_ref = separation_stderr(ref_result)
    v_rot = projected_rotation(nominal_lat)

    dlons, errs = [], []
    for pair in pairs:
        dv = (seps[pair] - seps_ref[pair]) / ref_centers[fei_of_pair[pair]] * C_KMS
        dlons.append(float(implied_longitude_offset(dv, nominal_lat)))
        e_dv = (np.hypot(sep_err[pair], sep_err_ref[pair])
                / ref_centers[fei_of_pair[pair]] * C_KMS)
        errs.append(e_dv / (v_rot * abs(np.cos(np.deg2rad(dlons[-1])))))

    lat, lat_err = IMPLIED_LATITUDE.get(position, (nominal_lat, np.nan))
    return {'lon': float(np.mean(dlons)),
            'lon_err': float(np.hypot(np.hypot(*errs) / 2,
                                      np.std(dlons) if len(dlons) > 1 else 0)),
            'lat': lat, 'lat_err': lat_err}


def load_l4_scan(sequence: int):
    """
    Return (config, L4 scan CycleSet, wavelength [nm]) for a scan sequence.

    Same reading path as reduce_l3_to_4: the file set is built for the given
    sequence and the L4 file is read with tio.read_any_file(..., status='l4').
    Returns (config, None, wl) if the L4 file does not exist yet.
    """
    config = get_config(line=LINE, config_path=CONFIG_PATH,
                        auto_discover_files=True, auto_create_dirs=False)
    config.dataset['scan']['sequence'] = sequence
    config.dataset['scan']['files'] = _build_file_set(
        config.directories, config.dataset['scan'], config.data_types,
        config.reduction_levels, config.cam, config.line,
        yshift_calibration_date=config.yshift_calibration_date,
    )
    wl, _ = tio.read_any_file(config, 'scan', verbose=False, status='wl')

    l4_file = config.dataset['scan']['files'].get('l4')
    if l4_file is None or not Path(l4_file).exists():
        return config, None, np.asarray(wl, dtype='float64')

    data, _ = tio.read_any_file(config, 'scan', verbose=False, status='l4')
    return config, data, np.asarray(wl, dtype='float64')


def load_flat_center():
    """Return (config, L4 flat_center FramesSet, wavelength [nm])."""
    config = get_config(line=LINE, config_path=CONFIG_PATH,
                        auto_discover_files=True, auto_create_dirs=False)
    data, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='l4')
    wl, _ = tio.read_any_file(config, 'flat_center', verbose=False, status='wl')
    return config, data, np.asarray(wl, dtype='float64')


def load_disk_center(remove_stray_light: bool = REMOVE_STRAY_LIGHT):
    """
    Load the disk_center L4 scan; falls back to the flat_center L4 data while
    the disk_center scan has no L4 file yet. Returns a dict like the entries
    of load_all_positions plus a 'kind' key ('scan' or 'flat_center').
    """
    config, data, wl = load_l4_scan(DISK_CENTER_SEQUENCE)
    kind = 'scan'
    if data is None:
        print(f"  disk_center scan (seq={DISK_CENTER_SEQUENCE}) has no L4 file "
              f"yet — using flat_center instead")
        config, data, wl = load_flat_center()
        kind = 'flat_center'
    levels = sl.get_stray_light_levels(config)
    if remove_stray_light:
        sl.remove_stray_light_from_set(data, levels, inplace=True)
    print(f"  disk_center ({kind}): loaded L4")
    return {'data': data, 'wl': wl, 'roi': levels['spectral_roi'],
            'kind': kind, 'config': config}


def fts_atlas(config, wl_min: float, wl_max: float):
    """
    Hamburg FTS disk-center atlas in [wl_min, wl_max] nm (atlas_factory 'fts',
    as in test_remove_straylight.convolved_atlas).

    Returns (wl [nm], full-res intensity, convolved intensity, mean continuum).
    """
    with open(config.cam.atlas_fit_config, 'r') as f:
        atlas_cfg = yaml.safe_load(f)
    atlas = atlas_factory('fts', wl_min, wl_max, conversion=NM2ANGSTROM)
    fwhm = FWHM(atlas=atlas.fwhm, data=atlas_cfg['input']['fwhm'])
    convolved = DifferentialSpectralPSF(fwhm).convolve(atlas.intensity, atlas.wl)
    return atlas.wl, atlas.intensity, convolved, np.mean(atlas.continuum)


def mean_spectrum(frame: np.ndarray, spec_roi: slice) -> np.ndarray:
    """Average a frame half (spatial, spectral) over the slit within spec_roi."""
    return np.nanmean(frame[SLIT_MARGIN:-SLIT_MARGIN, spec_roi], axis=0)


def position_spectrum(entry) -> np.ndarray:
    """
    Normalized mean spectrum of the upper frame: averaged over all scan
    positions (or all flat frames) and the slit, divided by its own continuum
    (CONTINUUM_PERCENTILE).
    """
    spec = np.mean([mean_spectrum(h.data, entry['roi'])
                    for h in entry['data'].get_all_halves(HALF)], axis=0)
    return spec / np.percentile(spec, CONTINUUM_PERCENTILE)


def load_all_positions(positions=None, remove_stray_light: bool = REMOVE_STRAY_LIGHT):
    """
    Load the L4 scan of every position in `positions` (default: all in
    POSITION_TO_SEQUENCE). Skips positions without an L4 file.

    Returns dict position -> {'data': CycleSet, 'wl': nm, 'roi': slice}.
    """
    results = {}
    for position, sequence in (positions or POSITION_TO_SEQUENCE).items():
        config, data, wl = load_l4_scan(sequence)
        if data is None:
            print(f"FE {position} (seq={sequence}): no L4 file yet, skipped")
            continue
        levels = sl.get_stray_light_levels(config)
        roi = levels['spectral_roi']
        if remove_stray_light:
            sl.remove_stray_light_from_set(data, levels, inplace=True)
        print(f"FE {position} (seq={sequence}): loaded L4 "
              f"(stray light upper {levels['upper']} %, lower {levels['lower']} %, "
              f"spec roi {roi})")
        results[position] = {'data': data, 'wl': wl, 'roi': roi,
                             'kind': 'scan', 'config': config}
    return results


# ---------------------------------------------------------------------------
# Custom model: Lorentzian convolved with a rectangular (boxcar) function.
#
# The rectangular function has width 'rect_width' and is normalized to unit
# area:  R(x) = 1/rect_width  for |x| < rect_width/2,  else 0.
#
# The normalized Lorentzian (with 'sigma' = HWHM) is:
#   L(x) = (1/pi) * sigma / ((x - center)^2 + sigma^2)
#
# Convolving it with the rectangle gives an analytic result using arctan:
#
#   f(x) = (amplitude * sigma) / (pi * rect_width) *
#          [ arctan((x - center + rect_width/2) / sigma)
#          - arctan((x - center - w2) / sigma) ]
#
# 'amplitude' is the integrated area (same convention as LorentzianModel).
# ---------------------------------------------------------------------------
def lorentzian_conv_rect(x, amplitude, center, sigma, rect_width):
    w2 = rect_width / 2
    return (amplitude * sigma / (np.pi * rect_width)) * (
        np.arctan((x - center + w2) / sigma)
        - np.arctan((x - center - w2) / sigma)
    )


def build_line_model(y):
    """
    Build the 4-line model and the initial parameters (from the original fit
    script, wavelength unit: Angstrom):
    two Fe I lines as plain Gaussians, two telluric lines as a Lorentzian
    convolved with a rectangular slit function (rect_width shared), plus a
    constant background at the continuum level.
    """
    fei_1 = GaussianModel(prefix='fei_1_')
    fei_2 = GaussianModel(prefix='fei_2_')
    telluric_1 = Model(lorentzian_conv_rect, prefix='telluric_1_')
    telluric_2 = Model(lorentzian_conv_rect, prefix='telluric_2_')
    background = ConstantModel()
    model = fei_1 + telluric_1 + fei_2 + telluric_2 + background

    params = model.make_params()

    # manually set the center of each line [Angstrom]
    params['fei_1_center'].set(value=6301.5)
    params['telluric_1_center'].set(value=6302.0, min=6301.88, max=6302.08)
    params['fei_2_center'].set(value=6302.5)
    params['telluric_2_center'].set(value=6302.8, min=6302.68, max=6302.86)

    # rough guesses for the other parameters: in GaussianModel 'amplitude' is
    # the integrated area, the visible line DEPTH is the 'height' =
    # amplitude / (sigma * sqrt(2*pi)) -> depth is set, amplitude derived
    height = -0.4  # line depth for all lines (negative = absorption)

    # Fe I lines: broader, plain Gaussian
    sigma_fei = 0.05
    amplitude_fei = height * sigma_fei * np.sqrt(2 * np.pi)
    for prefix in ['fei_1_', 'fei_2_']:
        params[prefix + 'sigma'].set(value=sigma_fei, min=1e-3, max=1.0)
        params[prefix + 'amplitude'].set(value=amplitude_fei, min=-1e6, max=1e6)

    # Telluric lines: intrinsically very sharp Lorentzian, convolved with the
    # rectangular slit function. The convolution lowers the peak, so
    # rect_width is accounted for when converting height -> amplitude:
    #   f(center) = (amplitude * sigma / (pi * w)) * 2 * arctan(w / (2*sigma))
    sigma_telluric = 0.005
    rect_width0 = 0.02  # initial guess for the slit width [Angstrom]
    amplitude_telluric = (height * np.pi * rect_width0
                          / (2 * sigma_telluric * np.arctan(rect_width0 / (2 * sigma_telluric))))
    for prefix in ['telluric_1_', 'telluric_2_']:
        params[prefix + 'sigma'].set(value=sigma_telluric, min=1e-4, max=0.05)
        params[prefix + 'amplitude'].set(value=amplitude_telluric, min=-1e6, max=1e6)
        params[prefix + 'rect_width'].set(value=rect_width0, min=1e-4, max=0.5)

    # Constrain both tellurics to share the same rect_width
    params['telluric_2_rect_width'].set(expr='telluric_1_rect_width')

    # constant background: the continuum level
    params['c'].set(value=np.median(y))

    return model, params


def fit_spectrum(wl_nm: np.ndarray, spec: np.ndarray):
    """
    Fit the 4-line model to a normalized mean spectrum. The wavelength axis
    is converted to Angstrom so the initial guesses of the original fit
    script apply unchanged. Returns (lmfit result, wavelength [A]).
    """
    x = wl_nm * NM2ANGSTROM
    model, params = build_line_model(spec)
    result = model.fit(spec, params, x=x)
    return result, x


def plot_fit(position: str, x: np.ndarray, spec: np.ndarray, result,
             dc: dict = None, refined: dict = None):
    """
    One figure per position: data, total fit, the fitted center of each line
    (values in the legend, as in the original fit script; dotted lines show
    the parabola-refined centers if `refined` is given), and the disk_center
    data and its fit drawn faintly for comparison.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, spec, 'k-', label=f'{position} data')
    ax.plot(x, result.best_fit, 'r-', label='total fit')

    for name in LINE_NAMES:
        center = result.params[name + '_center'].value
        label = rf'{name}: center = {center:.4f} $\AA$'
        if refined is not None:
            label += rf' → {refined[name]:.4f}'
        ax.axvline(center, color=LINE_COLORS[name], linestyle='--',
                   label=label)
        if refined is not None:
            ax.axvline(refined[name], color=LINE_COLORS[name],
                       linestyle=':', linewidth=1)
    rect_w = result.params['telluric_1_rect_width'].value
    ax.plot([], [], ' ', label=rf'shared rect_width = {rect_w:.4f} $\AA$')

    if dc is not None and position != 'disk_center':
        ax.plot(dc['x'], dc['spec'], color='gray', linewidth=0.7, alpha=0.6,
                label='disk_center data')
        ax.plot(dc['x'], dc['result'].best_fit, color='gray', linewidth=0.7,
                alpha=0.6, linestyle='--', label='disk_center fit')

    ax.set_xlabel(r'Wavelength [$\AA$]')
    ax.set_ylabel('Intensity (normalized)')
    ax.set_title(f'FE {position} — fitted spectrum')
    ax.legend()
    ax.grid(linestyle=':')
    plt.show()


def line_centers(result) -> dict:
    """Fitted line centers {name: center [A]} of a lmfit result."""
    return {name: result.params[name + '_center'].value for name in LINE_NAMES}


# Half width [px] of the window used for the parabola refinement of each
# fitted line center (the spectrum minimum in this window is fitted with a
# parabola and its vertex is used as the refined center)
PARABOLA_HALF_WIDTH_PX = 3


def refine_center(x: np.ndarray, spec: np.ndarray, x0: float,
                  half_width: int = PARABOLA_HALF_WIDTH_PX) -> float:
    """
    Refine a fitted line center [A] with a parabola fit to the actual
    spectrum minimum: the local minimum within +-half_width px of x0 is
    found, a parabola is fitted over +-half_width px around it and the
    parabola vertex is returned. This removes any small systematic pull of
    the global model fit on the measured line position.
    """
    i0 = int(np.argmin(np.abs(x - x0)))
    lo, hi = max(i0 - half_width, 0), min(i0 + half_width + 1, x.size)
    i_min = lo + int(np.argmin(spec[lo:hi]))
    lo, hi = max(i_min - half_width, 0), min(i_min + half_width + 1, x.size)
    xs, ys = x[lo:hi], spec[lo:hi]
    if xs.size < 3:
        return float(x0)
    a, b, _ = np.polyfit(xs, ys, 2)
    return float(-b / (2 * a)) if a > 0 else float(xs[int(np.argmin(ys))])


def refine_centers(x: np.ndarray, spec: np.ndarray, result) -> dict:
    """Parabola-refined line centers {name: center [A]} for all LINE_NAMES."""
    return {name: refine_center(x, spec, result.params[name + '_center'].value)
            for name in LINE_NAMES}


def line_separations(centers: dict) -> dict:
    """Line separations fei_n - telluric_n [A] (Fe position relative to the
    telluric rest frame)."""
    return {'fei_1-telluric_1': centers['fei_1'] - centers['telluric_1'],
            'fei_2-telluric_2': centers['fei_2'] - centers['telluric_2']}


def _sphere(lon_deg, lat_deg):
    """
    Cartesian coords on the unit sphere for heliographic (lon, lat) [deg]:
    +x points toward the observer (sub-observer point at lon=0, lat=B0),
    +z is the solar north pole, positive lon is toward the receding side.
    """
    lo, la = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    return np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)


def _view2d(lon_deg, lat_deg):
    """
    Orthographic projection of a sphere point onto the observer's plane of
    the sky. Returns (east, north, depth) in units of R_sun; depth > 0 marks
    the visible hemisphere. North is up, east is to the right.
    """
    x, y, z = _sphere(lon_deg, lat_deg)
    b0 = np.deg2rad(B0_DEG)
    east = y
    north = -np.sin(b0) * x + np.cos(b0) * z
    depth = np.cos(b0) * x + np.sin(b0) * z
    return east, north, depth


def _plot_view_line(ax, lons, lats, **kwargs):
    """Draw a curve on the Sun; only the part on the visible hemisphere is
    drawn (orthographic projection)."""
    east, north, depth = _view2d(np.asarray(lons, dtype=float),
                                 np.asarray(lats, dtype=float))
    idx = np.where(depth > 0)[0]
    for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if len(seg):
            ax.plot(east[seg], north[seg], **kwargs)


def draw_sun_panel(ax, implied: dict, positions, pos_colors: dict):
    """
    Sun seen from Earth (orthographic projection, tilted by B0) with a
    latitude/longitude grid every 30 deg. The intended slit boxes on the
    central meridian (dashed) and the implied measurement boxes (solid, with
    an error cross at the center) are drawn as the flat rectangles the
    observer always sees (as in measurement_procedure.py), each position in
    its own color. `implied` maps position -> dict from implied_position
    (plus 'disk_center', which is referenced to the FTS atlas).
    """
    # limb
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='k', linewidth=1))

    # latitude/longitude grid every 30 deg
    for lat_g in range(-60, 61, 30):
        _plot_view_line(ax, np.linspace(-180, 180, 361),
                        np.full(361, lat_g), color='gray', linewidth=0.4,
                        alpha=0.7)
    for lon_g in range(-150, 181, 30):
        _plot_view_line(ax, np.full(181, lon_g),
                        np.linspace(-90, 90, 181), color='gray',
                        linewidth=0.4, alpha=0.7)
    # emphasize equator and central meridian
    _plot_view_line(ax, np.linspace(-180, 180, 361), np.zeros(361),
                    color='k', linewidth=0.8)
    _plot_view_line(ax, np.zeros(181), np.linspace(-90, 90, 181),
                    color='k', linewidth=0.8)

    w = BOX_WIDTH_ARCSEC / R_SOLAR_ARCSEC
    h = BOX_HEIGHT_ARCSEC / R_SOLAR_ARCSEC

    # intended positions: slit boxes centered on the central meridian
    for position in positions:
        lat_d = 0 if position == 'disk_center' else int(position.replace('m', '-'))
        e, n, _ = _view2d(0, lat_d)
        ax.add_patch(plt.Rectangle((e - w / 2, n - h / 2), w, h, fill=False,
                                   edgecolor=pos_colors[position],
                                   linestyle='--', linewidth=0.8, alpha=0.6))

    # implied positions: box + error cross along lat and lon
    for position, r in implied.items():
        color = pos_colors[position]
        e, n, _ = _view2d(r['lon'], r['lat'])
        ax.add_patch(plt.Rectangle((e - w / 2, n - h / 2), w, h, fill=False,
                                   edgecolor=color, linewidth=1.5))
        if np.isfinite(r['lon_err']):
            _plot_view_line(ax, np.linspace(r['lon'] - r['lon_err'],
                                            r['lon'] + r['lon_err'], 30),
                            np.full(30, r['lat']), color=color, linewidth=1.2)
        if np.isfinite(r['lat_err']):
            _plot_view_line(ax, np.full(30, r['lon']),
                            np.linspace(r['lat'] - r['lat_err'],
                                        r['lat'] + r['lat_err'], 30),
                            color=color, linewidth=1.2)
        ax.annotate(f"{position} ({r['lon']:+.0f}°, {r['lat']:+.0f}°)",
                    (e, n), xytext=(4, 4), textcoords='offset points',
                    color=color, fontsize='x-small')

    ax.set_aspect('equal')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r'east [$R_\odot$]')
    ax.set_ylabel(r'north [$R_\odot$]')
    ax.set_title('intended (dashed) vs. implied slit positions\n'
                 '(lon offset, lat offset)')


def plot_separation_summary(centers: dict, fts_centers: dict,
                            results: dict = None,
                            reference: str = 'disk_center'):
    """
    Summary plot: for every position, the change of the line separations
    fei_1 - telluric_1 and fei_2 - telluric_2 relative to the same separation
    at disk_center and relative to the FTS disk-center atlas, converted first
    to a Doppler shift in km/s (v = dlambda/lambda * c, with lambda the fei_n
    center of the respective reference) and then to the implied longitude
    offset from the central meridian in degrees (implied_longitude_offset,
    differential rotation from the Snodgrass & Ulrich 1990 law and the
    latitude given in the position name). Left axis: implied longitude
    offset; right axis (twin): the same curves as Doppler shift in km/s.
    A vertical line at latitude 0 separates the northern from the southern
    positions.

    `reference` chooses the reference for the plotted curves and the implied
    positions: 'disk_center', 'FTS', or 'both' (curves only; the Sun panel
    then uses disk_center). For the Sun panel, `results` must map position ->
    lmfit result and contain 'disk_center' and 'FTS'. The disk_center implied
    position is always referenced to FTS (the only available reference).
    """
    pairs = tuple(line_separations(centers['disk_center']))
    pair_colors = {'fei_1-telluric_1': 'tab:blue', 'fei_2-telluric_2': 'tab:green'}
    fei_of_pair = {'fei_1-telluric_1': 'fei_1', 'fei_2-telluric_2': 'fei_2'}
    ref_styles = {'disk_center': ('o-', 'disk_center'),
                  'FTS': ('s--', 'FTS')}
    refs = {'disk_center': centers['disk_center'], 'FTS': fts_centers}
    if reference != 'both':
        ref_styles = {reference: ref_styles[reference]}

    def lat(position):
        return 0 if position == 'disk_center' else int(position.replace('m', '-'))

    positions = sorted((p for p in centers if p != 'disk_center'), key=lat)
    x = np.array([lat(p) for p in positions])

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0])
    ax_doppler = ax.twinx()   # second y axis: Doppler shift in km/s
    for pair in pairs:
        sep_pos = [line_separations(centers[p])[pair] for p in positions]
        for ref_name, (style, label) in ref_styles.items():
            sep_ref = line_separations(refs[ref_name])[pair]
            wl_ref = refs[ref_name][fei_of_pair[pair]]   # fei center [A]
            doppler = (np.array(sep_pos) - sep_ref) / wl_ref * C_KMS
            dlon = implied_longitude_offset(doppler, x)
            ax.plot(x, dlon, style, color=pair_colors[pair],
                    label=f'{pair} − {label}')
            ax_doppler.plot(x, doppler, style, color=pair_colors[pair],
                            alpha=0.3, linewidth=0.8, markersize=3)

    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)   # latitude 0: north | south
    ax.set_xlabel('heliographic latitude [deg]')
    ax.set_ylabel('implied longitude offset [deg]')
    ax_doppler.set_ylabel('Doppler shift [km/s]')
    ax.set_title('Implied east/west offset from the central meridian '
                 '(left: deg, right: km/s)\n'
                 '(Fe − telluric Doppler shift interpreted as rotation)')
    ax.grid(linestyle=':')
    ax.legend(fontsize='small', loc='center left')

    ax_sun = fig.add_subplot(gs[1])
    if results is not None and {'disk_center', 'FTS'} <= set(results):
        # one color per |latitude| (m45 and 45 share a color), disk_center
        # first; intended boxes are drawn for all defined positions
        all_pos = (['disk_center'] +
                   sorted(POSITION_TO_SEQUENCE, key=lat))
        abs_lats = sorted({abs(lat(p)) for p in all_pos})
        lat_colors = {a: c for a, c in zip(abs_lats, plt.cm.tab10.colors)}
        pos_colors = {p: lat_colors[abs(lat(p))] for p in all_pos}
        ref_name = 'disk_center' if reference == 'both' else reference
        ref_ctr = (centers['disk_center'] if ref_name == 'disk_center'
                   else fts_centers)
        implied = {p: implied_position(p, results[p], results[ref_name],
                                       centers=centers[p],
                                       ref_centers=ref_ctr)
                   for p in positions}
        # disk_center can only be referenced to the FTS atlas
        implied['disk_center'] = implied_position(
            'disk_center', results['disk_center'], results['FTS'],
            centers=centers['disk_center'], ref_centers=fts_centers)
        draw_sun_panel(ax_sun, implied, all_pos, pos_colors)
    else:
        ax_sun.set_title('Sun (no fit results provided)')
        ax_sun.set_axis_off()

    plt.tight_layout()
    plt.show()


#%%
if __name__ == '__main__':
    #%%
    scans = load_all_positions()
    #%%
    disk_center = load_disk_center()
    #%%
    # Mean spectra (full scan + slit, upper frame, continuum-normalized)
    spectra = {p: position_spectrum(r) for p, r in scans.items()}
    dc_spec = position_spectrum(disk_center)
    dc_wl = disk_center['wl'][disk_center['roi']]
    #%%
    # Fit the 4-line model to every position and to disk_center
    fits = {p: fit_spectrum(scans[p]['wl'][scans[p]['roi']], spectra[p])
            for p in scans}
    dc_result, dc_x = fit_spectrum(dc_wl, dc_spec)
    dc = {'x': dc_x, 'spec': dc_spec, 'result': dc_result}
    #%%
    # Refine every fitted line center with a parabola fit to the local
    # spectrum minimum; all separations/differences below use these
    centers = {'disk_center': refine_centers(dc_x, dc_spec, dc_result)}
    centers.update({p: refine_centers(fits[p][1], spectra[p], fits[p][0])
                    for p in scans})
    print('\nFitted line centers [A] and offset to disk_center '
          '(mA, km/s at ~6302 A):')
    for position, c in centers.items():
        line = (f"  {position:12s} " +
                '  '.join(f"{n}={c[n]:9.4f}" for n in LINE_NAMES))
        if position != 'disk_center':
            d = {n: (c[n] - centers['disk_center'][n]) for n in LINE_NAMES}
            line += ('   |  ' +
                     '  '.join(f"{1000 * d[n]:+6.1f} mA/{d[n] / 6302 * C_KMS:+5.2f} km/s"
                              for n in LINE_NAMES))
        print(line)
    #%%
    # One figure per position (disk_center first)
    plot_fit('disk_center', dc_x, dc_spec, dc_result,
             refined=centers['disk_center'])
    for p in scans:
        plot_fit(p, fits[p][1], spectra[p], fits[p][0], dc,
                 refined=centers[p])
    #%%
    # Fit the same 4-line model to the FTS atlas (convolved with the data
    # FWHM, measured like the data) for the reference separations
    atlas_wl_nm, _, atlas_conv, atlas_cont = fts_atlas(
        disk_center['config'], dc_wl.min(), dc_wl.max())
    fts_result, fts_x = fit_spectrum(atlas_wl_nm, atlas_conv / atlas_cont)
    fts_centers = refine_centers(fts_x, atlas_conv / atlas_cont, fts_result)
    print('FTS atlas centers [A] (parabola-refined):',
          {n: round(fts_centers[n], 4) for n in LINE_NAMES})
    #%%
    # Summary: fei - telluric separation per position vs disk_center and FTS,
    # with the Sun panel. reference = 'disk_center' | 'FTS' | 'both'
    results = {'disk_center': dc_result, 'FTS': fts_result}
    results.update({p: res for p, (res, _) in fits.items()})
    plot_separation_summary(centers, fts_centers, results,
                            reference='disk_center')

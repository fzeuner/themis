#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script: compare THEMIS SR slit orientation with an HMI continuum image.

Loads the L4 SR-camera scan for 2025-07-07, sequence 24 (a sunspot, see
configs/spot_dataset_2025-07-07.toml), builds a continuum image from the
spectral line-free range, and downloads/loads the HMI continuum map closest
to 2025-07-07T10:04 UT. Both images are plotted side by side (with correct
physical scales in arcsec) so the orientation of the THEMIS slit can be
compared visually against solar N-S/E-W as seen by HMI.

THEMIS SR geometry:
    - slit width (scan step): 0.33 arcsec
    - spatial scale along the slit: 0.096 arcsec/pixel (cam1/SR pixel_scale)

Usage: run cell by cell.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

from themis.core import themis_tools as tt
from themis.core import themis_io as tio
from themis.datasets.themis_datasets_2025 import get_config

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

CONFIG_PATH = 'configs/spot_dataset_2025-07-07.toml'
LINE = 'sr'

# Continuum spectral pixel range (same as region_1 in test_alignement_in_continuum.py)
CONTINUUM_RANGE = (320, 337)  # inclusive spectral pixel indices

# Sunspot center position in helioprojective arcsec (Tx, Ty), as seen by HMI
SUNSPOT_TX = 674 * u.arcsec
SUNSPOT_TY = 319 * u.arcsec

# Rough sunspot center in the THEMIS continuum image (arcsec), measured in the
# unrotated display frame: x = scan direction (slit width steps),
# y = along the slit (camera pixels).
THEMIS_SPOT_X = 15
THEMIS_SPOT_Y = 41

# Number of 90 deg counter-clockwise rotations applied to the THEMIS image,
# plus flips of the rotated display axes -- the raster/slit directions ran
# in the opposite sense compared to the initial guess.
ROT_K = 1
FLIP_X = True
FLIP_Y = True

# HMI continuum (intensity) target time
HMI_TIME = '2025-07-07T10:04:00'
HMI_SEARCH_WINDOW = 90 * u.s  # +/- window around HMI_TIME used for the Fido search


# ----------------------------------------------------------------------------
# THEMIS continuum image
# ----------------------------------------------------------------------------

def load_sr_continuum_image(config, continuum_range=CONTINUUM_RANGE):
    """
    Load the L4 SR scan (sequence from config) and build a continuum image.

    Returns
    -------
    continuum_img : ndarray, shape (n_slit, ny)
        Continuum intensity, averaged over the spectral continuum range and
        over any map repeats. Axis 0 is the scan/raster direction (slit
        positions, step = config.slit_width arcsec), axis 1 is the spatial
        direction along the slit (step = config.cam.pixel_scale arcsec).
    header : astropy FITS header
    """
    scan, header = tio.read_any_file(config, data_type='scan', status='l4')
    result = tt.compute_polarimetry(scan)

    # result.ratio.I has shape (n_slit, n_map, ny, nx); average repeated maps
    I = result.ratio.I.mean(axis=1)  # (n_slit, ny, nx)

    lo, hi = continuum_range
    continuum_img = I[:, :, lo:hi + 1].mean(axis=2)  # (n_slit, ny)
    return continuum_img, header


# ----------------------------------------------------------------------------
# HMI continuum map
# ----------------------------------------------------------------------------

def load_hmi_continuum_map(config, time_str=HMI_TIME, search_window=HMI_SEARCH_WINDOW):
    """
    Download (or re-use a cached) HMI continuum (Ic) map closest to `time_str`.

    The file is cached under <config.directories.raw>/context so repeated
    runs of this script don't re-download the data.
    """
    context_dir = config.directories.raw / 'context'
    os.makedirs(context_dir, exist_ok=True)

    time_tag = time_str.replace(':', '').replace('-', '')
    hmi_file = context_dir / f'hmi_ic_{time_tag}.fits'

    if hmi_file.exists():
        return sunpy.map.Map(str(hmi_file))

    print(f"Downloading HMI continuum map for {time_str} UT ...")
    t_center = Time(time_str)
    t_start = t_center - search_window
    t_end = t_center + search_window
    result = Fido.search(a.Time(t_start.iso, t_end.iso),
                          a.Instrument('hmi'),
                          a.Physobs('intensity'))

    table = result[0]
    if len(table) == 0:
        raise RuntimeError(f"No HMI intensity data found near {time_str} UT")

    # Pick the record closest to the requested time (results typically come
    # at the ~45s HMI cadence, so the search window may return several).
    start_times = Time([str(t) for t in table['Start Time']])
    closest_idx = int(np.argmin(np.abs((start_times - t_center).sec)))

    downloaded_files = Fido.fetch(table[closest_idx], path=str(context_dir))
    hmi_map = sunpy.map.Map(downloaded_files[0])
    hmi_map.save(str(hmi_file), overwrite=True)
    print(f"HMI map saved to {hmi_file}")
    return hmi_map


def themis_fov_offsets(themis_disp):
    """
    Offsets (left, right, bottom, top) of the sunspot marker from the edges
    of the THEMIS display image, in arcsec. Used to cut out an HMI region
    with the *same* field of view and the same spot-to-edge distances (i.e.
    the spot sits at the same relative position in both images), rather
    than just the same total size centered on the spot.
    """
    left = themis_disp['spot_x']
    right = themis_disp['x_extent'] - themis_disp['spot_x']
    bottom = themis_disp['spot_y']
    top = themis_disp['y_extent'] - themis_disp['spot_y']
    return left, right, bottom, top


def cutout_hmi_map(hmi_map, tx=SUNSPOT_TX, ty=SUNSPOT_TY,
                    left=30 * u.arcsec, right=30 * u.arcsec,
                    bottom=30 * u.arcsec, top=30 * u.arcsec):
    """
    Crop the (rotated) HMI map to [tx-left, tx+right] x [ty-bottom, ty+top],
    i.e. the exact same field of view (and spot-relative position) as the
    THEMIS image being compared to.
    """
    hmi_rotated = hmi_map.rotate(order=3)
    center = SkyCoord(tx, ty, frame=hmi_rotated.coordinate_frame)
    bottom_left = SkyCoord(tx - left, ty - bottom, frame=hmi_rotated.coordinate_frame)
    top_right = SkyCoord(tx + right, ty + top, frame=hmi_rotated.coordinate_frame)

    print(f"HMI cutout FOV: Tx = [{(tx - left).value:.1f}\", {(tx + right).value:.1f}\"] "
          f"(width {(left + right).value:.1f}\"), "
          f"Ty = [{(ty - bottom).value:.1f}\", {(ty + top).value:.1f}\"] "
          f"(height {(bottom + top).value:.1f}\")")

    return hmi_rotated.submap(bottom_left, top_right=top_right), center


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def _rot90_point(row, col, shape, k):
    """
    Map a (row, col) point through np.rot90 applied `k` times.

    Returns (new_row, new_col, new_shape).
    """
    m, n = shape
    for _ in range(k % 4):
        row, col = n - 1 - col, row
        m, n = n, m
    return row, col, (m, n)


def build_themis_display(continuum_img, config, rot_k=ROT_K, flip_x=FLIP_X, flip_y=FLIP_Y,
                          spot_xy=(THEMIS_SPOT_X, THEMIS_SPOT_Y)):
    """
    Rotate/flip the THEMIS continuum image for display and map the sunspot
    center through the same transform.

    `spot_xy` is the estimated sunspot center in the *unrotated* display
    frame (x = scan direction, y = along slit), in arcsec.

    Returns a dict with the transformed image, its physical extent, axis
    labels/steps, and the transformed spot position (all in arcsec).
    """
    step_scan = config.slit_width       # arcsec per scan step
    step_slit = 0.09#config.cam.pixel_scale  # arcsec per pixel along the slit

    # Display frame: rows = along slit, cols = scan direction
    display_img = continuum_img.T
    rotated_img = np.rot90(display_img, k=rot_k)

    # Physical step per display axis after rotation
    if rot_k % 2 == 0:
        dx, dy = step_scan, step_slit
        xlabel, ylabel = 'scan direction', 'along slit'
    else:
        dx, dy = step_slit, step_scan
        xlabel, ylabel = 'along slit', 'scan direction'

    x_extent = rotated_img.shape[1] * dx
    y_extent = rotated_img.shape[0] * dy

    # Map the spot center (arcsec -> display pixels -> rotated display pixels)
    spot_col = spot_xy[0] / step_scan   # x = scan direction
    spot_row = spot_xy[1] / step_slit   # y = along slit
    r_row, r_col, _ = _rot90_point(spot_row, spot_col, display_img.shape, rot_k)
    spot_x_rot, spot_y_rot = r_col * dx, r_row * dy

    if flip_x:
        rotated_img = np.fliplr(rotated_img)
        spot_x_rot = x_extent - spot_x_rot
    if flip_y:
        rotated_img = np.flipud(rotated_img)
        spot_y_rot = y_extent - spot_y_rot

    return {
        'image': rotated_img,
        'x_extent': x_extent, 'y_extent': y_extent,
        'dx': dx, 'dy': dy,
        'xlabel': xlabel, 'ylabel': ylabel,
        'spot_x': spot_x_rot, 'spot_y': spot_y_rot,
    }


def add_compass(ax, hmi_map, anchor, length=10 * u.arcsec,
                 color='yellow'):
    """
    Draw small N and E arrows on an HMI (WCS) axis, anchored at `anchor`
    (a SkyCoord), to indicate solar-north/east orientation. The anchor is
    chosen for a clear, uncluttered placement -- it does not mark a
    physical location of interest.
    """
    frame = hmi_map.coordinate_frame

    def _to_pixel_xy(coord):
        px, py = hmi_map.world_to_pixel(coord)
        return np.array([px.to_value(u.pix), py.to_value(u.pix)])

    anchor_px = _to_pixel_xy(anchor)

    for d_tx, d_ty, label, dx_text, dy_text in (
        (0 * u.arcsec, length, 'N', 0, 1),
        (length, 0 * u.arcsec, 'E', 1, 0),
    ):
        tip = SkyCoord(anchor.Tx + d_tx, anchor.Ty + d_ty, frame=frame)
        tip_px = _to_pixel_xy(tip)
        ax.annotate('', xy=tuple(tip_px), xytext=tuple(anchor_px),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        label_px = tip_px + (tip_px - anchor_px) * 0.25
        ax.text(label_px[0], label_px[1], label, color=color,
                fontsize=11, fontweight='bold', ha='center', va='center')


def plot_comparison(themis_disp, config, hmi_cutout, center):
    """
    Plot the HMI cutout and the THEMIS SR continuum image stacked on top of
    each other (both images are wider than they are tall, so this uses the
    figure area more efficiently than placing them side by side).
    """
    fig = plt.figure(figsize=(9, 8))

    ax_hmi = fig.add_subplot(2, 1, 1, projection=hmi_cutout)
    med = np.nanmedian(hmi_cutout.data)
    im_hmi = hmi_cutout.plot(axes=ax_hmi, cmap='gray', vmin=0.45 * med, vmax=1.15 * med)
    ax_hmi.plot_coord(center, marker='+', color='red', markersize=14, markeredgewidth=2)

    # Compass anchored near the bottom-left corner of the cutout, away from the spot
    bottom_left = SkyCoord(hmi_cutout.bottom_left_coord.Tx + 8 * u.arcsec,
                           hmi_cutout.bottom_left_coord.Ty + 8 * u.arcsec,
                           frame=hmi_cutout.coordinate_frame)
    add_compass(ax_hmi, hmi_cutout, bottom_left)

    ax_hmi.set_title(f'HMI continuum, {hmi_cutout.date.iso} UT\n'
                     f'spot: ({center.Tx.value:.0f}", {center.Ty.value:.0f}")',
                     fontsize=11)
    fig.colorbar(im_hmi, ax=ax_hmi, shrink=0.85, label='DN')

    flip_tag = ''.join(f for f, on in (('x', FLIP_X), ('y', FLIP_Y)) if on)
    xform_str = f'rot {ROT_K * 90}°' + (f', flip {flip_tag}' if flip_tag else '')
    ax_sr = fig.add_subplot(2, 1, 2)
    im = ax_sr.imshow(themis_disp['image'], origin='lower', cmap='gray',
                       extent=[0, themis_disp['x_extent'], 0, themis_disp['y_extent']],
                       aspect='equal')
    ax_sr.plot(themis_disp['spot_x'], themis_disp['spot_y'], 'r+',
               markersize=14, markeredgewidth=2)
    ax_sr.set_xlabel(f"{themis_disp['xlabel']} [arcsec]  ({themis_disp['dx']}\"/step)")
    ax_sr.set_ylabel(f"{themis_disp['ylabel']} [arcsec]  ({themis_disp['dy']}\"/step)")
    ax_sr.set_title(f"THEMIS {LINE.upper()} seq={config.dataset['scan']['sequence']} ({xform_str})\n"
                     f"spot: ({themis_disp['spot_x']:.1f}\", {themis_disp['spot_y']:.1f}\")",
                     fontsize=11)
    fig.colorbar(im, ax=ax_sr, shrink=0.85, label='continuum intensity')

    fig.subplots_adjust(hspace=0.3, top=0.93, bottom=0.07)
    return fig


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    #%%
    config = get_config(line=LINE, config_path=CONFIG_PATH,
                         auto_discover_files=True, auto_create_dirs=False)

    continuum_img, header = load_sr_continuum_image(config)
    print(f"THEMIS continuum image shape (n_slit, ny): {continuum_img.shape}")

    themis_disp = build_themis_display(continuum_img, config)

    #%%
    # Cut out HMI with the same FOV as the THEMIS image, and with the spot
    # at the same relative position (distance to each edge) in both images.
    left, right, bottom, top = themis_fov_offsets(themis_disp)
    hmi_map = load_hmi_continuum_map(config)
    hmi_cutout, center = cutout_hmi_map(
        hmi_map,
        left=left * u.arcsec, right=right * u.arcsec,
        bottom=bottom * u.arcsec, top=top * u.arcsec,
    )

    #%%
    fig = plot_comparison(themis_disp, config, hmi_cutout, center)

    figures_dir = config.directories.figures / 'slit_orientation_vs_hmi'
    figures_dir.mkdir(exist_ok=True, parents=True)
    out_path = figures_dir / f'sr_seq{config.dataset["scan"]["sequence"]}_vs_hmi.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {out_path}")

    plt.show()

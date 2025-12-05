#!/usr/bin/env python3
"""
Test script to calculate offset between upper and lower frame halves.

This script applies all flat-field corrections and then computes the spatial/spectral
shift between upper and lower frames.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.ndimage import shift as scipy_shift
from scipy.signal import correlate2d

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
# Add atlas-fit to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'atlas-fit' / 'src'))

from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio
from spectroflat.smile import SmileInterpolator, OffsetMap
from spectroflat.utils.processing import MP

from spectator.controllers.app_controller import display_data # from spectator

import imreg_dft as ird

# Atlas-fit imports
from atlas_fit.atlantes import atlas_factory
from atlas_fit.base.constants import NM2ANGSTROM

#%%
def apply_desmiling_spectroflat(image, offset_map_array, roi):
    """
    Apply desmiling row-by-row using spectroflat's SmileInterpolator.desmile_row.
    Uses multiprocessing for speed. Only applies desmiling within the ROI region.
    
    Parameters
    ----------
    image : np.ndarray
        2D image [spatial, wavelength] - full frame
    offset_map_array : np.ndarray
        2D offset map array [spatial, wavelength] for this specific state.
    roi : tuple of slices
        Region of interest from offset_map.header['ROI'], e.g.
        (slice(20, 858, None), slice(20, 1230, None))
    
    Returns
    -------
    np.ndarray
        Desmiled image [spatial, wavelength] - full frame with ROI region desmiled
    """
    # Extract ROI from image and offset map
    image_roi = image[roi]
    offset_roi = offset_map_array[roi]
    ny_roi, nx_roi = image_roi.shape
    
    rows = np.arange(ny_roi)
    xes = np.arange(nx_roi)
    
    # Apply desmiling row by row with multiprocessing (only within ROI)
    # Arguments: (row_index, x_coordinates, offsets_for_this_row, data_for_this_row)
    arguments = [(r, xes, offset_roi[r], image_roi[r]) for r in rows]
    res = dict(MP.simultaneous(SmileInterpolator.desmile_row, arguments))
    
    # Reconstruct the desmiled ROI from row results
    desmiled_roi = np.array([res[row] for row in rows])
    
    # Create output array (copy of input) and insert desmiled ROI
    desmiled = image.copy()
    desmiled[roi] = desmiled_roi
    
    return desmiled


def apply_continuum_correction(upper_data, lower_data, wl_upper, wl_lower, dataset_config_path, 
                                atlas_fit_config_path='/home/franziskaz/code/themis/configs/atlas_fit_config_cam1.yml',
                                spline_smoothing=0):
    """
    Apply continuum correction to bring lower frame continuum to match upper frame.
    
    1. Apply stray light correction to both frames
    2. Fit spline through continuum wavelength points for each row in both frames
    3. Compute ratio: upper_spline / lower_spline
    4. Multiply lower data by this ratio to match upper continuum
    
    Parameters
    ----------
    upper_data : ndarray
        2D array (spatial, spectral) for upper frame
    lower_data : ndarray
        2D array (spatial, spectral) for lower frame (aligned to upper)
    wl_upper : ndarray
        1D wavelength array for upper frame [nm]
    wl_lower : ndarray
        1D wavelength array for lower frame [nm]
    dataset_config_path : str
        Path to dataset config TOML file containing [continuum] section
    atlas_fit_config_path : str
        Path to atlas_fit config YAML file for stray light values
    spline_smoothing : float
        Smoothing factor for spline (0 = interpolating spline through points)
    
    Returns
    -------
    upper_corrected : ndarray
        Upper frame with stray light correction applied
    lower_corrected : ndarray
        Lower frame with stray light + continuum matching applied
    spline_ratio : ndarray
        The 2D ratio array (upper_spline / lower_spline) applied to lower
    """
    from scipy.interpolate import UnivariateSpline
    import tomllib
    import yaml
    
    # Load dataset config for continuum wavelengths
    with open(dataset_config_path, 'rb') as f:
        config = tomllib.load(f)
    
    continuum_cfg = config.get('continuum', {})
    continuum_wavelengths = continuum_cfg.get('continuum_wavelengths', [460.55, 460.70, 460.85, 461.00])
    spline_degree = continuum_cfg.get('spline_degree', 3)
    
    # Load atlas_fit config for stray light
    with open(atlas_fit_config_path, 'r') as f:
        atlas_config = yaml.safe_load(f)
    
    input_cfg = atlas_config.get('input', {})
    stray_light_upper = 3#input_cfg.get('stray_light', 2.96)
    stray_light_lower = 3.5#input_cfg.get('stray_light_lower', 4.38)
    print('Stray light values are hardcoded!')
    ny, nx = upper_data.shape
    
    # Apply stray light correction to both frames
    #upper_sl_corrected = upper_data - (np.mean(upper_data) * stray_light_upper / 100)
    #lower_sl_corrected = lower_data - (np.mean(lower_data) * stray_light_lower / 100)
    
    # Micheles stray light correction
    upper_sl_corrected = (1.+stray_light_upper / 100) * upper_data/np.mean(upper_data) - (np.mean(upper_data) * stray_light_upper / 100)
    lower_sl_corrected = (1.+stray_light_lower / 100) * lower_data/np.mean(lower_data) - (np.mean(lower_data) * stray_light_lower / 100)

    lower_sl_corrected = lower_sl_corrected-lower_sl_corrected.mean()+upper_sl_corrected.mean()
    
    # Convert wavelength points to pixel indices for both frames
    def get_pixel_indices(wl_array, wavelengths):
        indices = []
        for wl in wavelengths:
            idx = np.argmin(np.abs(wl_array - wl))
            indices.append(idx)
        return np.array(indices)
    
    cont_idx_upper = get_pixel_indices(wl_upper, continuum_wavelengths)
    cont_idx_lower = get_pixel_indices(wl_lower, continuum_wavelengths)
    
    # Fit spline for each row in both frames
    spline_upper = np.zeros_like(upper_sl_corrected)
    spline_lower = np.zeros_like(lower_sl_corrected)
    xes = np.arange(nx)
    
    for row in range(ny):
        # Upper frame spline
        cont_values_upper = upper_sl_corrected[row, cont_idx_upper]
        if len(cont_idx_upper) > spline_degree:
            spl_upper = UnivariateSpline(cont_idx_upper, cont_values_upper, 
                                          k=spline_degree, s=spline_smoothing)
            spline_upper[row, :] = spl_upper(xes)
        else:
            spline_upper[row, :] = np.interp(xes, cont_idx_upper, cont_values_upper)
        
        # Lower frame spline
        cont_values_lower = lower_sl_corrected[row, cont_idx_lower]
        if len(cont_idx_lower) > spline_degree:
            spl_lower = UnivariateSpline(cont_idx_lower, cont_values_lower, 
                                          k=spline_degree, s=spline_smoothing)
            spline_lower[row, :] = spl_lower(xes)
        else:
            spline_lower[row, :] = np.interp(xes, cont_idx_lower, cont_values_lower)
    
    # Compute ratio: upper_spline / lower_spline
    # This will scale lower continuum to match upper
    spline_ratio = spline_upper / spline_lower
    
    # Apply ratio to lower frame
    lower_corrected = lower_sl_corrected * spline_ratio
    
    return upper_sl_corrected, lower_corrected, spline_ratio


def plot_spectra_vs_atlas(corrected_upper, corrected_lower, wl_upper, wl_lower, config, 
                          row_indices=None, atlas_key='hhdc', fwhm=0.00332, 
                          stray_light_upper=2.96, stray_light_lower=4.38):
    """
    Plot spectra from multiple rows compared to the solar atlas.
    
    The data and atlas are treated the same way as in atlas-fit:
    - Atlas: Convolved with differential FWHM
    - Data: Stray light SUBTRACTED, then normalized
    
    Parameters
    ----------
    corrected_upper : ndarray
        Corrected upper frame (spatial, spectral)
    corrected_lower : ndarray
        Corrected lower frame (spatial, spectral)
    wl_upper : ndarray
        Wavelength array for upper frame (from offset map)
    wl_lower : ndarray
        Wavelength array for lower frame (from offset map)
    config : Config
        Configuration object
    row_indices : list, optional
        List of row indices to plot. Default: [low, middle, high]
    atlas_key : str
        Atlas key for atlas_factory (default: 'hhdc')
    fwhm : float
        FWHM of the data in nm (for differential convolution)
    stray_light_upper : float
        Stray light level in percent for upper frame
    stray_light_lower : float
        Stray light level in percent for lower frame
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d
    from scipy.signal import resample
    
    ny, nx = corrected_upper.shape
    
    # Default row indices: low, middle, high
    if row_indices is None:
        row_indices = [ny // 6, ny // 2, 5 * ny // 6]
    
    # Determine wavelength range from offset maps
    wl_start = min(wl_upper.min(), wl_lower.min())
    wl_end = max(wl_upper.max(), wl_lower.max())
    
    # Load the atlas
    print(f"\n4. Loading atlas ({atlas_key}) for wavelength range {wl_start:.4f} - {wl_end:.4f} nm...")
    atlas = atlas_factory(atlas_key, wl_start, wl_end, conversion=NM2ANGSTROM)
    print(f"   ✓ Atlas loaded: {len(atlas.wl)} points")
    
    # Treat atlas like atlas-fit does:
    # 1. Convolve with differential FWHM (data FWHM - atlas FWHM)
    atlas_fwhm = getattr(atlas, 'fwhm', 0.001)  # Atlas FWHM, typically ~1 mÅ
    delta_fwhm = np.sqrt(fwhm**2 - atlas_fwhm**2) if fwhm > atlas_fwhm else 0
    
    if delta_fwhm > 0:
        # Convert FWHM to sigma for Gaussian filter
        # FWHM = 2.355 * sigma, and we need sigma in pixels
        dispersion = (atlas.wl[-1] - atlas.wl[0]) / len(atlas.wl)  # nm/pixel
        sigma_pixels = delta_fwhm / (2.355 * dispersion)
        atlas_convolved = gaussian_filter1d(atlas.intensity, sigma_pixels)
        print(f"   ✓ Atlas convolved with FWHM={delta_fwhm:.2e} nm (sigma={sigma_pixels:.1f} px)")
    else:
        atlas_convolved = atlas.intensity.copy()
    
    print(f"   ✓ Stray light will be SUBTRACTED from data: upper={stray_light_upper:.2f}%, lower={stray_light_lower:.2f}%")
    
    # Create figure with 2x2 subplots (left: spectra, right: differences)
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig.suptitle(f'Spectra vs Atlas Comparison\nWavelength: {wl_start:.4f} - {wl_end:.4f} nm (FWHM={fwhm*1000:.2f} pm)', 
                 fontsize=14, fontweight='bold')
    
    colors = ['tab:blue', 'tab:green', 'tab:red']
    row_labels = ['Low', 'Middle', 'High']
    
    # Store normalized spectra for difference plots
    spectra_normalized = {'Upper': {}, 'Lower': {}}
    
    for ax_idx, (ax, frame_data, frame_name, wl_data, stray_light) in enumerate([
        (axes[0, 0], corrected_upper, 'Upper', wl_upper, stray_light_upper),
        (axes[1, 0], corrected_lower, 'Lower', wl_lower, stray_light_lower)
    ]):
        ax.set_title(f'{frame_name} Frame (stray light: {stray_light:.2f}%)')
        
        # Resample atlas to data wavelength grid
        atlas_interp = interp1d(atlas.wl, atlas_convolved, kind='cubic', fill_value='extrapolate')
        atlas_resampled = atlas_interp(wl_data)
        
        # Normalize atlas to mean of atlas intensity (like atlas-fit does)
        atlas_norm = atlas_resampled / np.mean(atlas_resampled)
        ax.plot(wl_data, atlas_norm, 'k-', linewidth=1.5, alpha=0.7, label=f'Atlas ({atlas_key.upper()})')
        
        # Plot spectra from selected rows
        for i, (row_idx, color, label) in enumerate(zip(row_indices, colors, row_labels)):
            spectrum_raw = frame_data[row_idx, :].copy()
            
            # Store raw (no stray light correction) normalized spectrum
            spectrum_raw_norm = spectrum_raw / np.mean(spectrum_raw)
            spectra_normalized[frame_name][label + '_raw'] = spectrum_raw_norm
            
            # Apply stray light correction to DATA (subtract, like atlas-fit does)
            # data = data - (data.mean() * stray_light / 100)
            spectrum = spectrum_raw - (spectrum_raw.mean() * stray_light / 100)
            
            # Normalize spectrum to same scale as atlas
            spectrum_norm = (spectrum / spectrum.mean()) * atlas_resampled.mean()
            spectrum_norm = spectrum_norm / np.mean(spectrum_norm)
            
            # Store for difference plot
            spectra_normalized[frame_name][label] = spectrum_norm
            
            ax.plot(wl_data, spectrum_norm, color=color, linewidth=0.8, 
                    alpha=0.8, label=f'Row {row_idx} ({label})')
        
        ax.set_ylabel('Normalized Intensity')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(wl_data.min(), wl_data.max())
    
    axes[1, 0].set_xlabel('Wavelength [nm]')
    
    # Plot differences in percent (right column)
    for ax_idx, (ax, frame_name, wl_data) in enumerate([
        (axes[0, 1], 'Upper', wl_upper),
        (axes[1, 1], 'Lower', wl_lower)
    ]):
        spectra = spectra_normalized[frame_name]
        
        # Differences WITH stray light correction (solid lines)
        diff_mid_low = (spectra['Middle'] - spectra['Low']) / spectra['Middle'] * 100
        ax.plot(wl_data, diff_mid_low, color='tab:purple', linewidth=1.0, 
                label='Mid-Low (stray corr)')
        
        diff_mid_high = (spectra['Middle'] - spectra['High']) / spectra['Middle'] * 100
        ax.plot(wl_data, diff_mid_high, color='tab:orange', linewidth=1.0, 
                label='Mid-High (stray corr)')
        
        # Differences WITHOUT stray light correction (dashed lines)
        diff_mid_low_raw = (spectra['Middle_raw'] - spectra['Low_raw']) / spectra['Middle_raw'] * 100
        ax.plot(wl_data, diff_mid_low_raw, color='tab:purple', linewidth=1.0, linestyle='--',
                alpha=0.6, label='Mid-Low (no corr)')
        
        diff_mid_high_raw = (spectra['Middle_raw'] - spectra['High_raw']) / spectra['Middle_raw'] * 100
        ax.plot(wl_data, diff_mid_high_raw, color='tab:orange', linewidth=1.0, linestyle='--',
                alpha=0.6, label='Mid-High (no corr)')
        
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_title(f'{frame_name} Frame - Spectral Differences')
        ax.set_ylabel('Difference [%]')
        ax.set_ylim(-1, 1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(wl_data.min(), wl_data.max())
    
    axes[1, 1].set_xlabel('Wavelength [nm]')
    
    plt.tight_layout()
    
    # Save figure
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures_dir = Path(config.directories.figures)
    figures_dir.mkdir(exist_ok=True, parents=True)
    output_plot = figures_dir / f'spectra_vs_atlas_{timestamp}.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved spectra vs atlas plot: {output_plot}")
    
    plt.show()
    
    return fig


def plot_alignment_diff(upper_data, lower_data_aligned, corrected_upper, corrected_lower_aligned, 
                        tvec, angle, scale, config, margin=50):
    """
    Plot differences between upper and aligned lower frames.
    
    Parameters
    ----------
    upper_data : ndarray
        L0 upper frame
    lower_data_aligned : ndarray
        L0 lower frame after alignment
    corrected_upper : ndarray
        Corrected upper frame
    corrected_lower_aligned : ndarray
        Corrected lower frame after alignment
    tvec : tuple
        Translation vector (y, x)
    angle : float
        Rotation angle in degrees
    scale : float
        Scale factor
    config : Config
        Configuration object
    margin : int
        Margin for percentile calculation
    
    Returns
    -------
    fig : Figure
        Matplotlib figure
    """
    from datetime import datetime
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Upper vs Lower Frame Alignment\nTranslation: ({tvec[0]:.2f}, {tvec[1]:.2f}) px, Rotation: {angle:.3f}°, Scale: {scale:.5f}', 
                 fontsize=14, fontweight='bold')
    
    # Left: L0 difference (upper - aligned lower)
    diff_l0 = upper_data - lower_data_aligned
    if margin > 0:
        vmax_l0 = np.percentile(np.abs(diff_l0[margin:-margin, margin:-margin]), 99)
    else:
        vmax_l0 = np.percentile(np.abs(diff_l0), 99)
    
    ax = axes[0]
    im = ax.imshow(diff_l0, aspect='auto', cmap='RdBu_r', vmin=-vmax_l0, vmax=vmax_l0)
    ax.set_title('L0: Upper - Aligned Lower')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Difference')
    
    # Right: Corrected difference (corrected_upper - aligned corrected_lower)
    diff_corrected = corrected_upper - corrected_lower_aligned
    if margin > 0:
        vmax_corr = np.percentile(np.abs(diff_corrected[margin:-margin, margin:-margin]), 99)
    else:
        vmax_corr = np.percentile(np.abs(diff_corrected), 99)
    
    ax = axes[1]
    im = ax.imshow(diff_corrected, aspect='auto', cmap='RdBu_r', vmin=-vmax_corr, vmax=vmax_corr)
    ax.set_title('Corrected: Upper - Aligned Lower')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Difference')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures_dir = Path(config.directories.figures)
    figures_dir.mkdir(exist_ok=True, parents=True)
    output_plot = figures_dir / f'frame_alignment_diff_{timestamp}.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved alignment difference plot: {output_plot}")
    
    plt.show()
    
    return fig

   
#%%

if __name__ == "__main__":
    """Main test function."""
    print("="*70)
    print("Frame Alignment Test: Upper vs Lower")
    print("="*70)
    
    # Load configuration
    config = get_config(
        config_path='configs/sample_dataset_sr_2025-07-07.toml',
        auto_discover_files=True
    )
    
    data_type = 'flat_center'
    
    # Step 0: Read L0 flat
    print("\n0. Loading L0 flat data...")
    l0_data, l0_header = tio.read_any_file(config, data_type, verbose=False, status='l0')
    
    
    l0_scan, l0_header_scan = tio.read_any_file(config, 'scan', verbose=False, status='l0')
    
    upper_data = l0_data[0]['upper'].data
    lower_data = l0_data[0]['lower'].data
    

    
   
    # Load dust flats from auxiliary files
    
    with fits.open(config.dataset[data_type]['files'].auxiliary.get('dust_flat_upper')) as hdu:
        dust_flat_upper = hdu[0].data
    with fits.open(config.dataset[data_type]['files'].auxiliary.get('dust_flat_lower')) as hdu:
        dust_flat_lower = hdu[0].data
    print(f"   ✓ Loaded dust flats from auxiliary files")
    
    upper_scan = l0_scan['pQ',0,0]['upper'].data/dust_flat_upper
    lower_scan = l0_scan['pQ',0,0]['lower'].data/dust_flat_lower
    
    # Load offset maps (both outdated and amended)
    offset_map_upper_outdated_file = config.dataset[data_type]['files'].auxiliary.get('offset_map_upper_outdated')
    offset_map_lower_outdated_file = config.dataset[data_type]['files'].auxiliary.get('offset_map_lower_outdated')
    offset_map_upper_amended_file = config.dataset[data_type]['files'].auxiliary.get('offset_map_upper')
    offset_map_lower_amended_file = config.dataset[data_type]['files'].auxiliary.get('offset_map_lower')
    
    # Load offset maps and extract arrays, then close files
    print(f"   Loading offset maps...")
    offset_map_upper_outdated = OffsetMap.from_file(offset_map_upper_outdated_file)
    offset_map_lower_outdated = OffsetMap.from_file(offset_map_lower_outdated_file)
    # offset_map_upper_amended = OffsetMap.from_file(offset_map_upper_amended_file)
    # offset_map_lower_amended = OffsetMap.from_file(offset_map_lower_amended_file)
    
    # Extract arrays (state 0), ROI and wavelength from header before closing
    offset_upper_outdated_2d = offset_map_upper_outdated.map[0].copy()
    offset_lower_outdated_2d = offset_map_lower_outdated.map[0].copy()
    roi_upper = eval(offset_map_upper_outdated.header['ROI'])  # e.g. (slice(20, 858, None), slice(20, 1230, None))
    roi_lower = eval(offset_map_lower_outdated.header['ROI'])
    # Extract wavelength calibration from offset maps
    wl_upper = offset_map_upper_outdated.wl.copy()
    wl_lower = offset_map_lower_outdated.wl.copy()
    # offset_upper_amended_2d = offset_map_upper_amended.map[0].copy()
    # offset_lower_amended_2d = offset_map_lower_amended.map[0].copy()
    
    print(f"   ROI upper: {roi_upper}")
    print(f"   ROI lower: {roi_lower}")
    print(f"   WL upper: {wl_upper.min():.4f} - {wl_upper.max():.4f} nm ({len(wl_upper)} points)")
    print(f"   WL lower: {wl_lower.min():.4f} - {wl_lower.max():.4f} nm ({len(wl_lower)} points)")
    
    # Explicitly delete OffsetMap objects to close file handles
    del offset_map_upper_outdated, offset_map_lower_outdated
    # del offset_map_upper_amended, offset_map_lower_amended
    
    print(f"   ✓ Loaded offset maps (outdated and amended) and closed files")
    
    # Load illumination patterns
    with fits.open(config.dataset[data_type]['files'].auxiliary.get('illumination_pattern_upper')) as hdu:
        illum_upper = hdu[0].data
    with fits.open(config.dataset[data_type]['files'].auxiliary.get('illumination_pattern_lower')) as hdu:
        illum_lower = hdu[0].data
    print(f"   ✓ Loaded illumination patterns")
    
    # Step 1: Apply dust flats
    print("\n1. Applying dust flats...")
    upper_dust_corrected = upper_data / dust_flat_upper
    lower_dust_corrected = lower_data / dust_flat_lower
    print(f"   ✓ Upper dust corrected: {upper_dust_corrected.shape}")
    print(f"   ✓ Lower dust corrected: {lower_dust_corrected.shape}")
    
    # Step 2: De-smile (apply offset maps in two stages: outdated + delta)
    print("\n2. De-smiling frames with offset maps (two-stage process)...")
    
    # Calculate delta offsets (like atlas-fit does)
    # delta_upper = offset_upper_amended_2d - offset_upper_outdated_2d
    # delta_lower = offset_lower_amended_2d - offset_lower_outdated_2d
    # print(f"\n   Computed delta offsets (amended - outdated)")
    # print(f"     Upper delta: mean={np.mean(delta_upper):.4f}, std={np.std(delta_upper):.4f}, max={np.max(np.abs(delta_upper)):.4f} px")
    # print(f"     Lower delta: mean={np.mean(delta_lower):.4f}, std={np.std(delta_lower):.4f}, max={np.max(np.abs(delta_lower)):.4f} px")
    
    # Step 2a: Apply OUTDATED offset map first
    print(f"\n   Step 2a: Desmile with OUTDATED offset map...")
    print(f"     Processing ROI rows (multiprocessing)...")
    upper_desmiled = apply_desmiling_spectroflat(upper_dust_corrected, offset_upper_outdated_2d, roi_upper)
    print(f"     ✓ Upper desmiled")
    lower_desmiled = apply_desmiling_spectroflat(lower_dust_corrected, offset_lower_outdated_2d, roi_lower)
    print(f"     ✓ Lower desmiled")
    
    upper_scan_desmiled = apply_desmiling_spectroflat(upper_scan, offset_upper_outdated_2d, roi_upper)
    lower_scan_desmiled = apply_desmiling_spectroflat(lower_scan, offset_upper_outdated_2d, roi_upper)
    
    
    # print(f"     ✓ Upper desmiled (outdated): {upper_desmiled_outdated.shape}")
    # print(f"     ✓ Lower desmiled (outdated): {lower_desmiled_outdated.shape}")
    
    # # Step 2b: Apply DELTA corrections to already-desmiled data
    # print(f"\n   Step 2b: Apply small DELTA corrections to desmiled data...")
    # upper_desmiled = apply_desmiling_spectroflat(upper_desmiled_outdated, delta_upper)
    # lower_desmiled = apply_desmiling_spectroflat(lower_desmiled_outdated, delta_lower)
    # print(f"     ✓ Upper fully desmiled (outdated + delta): {upper_desmiled.shape}")
    # print(f"     ✓ Lower fully desmiled (outdated + delta): {lower_desmiled.shape}")
    
    # Step 3: Apply illumination patterns
    print("\n3. Applying illumination patterns...")
    corrected_upper = upper_desmiled / illum_upper[0]  # state 0
    corrected_lower = lower_desmiled / illum_lower[0]  # state 0
    
    print(f"   ✓ Upper fully corrected: {corrected_upper.shape}")
    print(f"   ✓ Lower fully corrected: {corrected_lower.shape}")

    # Step 4: Compare spectra with atlas
    # Use wavelength calibration from offset maps
    # FWHM and stray_light from atlas-fit config (configs/atlas_fit_config_cam1.yml)
#     fwhm = 0.00332  # nm
#     stray_light_upper = 2.96  # %
#     stray_light_lower = 4.38  # %
# #%%    
#     pl = plot_spectra_vs_atlas(
#         corrected_upper, 
#         corrected_lower, 
#         wl_upper,
#         wl_lower,
#         config,
#         row_indices=None,  # Will use default: [low, middle, high]
#         atlas_key='hhdc',  # Hamburg Disk-Center atlas
#         fwhm=fwhm,
#         stray_light_upper=stray_light_upper,
#         stray_light_lower=stray_light_lower
#     )
#%%
    # Step 5: Calculate alignment using imreg_dft on L0 data
    print("\n5. Calculating alignment transformation using L0 data...")
    
    # Normalize images to maximum before alignment
    # Crop edges to avoid border effects
    margin = 50
    upper_data_norm = (upper_data / np.mean(upper_data))[margin:-margin, margin:-margin]
    lower_data_norm = (lower_data / np.mean(lower_data))[margin:-margin, margin:-margin]
    
    corrected_upper_norm = (corrected_upper / np.mean(corrected_upper)) [margin:-margin, margin:-margin]
    corrected_lower_norm = (corrected_lower / np.mean(corrected_lower)) [margin:-margin, margin:-margin]

    upper_scan_norm = (upper_scan_desmiled/np.mean(upper_scan_desmiled))[margin:-margin, margin:-margin]
    lower_scan_norm = (lower_scan_desmiled/np.mean(lower_scan_desmiled))[margin:-margin, margin:-margin]
    
    upper_scan_nondesmiled_norm = (upper_scan/np.mean(upper_scan))[margin:-margin, margin:-margin]
    lower_scan_nondesmiled_norm = (lower_scan/np.mean(lower_scan))[margin:-margin, margin:-margin]
    
    # Step 5a: Determine initial large shift in spectral direction (axis 1)
    # by cross-correlating averaged spectra over central 400 spatial pixels
    print("   Finding initial spectral shift from averaged spectra...")
    ny, nx = upper_scan_norm.shape
    spatial_center = ny // 2
    spatial_half_width = 200  # 400 pixels total
    
    # Average over central spatial pixels to get 1D spectra
    upper_spectrum_avg = np.mean(upper_scan_norm[spatial_center-spatial_half_width:spatial_center+spatial_half_width, 250:-450], axis=0)
    lower_spectrum_avg = np.mean(lower_scan_norm[spatial_center-spatial_half_width:spatial_center+spatial_half_width,250 :-450], axis=0)
    
    # Cross-correlate to find spectral shift - does work very well
    #correlation = np.correlate(upper_spectrum_avg, lower_spectrum_avg, mode='full')
    #lag = np.arange(-len(lower_spectrum_avg) + 1, len(upper_spectrum_avg))
    # hardcoded!!!
    initial_shift_x = -8#lag[np.argmax(correlation)] #-8#
    print(f"   ✓ Initial spectral shift (x): {initial_shift_x} px - HARDCODED")
    
    # Pre-shift lower images by initial spectral shift
    from scipy.ndimage import shift as scipy_shift
    lower_scan_norm_preshifted = scipy_shift(lower_scan_norm, (0, initial_shift_x), order=3, mode='constant')
    
    lower_data_norm_preshifted = scipy_shift(lower_data_norm, (0, initial_shift_x), order=3, mode='constant')
    corrected_lower_norm_preshifted = scipy_shift(corrected_lower_norm, (0, initial_shift_x), order=3, mode='constant')
    
    # Use imreg_dft to find residual transformation (on pre-shifted data)
    print("   Finding residual transformation with imreg_dft...")
    result = ird.similarity(
        upper_scan_nondesmiled_norm, 
        lower_scan_nondesmiled_norm, # preshifted for better results
        numiter=5,
        constraints={
        'tx': (-1,1),      # x translation bounds
        'ty': (-2, 2),      # y translation bounds  
        'angle': (-0.0001,0.0001),     # rotation bounds in degrees
        'scale': (0.99999, 1.00001),  # scale bounds
        }
    )
    
    # Extract transformation parameters and add initial shift
    tvec = result['tvec']  # Translation vector (y, x)
    angle = result['angle']  # Rotation angle
    scale = result['scale']  # Scale factor
    
    # Total shift = initial + residual
    total_tvec = (tvec[0]+2.7, tvec[1] + initial_shift_x)
    print(f"   ✓ Residual shift: ({tvec[0]:.3f}, {tvec[1]:.3f}) px")
    print(f"   ✓ Total transformation:")
    print(f"     Translation: ({total_tvec[0]:.3f}, {total_tvec[1]:.3f}) px (y, x)")
    print(f"     Rotation: {angle:.4f}°")
    print(f"     Scale: {scale:.6f}")
    
    # Apply full transformation (total_tvec) to original normalized data
    lower_data_norm_aligned = ird.transform_img(lower_data_norm, tvec=total_tvec, angle=angle, scale=scale)
    
    # Apply the SAME transformation to normalized corrected_lower
    corrected_lower_norm_aligned = ird.transform_img(corrected_lower_norm, tvec=total_tvec, angle=angle, scale=scale)
    
    # Apply the SAME transformation to normalized corrected_lower_scan
    corrected_lower_norm_scan_aligned = ird.transform_img(lower_scan_norm, tvec=total_tvec, angle=angle, scale=scale)
    
    
    total_tvec = (1.4,  -8)
    lower_scan_nondesmiled_aligned = ird.transform_img(lower_scan_nondesmiled_norm,tvec=total_tvec, angle=0, scale=1)
    print(f"   ✓ Applied transformation to both L0 and corrected lower frames")
  #%%  
    # Step 6: Continuum correction
    print("\n6. Applying continuum correction...")
    
    dataset_config_path = '/home/franziskaz/code/themis/configs/sample_dataset_sr_2025-07-07.toml'
    
    # Apply continuum correction to bring lower continuum to match upper
    upper_cont_corrected, lower_cont_corrected, spline_ratio = apply_continuum_correction(
        corrected_upper_norm, corrected_lower_norm_aligned, 
        wl_upper[margin:-margin], wl_lower[margin:-margin], 
        dataset_config_path
    )
    
    print(f"   ✓ Spline ratio (upper/lower): mean={np.mean(spline_ratio):.4f}, std={np.std(spline_ratio):.4f}")
    print(f"   ✓ Upper after stray light: mean={np.mean(upper_cont_corrected):.4f}")
    print(f"   ✓ Lower after continuum match: mean={np.mean(lower_cont_corrected):.4f}")
    
    t = plot_alignment_diff(
        corrected_upper_norm, corrected_lower_norm_aligned,
        upper_cont_corrected, lower_cont_corrected,
        total_tvec, angle, scale, config, margin=0
    )
 #%%   
    # Step 7: Plot differences
    print("\n7. Plotting alignment results...")
    plot_alignment_diff(
        upper_data_norm, lower_data_norm_aligned,
        corrected_upper_norm, corrected_lower_norm_aligned,
        total_tvec, angle, scale, config, margin=0
    )
    
    t=plot_alignment_diff(
        upper_scan_nondesmiled_norm, lower_scan_nondesmiled_aligned,
        upper_scan_norm, corrected_lower_norm_scan_aligned,
        total_tvec, angle, scale, config, margin=0
    )
    
    print("\n" + "="*70)
    print("Frame alignment test complete!")
    print("="*70)
    
    
#%%

    total_tvec = (1,  -8)
    lower_scan_nondesmiled_aligned = ird.transform_img(lower_scan_nondesmiled_norm,tvec=total_tvec, angle=-0.09, scale=1)
    t=plot_alignment_diff(
        upper_scan_nondesmiled_norm, lower_scan_nondesmiled_aligned,
        upper_scan_norm, corrected_lower_norm_scan_aligned,
        total_tvec, angle, scale, config, margin=0
    )
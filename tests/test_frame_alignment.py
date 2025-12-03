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


def plot_spectra_vs_atlas(corrected_upper, corrected_lower, wl_start, wl_end, config, 
                          row_indices=None, atlas_key='hhdc'):
    """
    Plot spectra from multiple rows compared to the solar atlas.
    
    Parameters
    ----------
    corrected_upper : ndarray
        Corrected upper frame (spatial, spectral)
    corrected_lower : ndarray
        Corrected lower frame (spatial, spectral)
    wl_start : float
        Start wavelength in nm
    wl_end : float
        End wavelength in nm
    config : Config
        Configuration object
    row_indices : list, optional
        List of row indices to plot. Default: [low, middle, high]
    atlas_key : str
        Atlas key for atlas_factory (default: 'hhdc')
    """
    ny, nx = corrected_upper.shape
    
    # Default row indices: low, middle, high
    if row_indices is None:
        row_indices = [ny // 6, ny // 2, 5 * ny // 6]
    
    # Load the atlas
    print(f"\n4. Loading atlas ({atlas_key}) for wavelength range {wl_start:.2f} - {wl_end:.2f} nm...")
    atlas = atlas_factory(atlas_key, wl_start, wl_end, conversion=NM2ANGSTROM)
    print(f"   ✓ Atlas loaded: {len(atlas.wl)} points")
    
    # Create wavelength array for data (linear interpolation)
    wl_data = np.linspace(wl_start, wl_end, nx)
    
    # Create figure with 2 subplots (upper and lower)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Spectra vs Atlas Comparison\nWavelength: {wl_start:.2f} - {wl_end:.2f} nm', 
                 fontsize=14, fontweight='bold')
    
    colors = ['tab:blue', 'tab:green', 'tab:red']
    row_labels = ['Low', 'Middle', 'High']
    
    for ax_idx, (ax, frame_data, frame_name) in enumerate([
        (axes[0], corrected_upper, 'Upper'),
        (axes[1], corrected_lower, 'Lower')
    ]):
        ax.set_title(f'{frame_name} Frame')
        
        # Plot atlas (normalized)
        atlas_norm = atlas.intensity / np.mean(atlas.intensity)
        ax.plot(atlas.wl, atlas_norm, 'k-', linewidth=1.5, alpha=0.7, label='Atlas (HHDC)')
        
        # Plot spectra from selected rows
        for i, (row_idx, color, label) in enumerate(zip(row_indices, colors, row_labels)):
            spectrum = frame_data[row_idx, :]
            # Normalize spectrum
            spectrum_norm = spectrum / np.mean(spectrum)
            ax.plot(wl_data, spectrum_norm, color=color, linewidth=0.8, 
                    alpha=0.8, label=f'Row {row_idx} ({label})')
        
        ax.set_ylabel('Normalized Intensity')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(wl_start, wl_end)
    
    axes[1].set_xlabel('Wavelength [nm]')
    
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


def plot_results(corrected_upper, corrected_lower, shifted_lower, shift_y, shift_x, config):
    """
    Plot the corrected frames and differences.
    
    Parameters
    ----------
    corrected_upper : ndarray
        Corrected upper frame (reference)
    corrected_lower : ndarray
        Corrected lower frame (before alignment)
    shifted_lower : ndarray
        Lower frame after alignment
    shift_y, shift_x : float
        Detected shifts
    config : Config
        Configuration object
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Upper/Lower Frame Alignment Analysis\nDetected Shift: ({shift_y:.2f}, {shift_x:.2f}) px (spatial, spectral)', 
                 fontsize=14, fontweight='bold')
    
    # Row 0: Corrected frames
    ax = axes[0, 0]
    im = ax.imshow(corrected_upper, aspect='auto', cmap='viridis')
    ax.set_title('Corrected Upper Frame (Reference)')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    ax = axes[0, 1]
    im = ax.imshow(corrected_lower, aspect='auto', cmap='viridis')
    ax.set_title('Corrected Lower Frame (Before Alignment)')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    ax = axes[0, 2]
    im = ax.imshow(shifted_lower, aspect='auto', cmap='viridis')
    ax.set_title('Lower Frame (After Alignment)')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Row 1: Differences
    diff_before = corrected_lower - corrected_upper
    diff_after = shifted_lower - corrected_upper
    
    # Use same color scale for both differences
    vmax = max(np.abs(diff_before).max(), np.abs(diff_after).max())
    vmin = -vmax
    
    ax = axes[1, 0]
    im = ax.imshow(diff_before, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title('Difference: Lower - Upper (Before Alignment)')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Intensity difference')
    
    ax = axes[1, 1]
    im = ax.imshow(diff_after, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_title('Difference: Shifted Lower - Upper (After Alignment)')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im, ax=ax, label='Intensity difference')
    
    # Statistics comparison
    ax = axes[1, 2]
    ax.axis('off')
    
    stats_text = f"""
    Difference Statistics:
    
    Before Alignment:
      RMS: {np.sqrt(np.mean(diff_before**2)):.4f}
      Mean: {np.mean(diff_before):.4f}
      Std: {np.std(diff_before):.4f}
      Max: {np.max(np.abs(diff_before)):.4f}
    
    After Alignment:
      RMS: {np.sqrt(np.mean(diff_after**2)):.4f}
      Mean: {np.mean(diff_after):.4f}
      Std: {np.std(diff_after):.4f}
      Max: {np.max(np.abs(diff_after)):.4f}
    
    Improvement:
      RMS reduction: {(1 - np.sqrt(np.mean(diff_after**2))/np.sqrt(np.mean(diff_before**2)))*100:.1f}%
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures_dir = Path(config.directories.figures)
    figures_dir.mkdir(exist_ok=True, parents=True)
    output_plot = figures_dir / f'frame_alignment_test_{timestamp}.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved alignment plot: {output_plot}")
    
    plt.show()

   
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
    
    upper_data = l0_data[0]['upper'].data
    lower_data = l0_data[0]['lower'].data
    
    upper_data = upper_data/np.max(upper_data)
    lower_data = lower_data/np.max(lower_data)
    # Load dust flats from auxiliary files
    
    with fits.open(config.dataset[data_type]['files'].auxiliary.get('dust_flat_upper')) as hdu:
        dust_flat_upper = hdu[0].data
    with fits.open(config.dataset[data_type]['files'].auxiliary.get('dust_flat_lower')) as hdu:
        dust_flat_lower = hdu[0].data
    print(f"   ✓ Loaded dust flats from auxiliary files")
    
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
    
    # Extract arrays (state 0) and ROI from header before closing
    offset_upper_outdated_2d = offset_map_upper_outdated.map[0].copy()
    offset_lower_outdated_2d = offset_map_lower_outdated.map[0].copy()
    roi_upper = eval(offset_map_upper_outdated.header['ROI'])  # e.g. (slice(20, 858, None), slice(20, 1230, None))
    roi_lower = eval(offset_map_lower_outdated.header['ROI'])
    # offset_upper_amended_2d = offset_map_upper_amended.map[0].copy()
    # offset_lower_amended_2d = offset_map_lower_amended.map[0].copy()
    
    print(f"   ROI upper: {roi_upper}")
    print(f"   ROI lower: {roi_lower}")
    
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
    # Wavelength range from atlas-fit config (configs/atlas_fit_config_cam1.yml)
    wl_start = 460.5  # nm
    wl_end = 461.1    # nm
    
    plot_spectra_vs_atlas(
        corrected_upper, 
        corrected_lower, 
        wl_start, 
        wl_end, 
        config,
        row_indices=None,  # Will use default: [low, middle, high]
        atlas_key='hhdc'   # Hamburg Disk-Center atlas
    )

    # Step 5: Calculate shift
    #lower_corrected_shifted = np.roll(corrected_lower, 1,axis=0)   # FIRST ROUGH ESTIMATE - NEED AMYBE TO BE REFINED LATER
    #lower_corrected_shifted = np.roll(lower_corrected_shifted, -8, axis=1)
    
    # print("\n4. Calculating shift between frames...")
    
    #shift_y, shift_x = tt.find_shift(corrected_upper[50:-50,50:-50], corrected_lower[50:-50,50:-50]) #reference, target
    #better_shifted_lower = tt.shift_image( corrected_lower, shift_y, shift_x)
    
    #%%
    
    
    #%%
    # # Apply shift to lower frame
    # shifted_lower = apply_shift(corrected_lower, shift_y, shift_x)
    # print(f"   ✓ Applied shift to lower frame")
    
    # # Step 5: Display results
   # print("\n5. Plotting results...")
    # data_plot = np.array([
    # upper_data-lower_data,   
    # corrected_upper-corrected_lower,
    # corrected_upper-lower_corrected_shifted ,
    # corrected_upper-better_shifted_lower
    # ])
    # viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
    #                    title='scan l0', 
    #                    state_names=['l0 upper-lower','corrected upper-lower','lower_corrected_shifted', 'better shifted lower' ])
    
    # print("\n" + "="*70)
    # print("Frame alignment test complete!")
    # print("="*70)
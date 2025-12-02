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

from themis.datasets.themis_datasets_2025 import get_config
from themis.core import themis_io as tio
from spectroflat.smile import SmileInterpolator, OffsetMap
from spectroflat.utils.processing import MP
import themis.core.themis_tools as tt
from spectator.controllers.app_controller import display_data # from spectator

#%%
def apply_desmiling_spectroflat(image, offset_map_array):
    """
    Apply desmiling row-by-row using spectroflat's SmileInterpolator.desmile_row.
    Uses multiprocessing for speed.
    
    Parameters
    ----------
    image : np.ndarray
        2D image [spatial, wavelength]
    offset_map_array : np.ndarray
        2D offset map array [spatial, wavelength] for this specific state
    
    Returns
    -------
    np.ndarray
        Desmiled image [spatial, wavelength]
    """
    ny, nx = image.shape
    rows = np.arange(ny)
    xes = np.arange(nx)
    
    # Apply desmiling row by row with multiprocessing
    # Arguments: (row_index, x_coordinates, offsets_for_this_row, data_for_this_row)
    arguments = [(r, xes, offset_map_array[r], image[r]) for r in rows]
    res = dict(MP.simultaneous(SmileInterpolator.desmile_row, arguments))
    
    # Reconstruct the desmiled image from row results
    desmiled = np.array([res[row] for row in rows])
    
    return desmiled


def apply_corrections(frame_data, dust_flat, offset_map, illumination_pattern, frame_name):
    """
    Apply all flat-field corrections to a frame.
    
    Parameters
    ----------
    frame_data : ndarray
        Raw frame data (spatial, spectral)
    dust_flat : ndarray
        Dust flat correction
    offset_map : OffsetMap
        Smile correction offset map
    illumination_pattern : ndarray
        Illumination pattern correction
    frame_name : str
        'upper' or 'lower' for display
        
    Returns
    -------
    corrected : ndarray
        Fully corrected frame
    """
    print(f"\n  Applying corrections to {frame_name} frame:")
    print(f"    Input shape: {frame_data.shape}")
    
    # Step 1: Apply dust flat
    corrected = frame_data / dust_flat
    print(f"    ✓ Applied dust flat")
    
    # Step 2: De-smile using offset map
    # Need to use state 0 since offset maps have shape (n_states, spatial, spectral)
    corrector = SmileInterpolator(offset_map, corrected, mod_state=0)
    corrector.run()
    corrected = corrector.result
    print(f"    ✓ Applied smile correction")
    
    # Step 3: Apply illumination pattern (state 0)
    corrected = corrected / illumination_pattern[0]
    print(f"    ✓ Applied illumination pattern")
    
    return corrected


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
    
    # Extract arrays immediately (state 0) to allow file handles to close
    offset_upper_outdated_2d = offset_map_upper_outdated.map[0].copy()
    offset_lower_outdated_2d = offset_map_lower_outdated.map[0].copy()
    # offset_upper_amended_2d = offset_map_upper_amended.map[0].copy()
    # offset_lower_amended_2d = offset_map_lower_amended.map[0].copy()
    
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
    print(f"     Processing {upper_dust_corrected.shape[0]} rows (multiprocessing)...")
    upper_desmiled = apply_desmiling_spectroflat(upper_dust_corrected, offset_upper_outdated_2d)
    print(f"     ✓ Upper desmiled")
    lower_desmiled = apply_desmiling_spectroflat(lower_dust_corrected, offset_lower_outdated_2d)
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

    # Step 4: Calculate shift
    lower_corrected_shifted = np.roll(corrected_lower, 1,axis=0)   #NEED AMYBE TO BE REFINED LATER
    lower_corrected_shifted = np.roll(lower_corrected_shifted, -8, axis=1)
    
    # print("\n4. Calculating shift between frames...")
    
    shift_y, shift_x = tt.find_shift(corrected_upper[50:-50,50:-50], corrected_lower[50:-50,50:-50]) #reference, target
    better_shifted_lower = tt.shift_image( corrected_lower, shift_y, shift_x)
    
    #%%
    
    u_sp = corrected_upper[10:-10, 200:-200]#/illum_upper[0][10:-10, 200:-200]
    l_sp = corrected_lower[10:-10, 200:-200]#/illum_lower[0][10:-10, 200:-200]
    
    u_sp/=u_sp.max()
    l_sp/=l_sp.max()
    plt.plot(100*(u_sp[50,:]-u_sp[500,:]))
    plt.plot(100*(l_sp[50,:]-l_sp[500,:]))
    
    data_plot = np.array([(100*(corrected_upper-np.roll(corrected_upper, 50, axis=0)))[100:-100,100:-100],
                          (100*(corrected_lower-np.roll(corrected_lower, 50, axis=0)))[100:-100,100:-100]])
    viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
                       title='scan l0', 
                       state_names=['upper-shifted','lower-shifted' ])
    
    
    #%%
    # # Apply shift to lower frame
    # shifted_lower = apply_shift(corrected_lower, shift_y, shift_x)
    # print(f"   ✓ Applied shift to lower frame")
    
    # # Step 5: Display results
   # print("\n5. Plotting results...")
    data_plot = np.array([
    upper_data-lower_data,   
    corrected_upper-corrected_lower,
    corrected_upper-lower_corrected_shifted ,
    corrected_upper-better_shifted_lower
    ])
    viewer = display_data( data_plot, 'states',  'spatial', 'spectral',
                       title='scan l0', 
                       state_names=['l0 upper-lower','corrected upper-lower','lower_corrected_shifted', 'better shifted lower' ])
    
    # print("\n" + "="*70)
    # print("Frame alignment test complete!")
    # print("="*70)
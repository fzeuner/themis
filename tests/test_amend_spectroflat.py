#!/usr/bin/env python3
"""
Test script to verify amend-spectroflat functionality.
This script tests if the amend-spectroflat script runs correctly
with a sample configuration and line file.
"""
import subprocess
import sys
import tempfile
import shutil
import numpy as np
from pathlib import Path

def plot_amend_spectroflat_results(config, data_type='flat_center', frame_name='upper'):
    """
    Plot comparison of original and amended offset maps and soft flats for a single frame.
    
    Parameters
    ----------
    config : Config
        Configuration object containing dataset and directory information
    data_type : str
        Type of data being processed ('flat_center' or 'flat')
    frame_name : str
        Frame to plot ('upper' or 'lower')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.io import fits
    from pathlib import Path
    
    output_dir = Path(config.directories.reduced)
    
    # File paths for this specific frame
    original_offset_map = config.dataset[data_type]['files'].auxiliary.get(f'offset_map_{frame_name}_outdated')
    wl_calibrated_offsets = config.dataset[data_type]['files'].auxiliary.get(f'offset_map_{frame_name}')
    original_soft_flat = config.dataset[data_type]['files'].auxiliary.get(f'illumination_pattern_{frame_name}_outdated')
    amended_soft_flat = config.dataset[data_type]['files'].auxiliary.get(f'illumination_pattern_{frame_name}')
    
    # Check if all files exist
    if not all([f and f.exists() for f in [original_offset_map, wl_calibrated_offsets,
                original_soft_flat, amended_soft_flat]]):
        print(f"Error: Not all files exist for {frame_name} frame plotting")
        print(f"  offset_map_{frame_name}_outdated: {original_offset_map and original_offset_map.exists()}")
        print(f"  offset_map_{frame_name}: {wl_calibrated_offsets and wl_calibrated_offsets.exists()}")
        print(f"  illumination_pattern_{frame_name}_outdated: {original_soft_flat and original_soft_flat.exists()}")
        print(f"  illumination_pattern_{frame_name}: {amended_soft_flat and amended_soft_flat.exists()}")
        return
    
    # Read the files
    with fits.open(original_offset_map) as hdul:
        orig_offset_data = hdul[0].data
        orig_offset_header = hdul[0].header
    
    with fits.open(wl_calibrated_offsets) as hdul:
        wl_offset_data = hdul[0].data
        wl_offset_header = hdul[0].header
    
    with fits.open(original_soft_flat) as hdul:
        orig_flat_data = hdul[0].data
        orig_flat_header = hdul[0].header
    
    with fits.open(amended_soft_flat) as hdul:
        amended_flat_data = hdul[0].data
        amended_flat_header = hdul[0].header
    
    # Extract wavelength calibration info from headers (only amended files have WL info)
    has_wl_calibration = 'HIERARCH MIN_WL_NM' in wl_offset_header
    
    if has_wl_calibration:
        min_wl = wl_offset_header['HIERARCH MIN_WL_NM']
        max_wl = wl_offset_header['HIERARCH MAX_WL_NM']
        dispersion = wl_offset_header.get('HIERARCH DISPERSION', (max_wl - min_wl) / wl_offset_data.shape[-1])
    else:
        min_wl = None
        max_wl = None
        dispersion = None
    
    # Pixel extents for original (non-WL-calibrated) data
    n_wavelength_px = orig_offset_data.shape[-1]
    pixel_extent = [0, n_wavelength_px, orig_offset_data.shape[1], 0]
    
    # Wavelength extents for amended (WL-calibrated) data
    if has_wl_calibration:
        wl_extent = [min_wl, max_wl, wl_offset_data.shape[1], 0]
    else:
        wl_extent = pixel_extent
    
    # Use state 0 for plotting (all 4 states are identical for per-frame processing)
    state = 0
    
    # Create figure with 4 rows × 1 column
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(f'Amend Spectroflat Results Comparison ({frame_name.upper()} Frame)', 
                 fontsize=16, fontweight='bold')
    
    # Row 0: Original offset map (pixel space)
    ax = axes[0]
    im0 = ax.imshow(orig_offset_data[state], aspect='auto', cmap='RdBu_r', 
                     extent=pixel_extent)
    ax.set_title(f'Original Offset Map (Before WL Calibration)')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im0, ax=ax, label='Offset [px]')
    
    # Row 1: WL-calibrated offset map (wavelength space)
    ax = axes[1]
    im1 = ax.imshow(wl_offset_data[state], aspect='auto', cmap='RdBu_r',
                     extent=wl_extent)
    ax.set_title(f'WL-Calibrated Offset Map (After Amend)')
    ax.set_xlabel('Wavelength [nm]' if has_wl_calibration else 'Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im1, ax=ax, label='Offset [px]')
    
    # Row 2: Original soft flat (pixel space)
    ax = axes[2]
    im2 = ax.imshow(orig_flat_data[state], aspect='auto', cmap='viridis',
                     extent=pixel_extent)
    ax.set_title(f'Original Illumination Pattern (Before WL Calibration)')
    ax.set_xlabel('Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im2, ax=ax, label='Relative intensity')
    
    # Row 3: Amended soft flat (wavelength space)
    ax = axes[3]
    im3 = ax.imshow(amended_flat_data[state], aspect='auto', cmap='viridis',
                     extent=wl_extent)
    ax.set_title(f'Amended Illumination Pattern (After Amend)')
    ax.set_xlabel('Wavelength [nm]' if has_wl_calibration else 'Spectral [px]')
    ax.set_ylabel('Spatial [px]')
    plt.colorbar(im3, ax=ax, label='Relative intensity')
    
    # Adjust spacing: more space on top for title, more vertical space between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.0)
    
    # Save figure in the figures directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures_dir = Path(config.directories.figures)
    figures_dir.mkdir(exist_ok=True, parents=True)
    output_plot = figures_dir / f'amend_spectroflat_comparison_{frame_name}_{timestamp}.png'
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_plot}")
    
    plt.show()
    
    # Print summary statistics for this frame
    print(f"\n{'='*70}")
    print(f"Summary Statistics for {frame_name.upper()} Frame")
    print(f"{'='*70}")
    
    if has_wl_calibration:
        print(f"Wavelength range: {min_wl:.3f} - {max_wl:.3f} nm")
        print(f"Dispersion: {dispersion:.6f} nm/px")
    else:
        print(f"No wavelength calibration found in amended files")
        print(f"Spectral range: 0 - {n_wavelength_px} px")
    
    print("\nOffset map difference (RMS):")
    rms_offset = np.sqrt(np.mean((wl_offset_data[state] - orig_offset_data[state])**2))
    print(f"  {rms_offset:.4f} px")
    
    print("\nIllumination pattern difference (RMS):")
    rms_flat = np.sqrt(np.mean((amended_flat_data[state] - orig_flat_data[state])**2))
    print(f"  {rms_flat:.6f}")
    
    print(f"\nMax offset change: {np.max(np.abs(wl_offset_data[state] - orig_offset_data[state])):.4f} px")
    print(f"Mean offset change: {np.mean(wl_offset_data[state] - orig_offset_data[state]):.4f} px")


if __name__ == "__main__":
    """Test amend_spectroflat results plotting."""
    
    print("="*70)
    print("Test Amend Spectroflat - Plot Results")
    print("="*70)
    
    # Load config
    try:
        from themis.datasets.themis_datasets_2025 import get_config
        config = get_config(
            config_path='configs/sample_dataset_sr_2025-07-07.toml',
            auto_discover_files=True
        )
        
        print("\nAvailable frames: upper, lower")
        print("\nWould you like to plot the amend results?")
        print("  [u] Upper frame only")
        print("  [l] Lower frame only")
        print("  [b] Both frames")
        print("  [n] No, exit")
        
        user_input = input("\nChoice: ").strip().lower()
        
        if user_input == 'u':
            plot_amend_spectroflat_results(config, data_type='flat_center', frame_name='upper')
        elif user_input == 'l':
            plot_amend_spectroflat_results(config, data_type='flat_center', frame_name='lower')
        elif user_input == 'b':
            print("\n" + "="*70)
            print("Plotting UPPER frame...")
            print("="*70)
            plot_amend_spectroflat_results(config, data_type='flat_center', frame_name='upper')
            
            print("\n" + "="*70)
            print("Plotting LOWER frame...")
            print("="*70)
            plot_amend_spectroflat_results(config, data_type='flat_center', frame_name='lower')
        elif user_input == 'n':
            print("Exiting.")
        else:
            print("Invalid choice. Exiting.")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

